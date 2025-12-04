# pyright: enableExperimentalFeatures=true
from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, TypeVar, cast, overload

import ray
from packaging.version import Version
from ray.experimental import tqdm_ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_METRICS_LOGGER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from typing_extensions import Sentinel, TypeAliasType

from ray_utilities.callbacks.algorithm.seeded_env_callback import SeedEnvsCallback
from ray_utilities.config import DefaultArgumentParser, seed_environments_for_config
from ray_utilities.constants import ENVIRONMENT_RESULTS, RAY_METRICS_V2, RAY_VERSION, SEED, SEEDS
from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations, calculate_steps
from ray_utilities.learners import mix_learners
from ray_utilities.learners.torch_learner_with_gradient_accumulation_base import (
    TorchLearnerWithGradientAccumulationBase,
)
from ray_utilities.misc import AutoInt
from ray_utilities.nice_logger import ImportantLogger
from ray_utilities.warn import (
    warn_about_larger_minibatch_size,
    warn_if_batch_size_not_divisible,
    warn_if_minibatch_size_not_divisible,
)

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.core.rl_module import RLModuleSpec
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co, ExperimentSetupBase, ParserType_co
    from ray_utilities.typing import (
        RewardUpdaters,
        StrictAlgorithmReturnData,
    )
    from ray_utilities.typing.algorithm_return import EvalEnvRunnersResultsDict
    from ray_utilities.typing.metrics import (
        AnyLogMetricsDict,
        _LogMetricsEnvRunnersResultsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
        _NewLogMetricsEvaluationResultsDict,
    )
    from ray_utilities.typing.trainable_return import RewardUpdater


logger = logging.getLogger(__name__)

DefaultExperimentSetup = TypeAliasType(
    "DefaultExperimentSetup", "ExperimentSetupBase[DefaultArgumentParser, AlgorithmConfig, Algorithm]"
)

_AlgorithmConfigT = TypeVar("_AlgorithmConfigT", bound="AlgorithmConfig")

_NOT_FOUND = Sentinel("_NOT_FOUND")


@overload
def episode_iterator(
    args: dict[str, Any], hparams: Any, *, use_pbar: Literal[False], flush_interval: Optional[float] = 3.5
) -> range: ...


@overload
def episode_iterator(
    args: dict[str, Any], hparams: dict[Any, Any], *, use_pbar: Literal[True], flush_interval: Optional[float] = 3.5
) -> tqdm_ray.tqdm: ...


def episode_iterator(
    args: dict[str, Any], hparams: dict[str, Any], *, use_pbar: bool = True, flush_interval: Optional[float] = 3.5
) -> tqdm_ray.tqdm | range:
    """Creates an iterator for `args["iterations"]`

    Will create a `tqdm` if `use_pbar` is True, otherwise returns a range object.
    """
    if use_pbar:
        return tqdm_ray.tqdm(
            range(args["iterations"]), position=hparams.get("process_number", None), flush_interval_s=flush_interval
        )
    return range(args["iterations"])


def patch_model_config(config: AlgorithmConfig, model_config: dict[str, Any] | DefaultModelConfig):
    """Updates the config.model_config and rl_module_spec.model_config with the given model_config."""
    if dataclasses.is_dataclass(model_config):
        model_config = dataclasses.asdict(model_config)
    model_config = deepcopy(model_config)
    # We want to avoid setting auto filled keys
    if isinstance(config._model_config, dict):
        config._model_config.update(model_config)
    elif config._model_config is not None:
        config._model_config = dataclasses.asdict(config._model_config) | model_config
    else:
        config._model_config = model_config
    auto_model_config_keys = config._model_config_auto_includes.keys()
    for key in auto_model_config_keys:
        if key == "vf_share_layers":
            # This is an exception we keep. That it is autofilled is deprecated. Furthermore PPO sets it to False
            continue
        if (poped := config._model_config.pop(key, _NOT_FOUND)) is not _NOT_FOUND:
            logger.warning(
                "Removing auto-filled model_config key '%s':'%s' "
                "- should be auto filled to '%s' to allow config propagation. "
                "Config should not have auto-filled keys.",
                key,
                poped,
                model_config.get(key, "N/A"),
            )
    if config._rl_module_spec:
        if isinstance(config._rl_module_spec.model_config, dict):
            # Is normally the best to have rl_module_spec.model_config as None to allow config propagation
            if len(config._rl_module_spec.model_config) == 0:
                ImportantLogger.important_warning(
                    logger,
                    "Having an empty rl_module_spec.model_config is unexpected and is not updated by the config. "
                    "Setting it to None to allow config propagation during build. ",
                    stacklevel=2,
                )
                config._rl_module_spec.model_config = None
            else:
                config._rl_module_spec.model_config.update(model_config)
        elif config._rl_module_spec.model_config:
            config._rl_module_spec.model_config = dataclasses.asdict(config._rl_module_spec.model_config) | model_config


_learner_config_dict_update_keys = {
    "dynamic_buffer": "dynamic_buffer",
    "dynamic_batch": "dynamic_batch",
    "total_steps": "total_steps",
    "remove_masked_samples": ("not", "keep_masked_samples"),
    "min_dynamic_buffer_size": "min_step_size",
    "max_dynamic_buffer_size": "max_step_size",
    "accumulate_gradients_every": "accumulate_gradients_every",
}


def _patch_learner_config_with_param_space(
    args: dict[str, Any],
    config: AlgorithmConfig,
    *,
    hparams: dict[str, Any],
    config_inplace: bool = False,
):
    current_config_dict = config.learner_config_dict.copy() or {}
    checked_keys = set()
    for dest_key, source_key in _learner_config_dict_update_keys.items():
        checked_keys.add(source_key if isinstance(source_key, str) else source_key[1])
        if isinstance(source_key, tuple):
            # special handling for negation
            if source_key[0] == "not":
                value = hparams.get(source_key[1], _NOT_FOUND)
                if value is not _NOT_FOUND:
                    value = not value
            else:
                raise ValueError(f"Unknown special handling for learner_config_dict key {source_key}")
        else:
            value = hparams.get(source_key, _NOT_FOUND)
        if value is not _NOT_FOUND:
            current_config_dict[dest_key] = value
    if config_inplace:
        object.__setattr__(config, "learner_config_dict", current_config_dict)
    if current_config_dict:
        args["__overwritten_keys__"]["learner_config_dict"] = current_config_dict
    matching_keys_left = set(current_config_dict.keys()).intersection(hparams.keys() - checked_keys)
    if matching_keys_left:
        ImportantLogger.important_info(
            logger, "Found matching keys in hparams not applied to learner_config_dict: %s", matching_keys_left
        )


def _patch_model_config_with_param_space(
    args: dict[str, Any],
    config: _AlgorithmConfigT,
    setup_class: type[ExperimentSetupBase[Any, _AlgorithmConfigT, Any]],
    hparams_model_config: Optional[dict[str, Any]] = None,
    *,
    config_inplace: bool = False,
) -> tuple[dict[str, Any], _AlgorithmConfigT]:
    """Receives the already patched args and config and patches the model_config specifically."""
    if "model_config" in args and hparams_model_config:
        model_config_in_both = True
        # model_config is a property we do not want to set it
    else:
        model_config_in_both = False
    # Removing this blocks puts the work on the setup, kind of saver but easily pass arguments anymore from args -> model_config
    # completed_model_config = config.model_config | args
    model_config_from_args = setup_class._model_config_from_args(args)
    model_config_so_far = model_config_from_args or {}
    if hparams_model_config:
        # fill in extra keys constructed from args
        # we take priority constructed from args < hparams["model_config"]
        model_config_so_far = model_config_so_far | hparams_model_config
    if model_config_so_far:
        check_for_auto_filled_keys(model_config_so_far, config)
        patch_model_config(config, model_config_so_far)

    if model_config_in_both:
        args["__overwritten_keys__"]["model_config"] = hparams_model_config
    assert (config.minibatch_size or 0) <= config.train_batch_size_per_learner
    return args, config


def _patch_config_with_param_space(
    args: dict[str, Any],
    config: _AlgorithmConfigT,
    setup_class: Optional[type[ExperimentSetupBase[Any, _AlgorithmConfigT, Any]]] = None,  # noqa: ARG001
    *,
    hparams: dict[str, Any],
    config_inplace: bool = False,
) -> tuple[dict[str, Any], _AlgorithmConfigT]:
    auto_keys = config._model_config_auto_includes.keys()
    same_keys = set(args.keys()) & set(hparams.keys()) | (set(auto_keys) & set(hparams.keys()))
    # Also allow patching of auto-include keys
    # TODO: Allow any valid config key
    # other_keys = config.to_dict().keys() & hparams.keys()
    args["__overwritten_keys__"] = {}

    _patch_learner_config_with_param_space(config=config, args=args, hparams=hparams, config_inplace=config_inplace)
    if not same_keys and (not args["__overwritten_keys__"] or config_inplace):
        return args, config
    msg_dict = {k: f"{args[k] if k in args else 'to config'} -> {hparams[k]}" for k in same_keys}

    # Check if minibatch_scale is set (from args or hparams)
    # NOTE: Cli args priority is higher here. # XXX Why? # CRITICAL This could be wrong when we tune minibatch_scale. Test
    minibatch_scale = hparams.get("cli_args", {}).get("minibatch_scale", args.get("minibatch_scale", None))

    # Adjust minibatch_size if train_batch_size_per_learner changes
    if (
        "train_batch_size_per_learner" in same_keys
        and config.minibatch_size is not None
        and config.minibatch_size > hparams["train_batch_size_per_learner"]
    ):
        warn_about_larger_minibatch_size(
            minibatch_size=config.minibatch_size,
            train_batch_size_per_learner=hparams["train_batch_size_per_learner"],
            note_adjustment=True,
        )
        msg_dict["minibatch_size"] = f"{config.minibatch_size} -> {hparams['train_batch_size_per_learner']}"
        if config_inplace:
            object.__setattr__(config, "minibatch_size", hparams["train_batch_size_per_learner"])
        args["__overwritten_keys__"]["minibatch_size"] = hparams["train_batch_size_per_learner"]

    # Apply minibatch_scale logic if set
    elif minibatch_scale is not None and "train_batch_size_per_learner" in same_keys:
        new_batch_size = int(hparams["train_batch_size_per_learner"] * minibatch_scale)
        if config.minibatch_size != new_batch_size:
            msg_dict["minibatch_size"] = (
                f"{config.minibatch_size} -> {new_batch_size} (minibatch_scale={minibatch_scale})"
            )
            if config_inplace:
                object.__setattr__(config, "minibatch_size", new_batch_size)
            args["__overwritten_keys__"]["minibatch_size"] = new_batch_size

    logger.info("Patching args with hparams: %s", msg_dict)

    for key in same_keys:
        args[key] = hparams[key]
        if config_inplace and hasattr(config, key):
            object.__setattr__(config, key, hparams[key])
        args["__overwritten_keys__"][key] = hparams[key]
    if not args["__overwritten_keys__"]:
        logger.debug("No keys to patch in args with hparams: %s", hparams)
    elif not config_inplace:
        is_frozen = config._is_frozen
        config = cast("_AlgorithmConfigT", config.copy(copy_frozen=False))
        # NOTE: This might set attributes that to no belong to the config
        config.update_from_dict(args["__overwritten_keys__"])
        if is_frozen:
            config.freeze()
    assert (config.minibatch_size or 0) <= config.train_batch_size_per_learner
    return args, config


def _post_patch_config_with_param_space(
    args: dict[str, Any], config: AlgorithmConfig, env_seed: int | Sequence[int] | _NOT_FOUND | None
) -> None:
    """
    In-place post processing after patching config with param space.

    Should update important attributes based on the updated args
    """
    minibatch_scale = args.get("minibatch_scale", args.get("cli_args", {}).get("minibatch_scale", None))
    if minibatch_scale is not None:
        new_batch_size = int(config.train_batch_size_per_learner * minibatch_scale)
        if config.minibatch_size != new_batch_size:
            warn_about_larger_minibatch_size(
                minibatch_size=new_batch_size,
                train_batch_size_per_learner=config.train_batch_size_per_learner,
                note_adjustment=True,
            )
            logger.info(
                "Adjusting minibatch_size %d -> %d based on minibatch_scale %f",
                config.minibatch_size,
                new_batch_size,
                minibatch_scale,
            )
            args["minibatch_size"] = new_batch_size
            object.__setattr__(config, "minibatch_size", new_batch_size)
    # Change learner class if needed
    if args.get("accumulate_gradients_every", 0) > 1 or args.get("dynamic_batch", False):
        # import lazy as currently not used elsewhere
        if not issubclass(config.learner_class, TorchLearnerWithGradientAccumulationBase):
            if issubclass(config.learner_class, PPOTorchLearner):
                from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import (  # noqa: PLC0415
                    PPOTorchLearnerWithGradientAccumulation,
                )

                learner_class = PPOTorchLearnerWithGradientAccumulation
            elif issubclass(config.learner_class, DQNTorchLearner):
                from ray_utilities.learners.dqn_torch_learner_with_gradient_accumulation import (  # noqa: PLC0415
                    DQNTorchLearnerWithGradientAccumulation,
                )

                learner_class = DQNTorchLearnerWithGradientAccumulation
            else:
                ImportantLogger.important_warning(
                    logger,
                    "Detected neither a PPOTorchLearner nor DQNTorchLearner, cannot set gradient accumulation learner.",
                )
                learner_class = None
            if learner_class is not None and not issubclass(config.learner_class, learner_class):
                # need to account for mixer classes - take bases expect last
                # TODO: This is not 100% save better add correct learner class if accumulate_gradients is a tune parameter in Setup
                logger.info("Changing learner class to %s", learner_class)
                try:
                    config.learners(
                        learner_class=mix_learners(  # pyright: ignore[reportCallIssue]
                            [*config.learner_class.__bases__[:1], learner_class]
                        )
                    )
                except TypeError:
                    # Deprecated keyword
                    config.training(
                        learner_class=mix_learners(  # pyright: ignore[reportCallIssue]
                            [*config.learner_class.__bases__[:1], learner_class]
                        )  # pyright: ignore[reportCallIssue]
                    )
            else:
                logger.info(
                    "Learner class %s already is a subclass of %s. Not adjusting for gradient accumulation.",
                    config.learner_class,
                    TorchLearnerWithGradientAccumulationBase,
                )
    # Seeded environments - sequential seeds have to be set here, env_seed comes from Tuner
    if args.get("env_seeding_strategy", "sequential") == "sequential":
        # Warn if a seed is set but no env_seed is present
        if (env_seed is None or env_seed is _NOT_FOUND) and "cli_args" in args and args["cli_args"]["seed"] is not None:
            logger.warning(
                "cli_args has a seed(%d) set but env_seed is None, sequential seeding will not work. "
                "Assure that env_seed is passed as a parameter when creating the Trainable, "
                "e.g. sample env_seed by tune. Falling back to cli_args seed and env_seeding_strategy='same'.",
                args["cli_args"]["seed"],
            )
            env_seed = args["cli_args"]["seed"]
        elif env_seed is _NOT_FOUND:
            env_seed = None
        assert env_seed is not _NOT_FOUND
        seed_environments_for_config(config, env_seed)
    elif args["env_seeding_strategy"] == "same":
        # prefer seed coming from tuner
        seed_environments_for_config(config, env_seed if env_seed is not _NOT_FOUND else args["seed"])
    elif args["env_seeding_strategy"] == "constant":  # use default seed of class
        seed_environments_for_config(config, SeedEnvsCallback.env_seed)
    else:  # random
        seed_environments_for_config(config, None)

    # endregion seeding


def patch_config_with_param_space(
    args: dict[str, Any],
    config: _AlgorithmConfigT,
    setup_class: type[ExperimentSetupBase[Any, _AlgorithmConfigT, Any]],
    *,
    hparams: dict[str, Any],
    config_inplace: bool = False,
) -> tuple[dict[str, Any], _AlgorithmConfigT]:
    """Patch args and config with hyperparameters from param space.

    Args:
        args: Arguments dictionary to patch.
        config: AlgorithmConfig to patch.
        hparams: Hyperparameters from param space.
        config_inplace: Whether to modify the config in place. Defaults to False.
    Returns:
        Tuple of patched args and config.
    """
    hparams = hparams.copy()
    hparams_model_config = hparams.pop("model_config", None)
    args, config = _patch_config_with_param_space(
        args, config, hparams=hparams, config_inplace=config_inplace, setup_class=setup_class
    )
    check_for_auto_filled_keys(hparams_model_config, config)
    args_copy = args.copy()
    args_copy.pop("__overwritten_keys__")
    if args_copy or hparams_model_config:  # when we have no args there is nothing to update
        args, config = _patch_model_config_with_param_space(
            args,
            config,
            hparams_model_config=hparams_model_config,
            setup_class=setup_class,
            config_inplace=config_inplace,
        )
    _post_patch_config_with_param_space(args, config, env_seed=hparams.get("env_seed", _NOT_FOUND))
    return args, config


def check_for_auto_filled_keys(model_config: dict | Any, config: AlgorithmConfig) -> None:
    if not model_config or not isinstance(model_config, Mapping):
        # could still be a full ModelConfigDict but not much we can do then
        return
    auto_keys = config._model_config_auto_includes.keys()
    if model_config.keys() & auto_keys:
        raise RuntimeError(
            f"Model config contains auto-filled keys {model_config.keys() & auto_keys}. "
            "These should not be set manually as they are auto-filled by RLlib. "
            "Please remove them from the model_config passed to the algorithm."
        )


def get_args_and_config(
    hparams: dict,
    setup: Optional["ExperimentSetupBase[Any, ConfigType_co, Any]"] = None,
    setup_class: Optional[type["ExperimentSetupBase[Any, ConfigType_co, Any]"]] = None,
    model_config: Optional[dict[str, Any] | DefaultModelConfig] = None,
) -> tuple[dict[str, Any], ConfigType_co]:
    """
    Constructs the args and config from the given hparams, setup or setup_class.
    Either `setup` or `setup_class` must be provided, if both are provided, `setup` will be used.

    This function can be used in a trainable during tuning.

    Args:
        hparams: The hyperparameters selected for the trial from the search space from ray tune.
            Should include an `cli_args` key with the parsed arguments if `setup` is not provided.
        setup: An instance of `DefaultExperimentSetup` that contains the configuration and arguments.
        setup_class: A class of `DefaultExperimentSetup` that can be used to create the configuration
            and arguments. Ignored if `setup` is provided.
        model_config: Optional dictionary of model configuration overrides to apply.

    Returns:
        A tuple containing the parsed args (as a dict) and an AlgorithmConfig.
        If `setup` is provided, the args will be a copy of `setup.args` created with `vars`.
    """
    # region setup config
    args: dict[str, Any]
    if setup:
        # TODO: Use hparams
        args = setup.args_to_dict()
        config: ConfigType_co = setup.config.copy(copy_frozen=False)  # pyright: ignore[reportAssignmentType]
    elif setup_class:
        args = deepcopy(hparams["cli_args"])
        # TODO: this should use the parameters from the search space
        config = setup_class.config_from_args(SimpleNamespace(**args))
    else:
        raise ValueError("Either setup or setup_class must be provided.")
    completed_model_config = config.model_config
    model_config_matching_keys = completed_model_config.keys() & hparams.keys()
    if dataclasses.is_dataclass(model_config):
        model_config = dataclasses.asdict(model_config)
    if model_config_matching_keys:
        # heuristic remove config attributes - we assume config would propagate these, but keep them if they are in the model config
        model_config_matching_keys -= config.to_dict().keys() - (model_config or {}).keys()
    if model_config or model_config_matching_keys:
        # NOTE: Normally should put model parameters in hparams["model_config"]
        if model_config_matching_keys:
            logger.info(
                "hparams contains keys that belong to the AlgorithmConfig.model_config. "
                "It is recommended to put these into a subdict hparams['model_config']"
            )
        patch_model_config(config, dict(model_config or {}, **{k: hparams[k] for k in model_config_matching_keys}))

    args, config = patch_config_with_param_space(
        args,
        config,
        hparams=hparams,
        setup_class=setup_class or type(setup),  # pyright: ignore[reportArgumentType]
    )

    return args, config


def update_running_reward(new_reward: float, reward_array: list[float]) -> float:
    if not math.isnan(new_reward):
        reward_array.append(new_reward)
    return sum(reward_array[-100:]) / (min(100, len(reward_array)) or float("nan"))  # nan for 0


def create_running_reward_updater(initial_array: Optional[list[float]] = None) -> RewardUpdater:
    """
    Creates a partial function that updates the running reward.

    The partial function is stateful in their reward_array, which is initialized as an empty list if
    `initial_array` is not provided.
    """
    return cast(
        "RewardUpdater", partial(update_running_reward, reward_array=initial_array if initial_array is not None else [])
    )


@overload
def setup_trainable(
    hparams: dict[str, Any],
    setup: Optional["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"] = None,
    setup_class: Optional[type["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"]] = None,
    config_overrides: Optional[ConfigType_co | dict[str, Any]] = None,
    model_config: Optional[dict[str, Any] | DefaultModelConfig] = None,
    *,
    create_algo: Literal[False],
) -> tuple[dict[str, Any], "ConfigType_co", None, "RewardUpdaters"]: ...


@overload
def setup_trainable(
    hparams: dict[str, Any],
    setup: Optional["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"] = None,
    setup_class: Optional[type["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"]] = None,
    config_overrides: Optional[ConfigType_co | dict[str, Any]] = None,
    model_config: Optional[dict[str, Any] | DefaultModelConfig] = None,
    *,
    create_algo: Literal[True] = True,
) -> tuple[dict[str, Any], "ConfigType_co", "AlgorithmType_co", "RewardUpdaters"]: ...


def setup_trainable(
    hparams: dict[str, Any],
    setup: Optional["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"] = None,
    setup_class: Optional[type["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"]] = None,
    config_overrides: Optional[ConfigType_co | dict[str, Any]] = None,
    model_config: Optional[dict[str, Any] | DefaultModelConfig] = None,
    *,
    create_algo: bool = True,
) -> tuple[dict[str, Any], "ConfigType_co", "AlgorithmType_co | None", "RewardUpdaters"]:
    """
    Sets up the trainable by getting the args and config from the given hparams, setup or setup_class.
    Either `setup` or `setup_class` must be provided, if both are provided, `setup` will be used.

    Args:
        hparams: The hyperparameters selected for the trial from the search space from ray tune.
            Should include an `cli_args` key with the parsed arguments if `setup` is not provided.
        setup: An instance of `DefaultExperimentSetup` that contains the configuration and arguments.
        setup_class: A class of `DefaultExperimentSetup` that can be used to create the configuration
            and arguments. Ignored if `setup` is provided.
        create_algo: Will build or load from checkpoint when True (default). Pass False to skip this
            step, e.g. when the algorithm is created in a Trainable (and the one here is discarded).
        config_overrides: Optional dictionary of configuration overrides to apply.
        model_config: Optional dictionary of model configuration overrides to apply.

    Returns:
        A tuple containing the parsed args (as a dict), an AlgorithmConfig, and an Algorithm.

    Note:
        - The returned config of algorithm.config is frozen to prevent unexpected behavior this config.
        - The type of the Algorithm is determined by the `algo_class` attribute of the config.
            This is not entirely type-safe.

    See Also:
        :func:`get_args_and_config`: Helper function build the AlgorithmConfig.
    """
    args, config = get_args_and_config(
        hparams,
        setup=setup,
        setup_class=setup_class,
        model_config=model_config,
    )

    # Apply minibatch_scale setting if set in args
    minibatch_scale: float | None = args.get("minibatch_scale", None)
    if minibatch_scale is not None:
        scaled_minibatch = int(config.train_batch_size_per_learner * minibatch_scale)
        if config.minibatch_size != scaled_minibatch:
            config = config.copy(copy_frozen=False)
            config.minibatch_size = scaled_minibatch
            config.freeze()

    if config_overrides:
        if isinstance(config_overrides, AlgorithmConfig):
            config_overrides = config_overrides.to_dict()

        # Check if minibatch_scale is set and adjust minibatch_size accordingly
        if minibatch_scale is not None and "train_batch_size_per_learner" in config_overrides:
            config_overrides["minibatch_size"] = int(config_overrides["train_batch_size_per_learner"] * minibatch_scale)

        if "train_batch_size_per_learner" in config_overrides or "minibatch_size" in config_overrides:
            batch_size = config_overrides.get("train_batch_size_per_learner", config.train_batch_size_per_learner)
            minibatch_size = config_overrides.get("minibatch_size", config.minibatch_size)
            if minibatch_size > batch_size:
                warn_about_larger_minibatch_size(
                    minibatch_size=minibatch_size, train_batch_size_per_learner=batch_size, note_adjustment=True
                )
                config_overrides["minibatch_size"] = batch_size
        warn_if_batch_size_not_divisible(
            batch_size=config_overrides.get("train_batch_size_per_learner", config.train_batch_size_per_learner),
            num_envs_per_env_runner=config_overrides.get("num_envs_per_env_runner", config.num_envs_per_env_runner),
        )
        warn_if_minibatch_size_not_divisible(
            minibatch_size=config_overrides.get("minibatch_size", config.minibatch_size),
            num_envs_per_env_runner=config_overrides.get("num_envs_per_env_runner", config.num_envs_per_env_runner),
        )
        config = cast("ConfigType_co", config.update_from_dict(config_overrides))

    if "train_batch_size_per_learner" in hparams or (
        config_overrides and "train_batch_size_per_learner" in config_overrides
    ):
        # recalculate iterations and total_steps
        if isinstance(args["iterations"], AutoInt):
            new_iterations = calculate_iterations(
                dynamic_buffer=args["dynamic_buffer"],
                batch_size=config.train_batch_size_per_learner,
                total_steps=args["total_steps"],
                assure_even=not args["use_exact_total_steps"],
                min_size=args["min_step_size"],
                max_size=args["max_step_size"],
            )
            logger.info(
                "Adjusted iterations %d -> %d based on train_batch_size_per_learner %d",
                args["iterations"],
                new_iterations,
                config.train_batch_size_per_learner,
            )
            args["iterations"] = new_iterations
        else:
            logger.debug("Not adjusting iterations, as it was not an 'auto' value.")
    if create_algo:
        if not args["from_checkpoint"]:
            try:
                # new API; Note: copies config!
                algo = config.build_algo(use_copy=True)  # copy=True is default; maybe use False
            except AttributeError:
                algo = config.build()
            if algo.config and len(algo.config.model_config) > 25:
                logger.warning(
                    "Large model_config detected with %d keys. Possibility of wrong config.",
                    len(algo.config.model_config),
                )
            # NOTE in case of too much info in config.model_config (most cli args are present there),
            # Check if there is an args.to_dict or vars(args) dump into it.

        # Load from checkpoint
        elif checkpoint_loader := (
            setup or setup_class
        ):  # TODO: possibly do not load algorithm and let Trainable handle it (should be an option)
            algo = checkpoint_loader.algorithm_from_checkpoint(
                args["from_checkpoint"],
                config=config,  # pyright: ignore[reportArgumentType]
                args=args,
            )
            sync_env_runner_states_after_reload(algo)
            if config.algo_class is not None and not isinstance(algo, config.algo_class):
                logger.warning(
                    "Loaded algorithm from checkpoint is not of the expected type %s, got %s. "
                    "Check your setup class %s.algo_class.",
                    config.algo_class,
                    type(algo),
                    type(setup) if setup is not None else setup_class,
                )
        else:
            # Should not happen, is covered by checks in get_args_and_config
            logger.warning("No setup or setup_class provided, using default PPOSetup. ")
            algo = cast("Algorithm", config.algo_class).from_checkpoint(args["from_checkpoint"])
    else:
        algo = None
    reward_updaters: RewardUpdaters = {
        "running_reward": create_running_reward_updater(),
        "eval_reward": create_running_reward_updater(),
        "disc_eval_reward": create_running_reward_updater(),
    }
    config.freeze()
    return (
        args,
        config,  # NOTE: a copy of algo.config
        algo,  # pyright: ignore[reportReturnType]
        reward_updaters,
    )


def get_total_steps(args: dict[str, Any], config: "AlgorithmConfig") -> int | None:
    return (
        args.get("total_steps", None)
        if args["iterations"] == "auto" or args["use_exact_total_steps"]
        else (
            calculate_steps(
                args["iterations"],
                total_steps_default=args["total_steps"],
                min_step_size=args["min_step_size"],
                max_step_size=args["max_step_size"],
            )
            if args["dynamic_buffer"]
            else (
                config.train_batch_size_per_learner
                * max(1, config.num_learners)  # pyright: ignore[reportArgumentType]
                * args["iterations"]
            )
        )
    )


def _set_env_runner_state(
    env_runner: EnvRunner | SingleAgentEnvRunner | MultiAgentEnvRunner,
    state_ref: ray.ObjectRef,
    config_ref: ray.ObjectRef,
):
    state: dict = ray.get(state_ref)
    if COMPONENT_METRICS_LOGGER not in state:
        raise KeyError(f"State dictionary missing required key '{COMPONENT_METRICS_LOGGER}'.")
    config: AlgorithmConfig = ray.get(config_ref)
    env_runner.config = config
    env_runner.metrics.set_state(state[COMPONENT_METRICS_LOGGER])
    if hasattr(env_runner, "num_envs"):
        if 0 != env_runner.num_envs != env_runner.config.num_envs_per_env_runner:  # pyright: ignore[reportAttributeAccessIssue]
            # local env runner can have zero envs when we are remote
            ImportantLogger.important_warning(
                logger,
                "EnvRunner has %d envs, but config.num_envs_per_env_runner is %d. Recreating envs.",
                env_runner.num_envs,  # pyright: ignore[reportAttributeAccessIssue]
                env_runner.config.num_envs_per_env_runner,
            )
            env_runner.make_env()


def _clear_nan_stats(stat: dict[str, Any | list[list[float]]] | dict[str, Any | list[float]]):
    for k, v in stat.items():
        if k not in ("_hist", "_last_reduced"):
            continue
        if k == "_hist":  # change in ray 2.50 to _last_reduced
            hist: list[list[float]] = v  # pyright: ignore[reportAssignmentType]
            for i, h in enumerate(hist):
                hist[i] = [0.0 if (isinstance(x, float) and math.isnan(x)) else x for x in h]
        else:  # 2.50.0+
            red: list[float] = v  # pyright: ignore[reportAssignmentType]
            for i, r in enumerate(red):
                red[i] = 0.0 if (isinstance(r, float) and math.isnan(r)) else r
    return stat


def split_sum_stats_over_env_runners(
    struct: Any, path: tuple[str, ...] = (), parent=None, *, num_env_runners: int
) -> Any:
    """
    As sum values are aggregated over all env runners, split them evenly over the env runners
    again for every to have roughly its own metric.

    Args:
        struct: The structure to split, can be a dict or a list.
        path: private, used to track the path in the structure.
        parent: private, used to track the parent structure.
        num_env_runners: The number of env runners to split the stats over.
    """
    if num_env_runners <= 1:
        return struct  # No need to split if only one env runner
    if isinstance(struct, dict):
        return {
            k: split_sum_stats_over_env_runners(v, (*path, k), struct, num_env_runners=num_env_runners)
            for k, v in struct.items()
        }
    if isinstance(struct, list):
        return [split_sum_stats_over_env_runners(v, path, parent, num_env_runners=num_env_runners) for v in struct]
    if (
        parent is not None
        and parent["reduce"] == "sum"
        and parent["clear_on_reduce"] is False
        and (
            path[-1] == "values"
            or (path[-1] in ("_hist", "_last_reduced") and parent["window"] in (None, float("inf")))
        )
    ):
        return struct / num_env_runners
    return struct


def nan_to_zero_hist_leaves(
    struct: Any,
    path: tuple[str, ...] = (),
    parent=None,
    *,
    key: Optional[str] = "_hist" if RAY_VERSION < Version("2.50.0") else "_last_reduced",  # noqa: SIM300
    remove_all: bool = False,
    replace: Any = 0.0,
) -> Any:
    """
    With a bug in ray updating a metric with -= value, where value could be NaN,
    replace such leafs with `replace`, 0 by default.

    Also useful for testing where nan != nan.
    """
    if isinstance(struct, dict):
        return {
            k: nan_to_zero_hist_leaves(v, (*path, k), struct, key=key, remove_all=remove_all, replace=replace)
            for k, v in struct.items()
        }
    if isinstance(struct, list):
        return [
            nan_to_zero_hist_leaves(v, path, parent, key=key, remove_all=remove_all, replace=replace) for v in struct
        ]
    # Only modify if parent has "reduce" == "sum"
    if (
        path
        and (key is None or path[-1] == key)
        and (
            remove_all
            or (
                parent is not None
                and parent["reduce"] == "sum"
                and parent["clear_on_reduce"] is False
                and parent["window"] in (None, float("inf"))
            )
        )
    ):
        return replace if (isinstance(struct, float) and math.isnan(struct)) else struct
    return struct


def sync_env_runner_states_after_reload(algorithm: Algorithm) -> None:
    """
    Syncs metric states for env runners, fixing a bug with restored metrics

    See my PR: https://github.com/ray-project/ray/pull/54148
    """
    assert algorithm.learner_group is not None
    assert algorithm.metrics
    assert algorithm.config
    assert algorithm.evaluation_config
    rl_module_state = algorithm.learner_group.get_state(
        components=COMPONENT_LEARNER + "/" + COMPONENT_RL_MODULE,
        inference_only=True,
    )[COMPONENT_LEARNER]

    # Sync states, especially env_steps
    metrics_state = {"stats": {}} if RAY_METRICS_V2 else algorithm.metrics.get_state()
    # State keys are "--" joined
    env_runner_metrics_state = {
        COMPONENT_METRICS_LOGGER: {
            "stats": {
                k.removeprefix(ENV_RUNNER_RESULTS + "--"): split_sum_stats_over_env_runners(
                    nan_to_zero_hist_leaves(v), num_env_runners=algorithm.config.num_env_runners or 1
                )
                for k, v in metrics_state["stats"].items()
                if k.startswith(ENV_RUNNER_RESULTS + "--")
            }
        }
    }
    # Do not sync EnvRunner seeds here as they are already set (or will be reset)
    env_runner_metrics_state[COMPONENT_METRICS_LOGGER]["stats"].pop(
        f"{ENVIRONMENT_RESULTS}--{SEEDS}--seed_sequence", None
    )
    env_runner_metrics_state[COMPONENT_METRICS_LOGGER]["stats"].pop(
        f"{ENVIRONMENT_RESULTS}--{SEED}--initial_seed", None
    )
    eval_stats = {
        k.removeprefix(EVALUATION_RESULTS + "--" + ENV_RUNNER_RESULTS + "--"): split_sum_stats_over_env_runners(
            nan_to_zero_hist_leaves(v), num_env_runners=algorithm.config.num_env_runners or 1
        )
        for k, v in metrics_state["stats"].items()
        if k.startswith("--".join((EVALUATION_RESULTS, ENV_RUNNER_RESULTS)))
    }
    eval_runner_metrics = {COMPONENT_METRICS_LOGGER: {"stats": eval_stats}}

    if algorithm.env_runner_group:
        env_steps_sampled = algorithm.metrics.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=0)
        # Helps with syncinc JAX models and non standard modules.
        # If not remote runners will sync the local one just like this
        if algorithm.env_runner_group.local_env_runner and algorithm.env_runner_group.num_healthy_remote_workers() > 0:
            cast("SingleAgentEnvRunner", algorithm.env_runner_group.local_env_runner).set_state(
                {
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: env_steps_sampled,
                    **(rl_module_state or {}),
                }
            )
        algorithm.env_runner_group.sync_env_runner_states(
            config=algorithm.config,
            rl_module_state=rl_module_state,
            env_steps_sampled=env_steps_sampled,
            env_to_module=algorithm.env_to_module_connector,
            module_to_env=algorithm.module_to_env_connector,
            # env_runner_metrics=env_runner_metrics,
        )
        # Sync metrics
        state_ref = ray.put(env_runner_metrics_state)
        config_ref = ray.put(algorithm.config)

        algorithm.env_runner_group.foreach_env_runner(
            partial(_set_env_runner_state, state_ref=state_ref, config_ref=config_ref),
            remote_worker_ids=None,  # pyright: ignore[reportArgumentType]
            local_env_runner=True,
            # kwargs is not save to use here, not used on all code paths
            timeout_seconds=0.0,  # This is a state update -> Fire-and-forget.
        )

    if algorithm.eval_env_runner_group:  # XXX Why elif here in RLLib code?
        env_steps_sampled = algorithm.metrics.peek(
            (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=0
        )
        if (
            algorithm.eval_env_runner_group.local_env_runner
            and algorithm.eval_env_runner_group.num_healthy_remote_workers() > 0
        ):
            cast("SingleAgentEnvRunner", algorithm.eval_env_runner_group.local_env_runner).set_state(
                {
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: env_steps_sampled,
                    **(rl_module_state or {}),
                }
            )
        algorithm.eval_env_runner_group.sync_env_runner_states(
            config=algorithm.evaluation_config,
            # NOTE: Ray does not use EVALUATION_RESULTS here!
            rl_module_state=rl_module_state,
            env_steps_sampled=env_steps_sampled,
            env_to_module=algorithm.env_to_module_connector,
            module_to_env=algorithm.module_to_env_connector,
            # env_runner_metrics=env_runner_metrics,
        )
        if algorithm.eval_env_runner_group.local_env_runner:
            cast("SingleAgentEnvRunner", algorithm.eval_env_runner_group.local_env_runner).set_state(rl_module_state)
        if eval_stats:
            eval_state_ref = ray.put(eval_runner_metrics)
            eval_config_ref = ray.put(algorithm.config.get_evaluation_config_object())
            algorithm.eval_env_runner_group.foreach_env_runner(
                partial(_set_env_runner_state, state_ref=eval_state_ref, config_ref=eval_config_ref),
                remote_worker_ids=None,  # pyright: ignore[reportArgumentType]
                local_env_runner=False,
                # kwargs is not save to use here, not used on all code paths
                timeout_seconds=0.0,  # This is a state update -> Fire-and-forget.
            )


def make_divisible(a: int, b: int | None) -> int:
    """Make a divisible by b"""
    if b is not None and a % b != 0:
        a = (a // b + 1) * b
    return a


def is_algorithm_callback_added(config: AlgorithmConfig, callback_class: type[RLlibCallback]) -> bool:
    return (
        config.callbacks_class is callback_class
        or (
            isinstance(callback_class, partial)
            and (
                config.callbacks_class is callback_class.func
                or (isinstance(config.callbacks_class, partial) and config.callbacks_class.func is callback_class.func)
            )
        )
        or (isinstance(config.callbacks_class, type) and issubclass(config.callbacks_class, callback_class))
        or (isinstance(config.callbacks_class, (list, tuple)) and callback_class in config.callbacks_class)
    )


def get_training_results(
    result: StrictAlgorithmReturnData | AnyLogMetricsDict | dict[str, Any],
) -> _LogMetricsEnvRunnersResultsDict:
    """
    Returns the training / env_runner results from the given result dict.

    This method is agnostic to RAY_UTILITIES_NEW_LOG_METRICS and works with both
    new and rllib metrics structure
    """
    if "training" in result:
        return result["training"]
    return result[ENV_RUNNER_RESULTS]


def get_evaluation_results(
    result: StrictAlgorithmReturnData | AnyLogMetricsDict | dict[str, Any],
) -> None | EvalEnvRunnersResultsDict | _LogMetricsEvalEnvRunnersResultsDict | _NewLogMetricsEvaluationResultsDict:
    if EVALUATION_RESULTS not in result:
        return None
    eval_results = result[EVALUATION_RESULTS]
    if ENV_RUNNER_RESULTS in eval_results:
        return eval_results[ENV_RUNNER_RESULTS]
    return eval_results


def rebuild_learner_group(algorithm: Algorithm, *, load_current_learner_state: bool = True) -> None:
    """
    Rebuilds the learner group of the given algorithm.

    This can be useful when the learner group needs to be rebuilt, e.g. after changing
    the config or the RL module.

    Assumes that the current env_runner_group returns the correct spaces.

    Args:
        algorithm: The algorithm to rebuild the learner group for.
    """
    assert algorithm.config
    spaces = {
        INPUT_ENV_SPACES: (
            algorithm.config.observation_space,
            algorithm.config.action_space,
        )
    }
    # env_runner_groups might be in an old state if spaces change due to config changes
    if algorithm.env_runner_group:
        spaces.update(algorithm.env_runner_group.get_spaces())
    elif algorithm.eval_env_runner_group:
        spaces.update(algorithm.eval_env_runner_group.get_spaces())
    else:
        spaces.update(
            {
                DEFAULT_MODULE_ID: (
                    algorithm.config.observation_space,
                    algorithm.config.action_space,
                ),
            }
        )

    assert algorithm.config is not None
    module_spec = algorithm.config.get_multi_rl_module_spec(
        spaces=spaces,  # pyright: ignore[reportArgumentType]
        inference_only=False,
    )
    current_state = algorithm.learner_group.get_state() if algorithm.learner_group is not None else None
    algorithm.learner_group = algorithm.config.build_learner_group(rl_module_spec=module_spec)
    if load_current_learner_state and current_state is not None:
        # Replace old module_config with new config in state / or pop?
        specs = module_spec.rl_module_specs
        rl_spec: RLModuleSpec
        if isinstance(specs, dict):
            for module_id, rl_spec in specs.items():
                if (
                    module_id in (rl_state := current_state["learner"]["rl_module"])
                    and "model_config" in rl_state[module_id]
                ):
                    rl_state[module_id]["model_config"] = rl_spec.model_config
        else:
            current_state["learner"]["rl_module"]["model_config"] = specs.model_config
        algorithm.learner_group.set_state(current_state)

    # Check if there are modules to load from the `module_spec`.
    rl_module_ckpt_dirs = {}
    module_specs = module_spec.rl_module_specs
    # Deprecated
    multi_rl_module_ckpt_dir = module_spec.load_state_path
    modules_to_load = module_spec.modules_to_load
    if not isinstance(module_specs, dict):
        # Very unlikely. Would rise AttributeError on ray if this actually were a dict.
        module_specs = {DEFAULT_MODULE_ID: module_specs}
    for module_id, sub_module_spec in module_specs.items():
        if sub_module_spec.load_state_path:
            rl_module_ckpt_dirs[module_id] = sub_module_spec.load_state_path
    # Deprecated
    if multi_rl_module_ckpt_dir or rl_module_ckpt_dirs:
        algorithm.learner_group.load_module_state(
            multi_rl_module_ckpt_dir=multi_rl_module_ckpt_dir,
            modules_to_load=modules_to_load,
            rl_module_ckpt_dirs=rl_module_ckpt_dirs,
        )


def get_algorithm_callback_states(algorithm: Algorithm) -> dict[str, dict[str, Any]]:
    """
    Returns the states of all algorithm callbacks as a dictionary.

    Args:
        algorithm: The algorithm to get the callback states from.
    Returns:
        A dictionary mapping callback names to their state dictionaries.
    """
    if not algorithm.callbacks:
        return {}
    algorithm_callback_states = {}
    # case single class
    if isinstance(algorithm.callbacks, RLlibCallback) and hasattr(algorithm.callbacks, "get_state"):
        algorithm_callback_states[type(algorithm.callbacks).__name__] = algorithm.callbacks.get_state()  # pyright: ignore[reportAttributeAccessIssue]
    elif isinstance(algorithm.callbacks, (list, tuple)):
        for callback in algorithm.callbacks:
            if hasattr(callback, "get_state"):
                algorithm_callback_states[type(callback).__name__] = callback.get_state()  # pyright: ignore[reportAttributeAccessIssue]
    return algorithm_callback_states


def set_algorithm_callback_states(algorithm: Algorithm, callback_states: dict[str, dict[str, Any]]) -> None:
    """
    Sets the states of all algorithm callbacks from the given state dictionary.

    Args:
        algorithm: The algorithm to set the callback states for.
        callback_states: A dictionary mapping callback names to their state dictionaries.
    """
    if not algorithm.callbacks:
        return
    # case single class
    names = set(callback_states.keys())
    if isinstance(algorithm.callbacks, RLlibCallback) and hasattr(algorithm.callbacks, "set_state"):
        callback_name = type(algorithm.callbacks).__name__
        if callback_name in callback_states:
            algorithm.callbacks.set_state(callback_states[callback_name])  # pyright: ignore[reportAttributeAccessIssue]
            names.discard(callback_name)
    elif isinstance(algorithm.callbacks, (list, tuple)):
        for callback in algorithm.callbacks:
            callback_name = type(callback).__name__
            if callback_name in callback_states and hasattr(callback, "set_state"):
                callback.set_state(callback_states[callback_name])  # pyright: ignore[reportAttributeAccessIssue]
                names.discard(callback_name)
    if names:
        logger.warning(
            "Could not set states for callbacks with names: %s. They are not present in the algorithm callbacks.",
            names,
        )
