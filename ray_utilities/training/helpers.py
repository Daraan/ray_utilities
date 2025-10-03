# pyright: enableExperimentalFeatures=true
from __future__ import annotations

import dataclasses
import logging
import math
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, cast, overload

import ray
from ray.experimental import tqdm_ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_METRICS_LOGGER, COMPONENT_RL_MODULE
from ray.rllib.utils.metrics import (
    ALL_MODULES,  # pyright: ignore[reportPrivateImportUsage]
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    LEARNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from typing_extensions import Sentinel, TypeAliasType

from ray_utilities.callbacks.algorithm.seeded_env_callback import SeedEnvsCallback
from ray_utilities.config import seed_environments_for_config
from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations, calculate_steps
from ray_utilities.misc import AutoInt
from ray_utilities.warn import (
    warn_about_larger_minibatch_size,
    warn_if_batch_size_not_divisible,
    warn_if_minibatch_size_not_divisible,
)

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co, ExperimentSetupBase, ParserType_co
    from ray_utilities.typing import (
        RewardUpdaters,
        StrictAlgorithmReturnData,
    )
    from ray_utilities.typing.algorithm_return import EvalEnvRunnersResultsDict
    from ray_utilities.typing.metrics import (
        AnyLogMetricsDict,
        LogMetricsDict,
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
def episode_iterator(args: dict[str, Any], hparams: Any, *, use_pbar: Literal[False]) -> range: ...


@overload
def episode_iterator(args: dict[str, Any], hparams: dict[Any, Any], *, use_pbar: Literal[True]) -> tqdm_ray.tqdm: ...


def episode_iterator(args: dict[str, Any], hparams: dict[str, Any], *, use_pbar: bool = True) -> tqdm_ray.tqdm | range:
    """Creates an iterator for `args["iterations"]`

    Will create a `tqdm` if `use_pbar` is True, otherwise returns a range object.
    """
    if use_pbar:
        return tqdm_ray.tqdm(range(args["iterations"]), position=hparams.get("process_number", None))
    return range(args["iterations"])


def get_current_step(result: StrictAlgorithmReturnData | LogMetricsDict | dict[str, Any] | MetricsLogger) -> int:
    # requires exact_sampling_callback to be set in the results, otherwise fallback
    if isinstance(result, MetricsLogger):
        # For peek do not use None as default as this triggers KeyError
        current_step = result.peek((LEARNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), default=_NOT_FOUND)
        if current_step is not _NOT_FOUND:
            return current_step
        current_step = result.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), default=_NOT_FOUND)
        if current_step is not _NOT_FOUND:
            return current_step
        return result.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=0)

    current_step = result.get("current_step")  # likely not present
    if current_step is not None:  # LogMetricsDict
        return current_step
    result = cast("StrictAlgorithmReturnData", result)
    if LEARNER_RESULTS in result:
        current_step = result[LEARNER_RESULTS][ALL_MODULES].get(NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME)
        if current_step is not None:
            return current_step
    # try metric logged on env runner; else defaults to NUM_ENV_STEPS_SAMPLED_LIFETIME
    return int(
        result[ENV_RUNNER_RESULTS].get(
            NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME, result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME]
        )
    )


def patch_model_config(config: AlgorithmConfig, model_config: dict[str, Any] | DefaultModelConfig):
    """Updates the config.model_config and rl_module_spec.model_config with the given model_config."""
    if dataclasses.is_dataclass(model_config):
        model_config = dataclasses.asdict(model_config)
    model_config = deepcopy(model_config)
    if isinstance(config._model_config, dict):
        config._model_config.update(model_config)
    elif config._model_config is not None:
        config._model_config = dataclasses.asdict(config._model_config) | model_config
    else:
        config._model_config = model_config
    if config._rl_module_spec:
        if config._rl_module_spec.model_config is None:
            config._rl_module_spec.model_config = config._model_config
        elif isinstance(config._rl_module_spec.model_config, dict):
            config._rl_module_spec.model_config.update(model_config)
        else:
            config._rl_module_spec.model_config = dataclasses.asdict(config._rl_module_spec.model_config) | model_config


def _patch_with_param_space(
    args: dict[str, Any],
    config: _AlgorithmConfigT,
    *,
    hparams: dict[str, Any],
    config_inplace: bool = False,
) -> tuple[dict[str, Any], _AlgorithmConfigT]:
    same_keys = set(args.keys()) & set(hparams.keys())
    args["__overwritten_keys__"] = {}
    if "model_config" in hparams:
        patch_model_config(config, hparams["model_config"])
    if not same_keys:
        logger.debug("No keys to patch in args with hparams: %s", hparams)
        return args, config
    msg_dict = {k: f"{args[k]} -> {hparams[k]}" for k in same_keys}
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

    logger.info("Patching args with hparams: %s", msg_dict)
    for key in same_keys:
        args[key] = hparams[key]
        if config_inplace:
            object.__setattr__(config, key, hparams[key])
        args["__overwritten_keys__"][key] = hparams[key]
    if not config_inplace:
        is_frozen = config._is_frozen
        config = cast("_AlgorithmConfigT", config.copy(copy_frozen=False))
        config.update_from_dict(args["__overwritten_keys__"])
        if is_frozen:
            config.freeze()
    assert config.minibatch_size or 0 <= config.train_batch_size_per_learner
    return args, config


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
    if model_config is not None:
        patch_model_config(config, model_config)
    args, config = _patch_with_param_space(args, config, hparams=hparams)
    # endregion

    # region seeding

    env_seed = hparams.get("env_seed", None)
    # Seeded environments - sequential seeds have to be set here, env_seed comes from Tuner
    if args["env_seeding_strategy"] == "sequential":
        # Warn if a seed is set but no env_seed is present
        if env_seed is None and "cli_args" in hparams and hparams["cli_args"]["seed"] is not None:
            logger.warning(
                "cli_args has a seed(%d) set but env_seed is None, sequential seeding will not work. "
                "Assure that env_seed is passed as a parameter when creating the Trainable, "
                "e.g. sample env_seed by tune. Falling back to cli_args seed and env_seeding_strategy='same'.",
                hparams["cli_args"]["seed"],
            )
            env_seed = hparams["cli_args"]["seed"]
        seed_environments_for_config(config, env_seed)
    elif args["env_seeding_strategy"] == "same":
        seed_environments_for_config(config, args["seed"])
    elif args["env_seeding_strategy"] == "constant":
        seed_environments_for_config(config, SeedEnvsCallback.env_seed)
    else:  # random
        seed_environments_for_config(config, None)

    # endregion seeding

    return args, config


def update_running_reward(new_reward: float, reward_array: list[float]) -> float:
    if not math.isnan(new_reward):
        reward_array.append(new_reward)
    running_reward = sum(reward_array[-100:]) / (min(100, len(reward_array)) or float("nan"))  # nan for 0
    return running_reward


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
    if config_overrides:
        if isinstance(config_overrides, AlgorithmConfig):
            config_overrides = config_overrides.to_dict()
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
            # FIXME too much info in model_config (most cli args)

        # Load from checkpoint
        elif checkpoint_loader := (
            setup or setup_class
        ):  # TODO: possibly do not load algorithm and let Trainable handle it (should be an option)
            algo = checkpoint_loader.algorithm_from_checkpoint(args["from_checkpoint"], config=config)
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
            logger.error(
                "EnvRunner has %d envs, but config.num_envs_per_env_runner is %d. Recreating envs.",
                env_runner.num_envs,  # pyright: ignore[reportAttributeAccessIssue]
                env_runner.config.num_envs_per_env_runner,
            )
            env_runner.make_env()


def _clear_nan_stats(stat: dict[str, Any | list[list[float]]]):
    for k, v in stat.items():
        if k != "_hist":
            continue
        hist: list[list[float]] = v
        for i, h in enumerate(hist):
            hist[i] = [0.0 if (isinstance(x, float) and math.isnan(x)) else x for x in h]
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
    if parent is not None and parent["reduce"] == "sum" and parent["clear_on_reduce"] is False:
        if path[-1] == "values" or (path[-1] == "_hist" and parent["window"] in (None, float("inf"))):
            return struct / num_env_runners
    return struct


def nan_to_zero_hist_leaves(
    struct: Any,
    path: tuple[str, ...] = (),
    parent=None,
    *,
    key: Optional[str] = "_hist",
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
    if path and (key is None or path[-1] == key):
        # Only modify if parent has "reduce" == "sum"
        if remove_all or (
            parent is not None
            and parent["reduce"] == "sum"
            and parent["clear_on_reduce"] is False
            and parent["window"] in (None, float("inf"))
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

    metrics_state = algorithm.metrics.get_state()
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
    env_runner_metrics_state[COMPONENT_METRICS_LOGGER]["stats"].pop("environments--seeds--seed_sequence", None)
    eval_stats = {
        k.removeprefix(EVALUATION_RESULTS + "--" + ENV_RUNNER_RESULTS + "--"): split_sum_stats_over_env_runners(
            nan_to_zero_hist_leaves(v), num_env_runners=algorithm.config.num_env_runners or 1
        )
        for k, v in metrics_state["stats"].items()
        if k.startswith("--".join((EVALUATION_RESULTS, ENV_RUNNER_RESULTS)))
    }
    eval_runner_metrics = {COMPONENT_METRICS_LOGGER: {"stats": eval_stats}}

    if algorithm.env_runner_group:
        algorithm.env_runner_group.sync_env_runner_states(
            config=algorithm.config,
            env_steps_sampled=algorithm.metrics.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=0),
            rl_module_state=rl_module_state,
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
        algorithm.eval_env_runner_group.sync_env_runner_states(
            config=algorithm.evaluation_config,
            # NOTE: Ray does not use EVALUATION_RESULTS here!
            env_steps_sampled=algorithm.metrics.peek(
                (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=0
            ),
            rl_module_state=rl_module_state,
            env_to_module=algorithm.env_to_module_connector,
            module_to_env=algorithm.module_to_env_connector,
            # env_runner_metrics=env_runner_metrics,
        )
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
