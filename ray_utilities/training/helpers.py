# pyright: enableExperimentalFeatures=true
from __future__ import annotations

import logging
import math
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, overload

from ray.experimental import tqdm_ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.utils.metrics import (
    ALL_MODULES,  # pyright: ignore[reportPrivateImportUsage]
    ENV_RUNNER_RESULTS,
    LEARNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from typing_extensions import TypeAliasType

from ray_utilities.config import seed_environments_for_config
from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_steps

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm

    from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co, ExperimentSetupBase, ParserType_co
    from ray_utilities.typing import (
        RewardUpdaters,
        StrictAlgorithmReturnData,
    )
    from ray_utilities.typing.metrics import LogMetricsDict
    from ray_utilities.typing.trainable_return import RewardUpdater

logger = logging.getLogger(__name__)

DefaultExperimentSetup = TypeAliasType(
    "DefaultExperimentSetup", "ExperimentSetupBase[DefaultArgumentParser, AlgorithmConfig, Algorithm]"
)


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


def get_current_step(result: StrictAlgorithmReturnData | LogMetricsDict) -> int:
    # requires exact_sampling_callback to be set in the results, otherwise fallback
    current_step = result.get("current_step")
    if current_step is not None:  # LogMetricsDict
        return current_step
    result = cast("StrictAlgorithmReturnData", result)
    current_step = result[LEARNER_RESULTS][ALL_MODULES].get(NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME)
    if current_step is not None:
        return current_step
    # try metric logged on env runner; else defaults to NUM_ENV_STEPS_SAMPLED_LIFETIME
    return result[ENV_RUNNER_RESULTS].get(
        NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME, result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME]
    )


def get_args_and_config(
    hparams: dict,
    setup: Optional["ExperimentSetupBase[Any, ConfigType_co, Any]"] = None,
    setup_class: Optional[type["ExperimentSetupBase[Any, ConfigType_co, Any]"]] = None,
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

    Returns:
        A tuple containing the parsed args (as a dict) and an AlgorithmConfig.
        If `setup` is provided, the args will be a copy of `setup.args` created with `vars`.
    """
    # region setup config
    args: dict[str, Any]
    if setup:
        # TODO: Use hparams
        args = setup.args_to_dict()
        config = setup.config.copy(copy_frozen=False)
    elif setup_class:
        args = hparams["cli_args"]
        # TODO: this should use the parameters from the search space
        config = setup_class.config_from_args(SimpleNamespace(**args))
    else:
        raise ValueError("Either setup or setup_class must be provided.")
    # endregion
    # region seeding
    if (run_seed := hparams.get("run_seed", None)) is not None:
        logger.debug("Using run_seed for config.seed %s", run_seed)
        config.debugging(seed=run_seed)
    # Seeded environments - sequential seeds have to be set here, run_seed comes from Tuner
    if args["env_seeding_strategy"] == "sequential":
        seed_environments_for_config(config, run_seed)
    elif args["env_seeding_strategy"] == "same":
        seed_environments_for_config(config, args["seed"])
    elif args["env_seeding_strategy"] == "constant":
        seed_environments_for_config(config, 0)
    else:
        seed_environments_for_config(config, None)
    # endregion
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


def setup_trainable(
    hparams: dict[str, Any],
    setup: Optional["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"] = None,
    setup_class: Optional[type["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"]] = None,
    overwrite_config: Optional[ConfigType_co | dict[str, Any]] = None,
) -> tuple[dict[str, Any], "ConfigType_co", "AlgorithmType_co", "RewardUpdaters"]:
    """
    Sets up the trainable by getting the args and config from the given hparams, setup or setup_class.
    Either `setup` or `setup_class` must be provided, if both are provided, `setup` will be used.

    Args:
        hparams: The hyperparameters selected for the trial from the search space from ray tune.
            Should include an `cli_args` key with the parsed arguments if `setup` is not provided.
        setup: An instance of `DefaultExperimentSetup` that contains the configuration and arguments.
        setup_class: A class of `DefaultExperimentSetup` that can be used to create the configuration
            and arguments. Ignored if `setup` is provided.

    Returns:
        A tuple containing the parsed args (as a dict), an AlgorithmConfig, and an Algorithm.

        Note:
            - The returned config of algorithm.config, to prevent unexpected behavior this config
              object is frozen.
            - The type of the Algorithm is determined by the `algo_class` attribute of the config.
            This is not entirely type-safe.
    """
    args, config = get_args_and_config(
        hparams,
        setup=setup,
        setup_class=setup_class,
    )
    if overwrite_config:
        if isinstance(overwrite_config, AlgorithmConfig):
            overwrite_config = overwrite_config.to_dict()
        config = config.update_from_dict(overwrite_config)
    if not args["from_checkpoint"]:
        try:
            # new API; Note: copies config!
            algo = config.build_algo(use_copy=True)  # copy=True is default; maybe use False
        except AttributeError:
            algo = config.build()
    # Load from checkpoint
    elif checkpoint_loader := (setup or setup_class):
        algo = checkpoint_loader.algorithm_from_checkpoint(args["from_checkpoint"])
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
        if args["iterations"] == "auto"
        else calculate_steps(
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
