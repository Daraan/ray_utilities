# pyright: enableExperimentalFeatures=true
from __future__ import annotations

import logging
import tempfile
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, TypeVar, cast, overload

from ray import tune
from ray.experimental import tqdm_ray
from ray.rllib.utils.metrics import (
    ALL_MODULES,  # pyright: ignore[reportPrivateImportUsage]
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    LEARNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from typing_extensions import TypeAliasType

from ray_utilities.callbacks.progress_bar import update_pbar
from ray_utilities.config import seed_environments_for_config
from ray_utilities.config.typed_argument_parser import LOG_STATS, DefaultArgumentParser, LogStatsChoices
from ray_utilities.constants import EVALUATED_THIS_STEP, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_steps
from ray_utilities.misc import is_pbar
from ray_utilities.postprocessing import (
    create_log_metrics,
    create_running_reward_updater,
    filter_metrics,
)
from ray_utilities.postprocessing import (
    verify_return as verify_return_type,
)
from ray_utilities.typing import RewardsDict, TrainableReturnData

if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.rllib.algorithms import Algorithm, AlgorithmConfig

    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co, ExperimentSetupBase, ParserType_co
    from ray_utilities.typing import LogMetricsDict, StrictAlgorithmReturnData, RewardUpdaters

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


def get_current_step(result: StrictAlgorithmReturnData) -> int:
    # requires exact_sampling_callback to be set in the results, otherwise fallback
    current_step = result[LEARNER_RESULTS][ALL_MODULES].get(NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME)
    if current_step is None:
        return result[ENV_RUNNER_RESULTS].get(NUM_ENV_STEPS_SAMPLED_LIFETIME)
    return current_step


def create_default_trainable(
    *,
    use_pbar: bool = True,
    discrete_eval: bool = False,
    setup: Optional[DefaultExperimentSetup] = None,
    setup_class: Optional[type[DefaultExperimentSetup]] = None,
    disable_report: bool = False,
    # Keywords not for default_trainable
    verify_return: bool = True,
) -> Callable[[dict[str, Any]], TrainableReturnData]:
    """
    Creates a wrapped `default_trainable` function with the given parameters.

    The resulting Callable only accepts one positional argument, `hparams`,
    which is the hyperparameters selected for the trial from the search space from ray tune.

    Args:
        verify_return: Whether to verify the return of the trainable function.
    """
    assert setup or setup_class, "Either setup or setup_class must be provided."
    trainable = partial(
        default_trainable,
        use_pbar=use_pbar,
        discrete_eval=discrete_eval,
        setup=setup,
        setup_class=setup_class,
        disable_report=disable_report,
    )
    if verify_return:
        return verify_return_type(TrainableReturnData)(trainable)
    return wraps(default_trainable)(trainable)


def get_args_and_config(
    hparams: dict,
    setup: Optional["ExperimentSetupBase[Any, ConfigType_co, Any]"] = None,
    setup_class: Optional[type["ExperimentSetupBase[Any, ConfigType_co, Any]"]] = None,
) -> tuple[dict[str, Any] | Any, ConfigType_co]:
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
    if setup:
        # TODO: Use hparams
        args = setup.args
        args = vars(args).copy()
        config = setup.config
    elif setup_class:
        args = hparams["cli_args"]
        # TODO: this should use the parameters from the search space
        config = setup_class.config_from_args(args)
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


def setup_trainable(
    hparams: dict[str, Any],
    setup: Optional["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"] = None,
    setup_class: Optional[type["ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]"]] = None,
) -> tuple[dict[str, Any] | Any, "ConfigType_co", "AlgorithmType_co", "RewardUpdaters"]:
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
            The type of the Algorithm is determined by the `algo_class` attribute of the config.
            This is not entirely type-safe.
    """
    args, config = get_args_and_config(
        hparams,
        setup=setup,
        setup_class=setup_class,
    )
    if not args["from_checkpoint"]:
        try:
            # new API
            algo = config.build_algo()
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
    return (
        args,
        config,
        algo,  # pyright: ignore[reportReturnType]
        reward_updaters,
    )

def training_step(
    algo: Algorithm,
    reward_updaters: RewardUpdaters,
    *,
    discrete_eval: bool = False,
    disable_report: bool = False,
    log_stats: LogStatsChoices = "minimal",
) -> tuple["StrictAlgorithmReturnData", "LogMetricsDict", "RewardsDict"]:
    # Prevent unbound variables
    metrics: TrainableReturnData | LogMetricsDict = {}  # type: ignore[assignment]
    disc_eval_mean = None
    disc_running_eval_reward = None
    # Train and get results
    result = cast("StrictAlgorithmReturnData", algo.train())

    # Reduce to key-metrics
    metrics = create_log_metrics(result, discrete_eval=discrete_eval, log_stats=log_stats)
    # Possibly use if train.get_context().get_local/global_rank() == 0 to save videos
    # Unknown if should save video here and clean from metrics or save in a callback later is faster.

    # Training
    train_reward = metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
    running_reward = reward_updaters["running_reward"](train_reward)

    # Evaluation:
    eval_mean = metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
    running_eval_reward = reward_updaters["eval_reward"](eval_mean)

    # Discrete rewards:
    if "discrete" in metrics[EVALUATION_RESULTS]:
        disc_eval_mean = metrics[EVALUATION_RESULTS]["discrete"][  # pyright: ignore[reportTypedDictNotRequiredAccess]
            ENV_RUNNER_RESULTS
        ][EPISODE_RETURN_MEAN]
        assert "disc_eval_reward" in reward_updaters
        disc_running_eval_reward = reward_updaters["disc_eval_reward"](disc_eval_mean)

    # Checkpoint
    report_metrics = cast("dict[str, Any]", metrics)  # satisfy train.report
    if (
        not disable_report
        and (EVALUATION_RESULTS in result and result[EVALUATION_RESULTS].get(EVALUATED_THIS_STEP, False))
        and False
        # and tune.get_context().get_world_rank() == 0 # deprecated
    ):
        with tempfile.TemporaryDirectory() as tempdir:
            algo.save_checkpoint(tempdir)
            tune.report(metrics=report_metrics, checkpoint=tune.Checkpoint.from_directory(tempdir))
    # Report metrics
    elif not disable_report:
        try:
            tune.report(report_metrics, checkpoint=None)
        except AttributeError:
            import ray.train

            ray.train.report(report_metrics, checkpoint=None)
    rewards: RewardsDict = {
        "running_reward": running_reward,
        "running_eval_reward": running_eval_reward,
        "eval_mean": eval_mean,
        "disc_eval_mean": disc_eval_mean or 0,
        "disc_eval_reward": disc_running_eval_reward or 0,
    }
    return result, metrics, rewards


def default_trainable(
    hparams: dict[str, Any],
    *,
    use_pbar: bool = True,
    discrete_eval: bool = False,
    setup: Optional[DefaultExperimentSetup] = None,
    setup_class: Optional[type[DefaultExperimentSetup]] = None,
    disable_report: bool = False,
) -> TrainableReturnData:
    """
    Args:
        hparams: The hyperparameters selected for the trial from the search space from ray tune.
            Should include an `args` key with the parsed arguments.

    Attention:
        Best practice is to not refer to any objects from outer scope in the training_function
    """
    args, config, algo, reward_updaters = setup_trainable(hparams=hparams, setup=setup, setup_class=setup_class)

    # Prevent unbound variables
    result: StrictAlgorithmReturnData = {}  # type: ignore[assignment]
    metrics: TrainableReturnData | LogMetricsDict = {}  # type: ignore[assignment]
    # disc_eval_mean = None
    # disc_running_eval_reward = None
    pbar = episode_iterator(args, hparams, use_pbar=use_pbar)
    for _episode in pbar:
        result, metrics, rewards = training_step(
            algo,
            reward_updaters=reward_updaters,
            discrete_eval=discrete_eval,
            disable_report=disable_report,
            log_stats=args[LOG_STATS],
        )
        # Update progress bar
        if is_pbar(pbar):
            update_pbar(
                pbar,
                rewards=rewards,
                metrics=metrics,
                result=result,
                current_step=get_current_step(result),
                total_steps=get_total_steps(args, config),
            )
    final_results = cast("TrainableReturnData", metrics)
    if "trial_id" not in final_results:
        final_results["trial_id"] = result["trial_id"]
    if EVALUATION_RESULTS not in final_results:
        final_results[EVALUATION_RESULTS] = algo.evaluate()  # type: ignore[assignment]
    if "done" not in final_results:
        final_results["done"] = True
    if args.get("comment"):
        final_results["comment"] = args["comment"]

    try:
        reduced_results = filter_metrics(
            final_results,
            extra_keys_to_keep=[
                # Should log as video! not array
                # (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, "episode_videos_best"),
                # (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, "episode_videos_worst"),
                # (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, "episode_videos_best"),
                # (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, "episode_videos_worst"),
            ],
            cast_to="TrainableReturnData",
        )  # if not args["test"] else [(LEARNER_RESULTS,)])
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.exception("Failed to reduce results")
        return final_results
    else:
        return reduced_results

SetupClass = TypeVar("SetupClass", bound="ExperimentSetupBase")

_ParserType = TypeVar("_ParserType", bound="DefaultArgumentParser")
_ConfigType = TypeVar("_ConfigType", bound="AlgorithmConfig")
_AlgorithmType = TypeVar("_AlgorithmType", bound="Algorithm")

class DefaultTrainable(tune.Trainable, Generic["ParserType_co", "ConfigType_co", "AlgorithmType_co"]):
    """Default trainable for RLlib algorithms.

    This class is used to create a trainable that can be used with ray tune.
    It is a wrapper around the `default_trainable` function.
    """

    setup_class: type[ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]]
    discrete_eval: bool = False
    use_pbar: bool = True

    @classmethod
    def define(
        cls, setup_cls: type[ExperimentSetupBase[_ParserType, _ConfigType, _AlgorithmType]],
        *, discrete_eval: bool = False, use_pbar: bool = True
    ) -> type[DefaultTrainable[_ParserType, _ConfigType, _AlgorithmType]]:
        """This creates a subclass with ``setup_class`` set to the given class."""
        # Avoid undefined variable error in class body
        discrete_eval_ = discrete_eval
        use_pbar_ = use_pbar
        if TYPE_CHECKING:
            class DefinedDefaultTrainable(DefaultTrainable[Any, Any, Any]):
                setup_class = setup_cls
                discrete_eval = discrete_eval_
                use_pbar = use_pbar_

        else:

            class DefinedDefaultTrainable(DefaultTrainable[_ParserType, _ConfigType, _AlgorithmType]):
                setup_class = setup_cls
                discrete_eval = discrete_eval_
                use_pbar = use_pbar_

        return DefinedDefaultTrainable

    def setup(self, config: dict[str, Any]) -> None:   # called once
        # NOTE: Called during __init__
        # Setup algo, config, args, etc.
        self._setup = self.setup_class()  # XXX # FIXME # likely does not work
        # use args = config["cli_args"]
        import sys
        print(sys.argv)
        print(self._setup.args)
        breakpoint()
        self.args, algo_config, algo, self._reward_updaters = setup_trainable(
            hparams=config,
            setup=self._setup,
            setup_class=self.setup_class,
        )
        self.algorithm: AlgorithmType_co = algo
        self.config: ConfigType_co = algo_config
        self._pbar = episode_iterator(self.args, config, use_pbar=self.use_pbar)
        self._iteration = 0

    def step(self) -> LogMetricsDict:  # iteratively
        result, metrics, rewards = training_step(
            self.algorithm,
            reward_updaters=self._reward_updaters,
            discrete_eval=self.args["discrete_eval"],
            disable_report=True,
            log_stats=self.args[LOG_STATS],
        )
        # Update progress bar
        if is_pbar(self._pbar):
            update_pbar(
                self._pbar,
                rewards=rewards,
                metrics=metrics,
                result=result,
                current_step=get_current_step(result),
                total_steps=get_total_steps(self.args, self.config),
            )
            self._pbar.update()
        return metrics

    # region checkpointing

    # TODO

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[dict]:
        return super().save_checkpoint()

    def load_checkpoint(self, checkpoint):
        # set pbar
        # set weights
        # set iterations
        # set reward_updaters
        return super().load_checkpoint(checkpoint)

    # endregion

    def reset_config(self, new_config):
        # Return True if the config was reset, False otherwise
        # This will be called when tune.TuneConfig(reuse_actors=True) is used
        # TODO
        return super().reset_config(new_config)

    def cleanup(self):
        return super().cleanup()


if TYPE_CHECKING:  # check ABC
    DefaultTrainable()
