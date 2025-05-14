# pyright: enableExperimentalFeatures=true
from __future__ import annotations

import logging
import tempfile
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, overload

from ray import tune
from ray.experimental import tqdm_ray
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from typing_extensions import TypeAliasType

from ray_utilities.callbacks.progress_bar import update_pbar
from ray_utilities.config.experiment_base import ExperimentSetupBase
from ray_utilities.config.typed_argument_parser import LOG_STATS, DefaultArgumentParser
from ray_utilities.constants import EVALUATED_THIS_STEP
from ray_utilities.environment import seed_environments_for_config
from ray_utilities.misc import is_pbar
from ray_utilities.postprocessing import (
    create_log_metrics,
    create_running_reward_updater,
    filter_metrics,
)
from ray_utilities.postprocessing import (
    verify_return as verify_return_type,
)
from ray_utilities.typing import TrainableReturnData

if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.rllib.algorithms import Algorithm, AlgorithmConfig

    from ray_utilities.typing import LogMetricsDict, StrictAlgorithmReturnData

logger = logging.getLogger(__name__)

DefaultExperimentSetup = TypeAliasType(
    "DefaultExperimentSetup", ExperimentSetupBase[DefaultArgumentParser, "AlgorithmConfig", "Algorithm"]
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


def default_trainable(
    hparams,
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
    # region setup config
    if setup:
        # TODO: Use hparams
        args = setup.args
        args = vars(args).copy()
        config = setup.config
    elif setup_class:
        args = hparams["cli_args"]
        # TODO: this should use the parameters from the search space
        # env_seed only for DDTSetup currently
        try:
            config = setup_class.config_from_args(args, env_seed=hparams.get("env_seed"))  # pyright: ignore[reportCallIssue]
        except TypeError:
            logger.debug("Setup class %s does not support 'env_seed'. Extend config_from_args", setup_class)
            config = setup_class.config_from_args(args)
    else:
        raise ValueError("Either setup or setup_class must be provided.")

    # endregion
    # region seeding
    if (run_seed := hparams.get("run_seed", None)) is not None:
        logger.debug("Using run_seed for config.seed %s", run_seed)
        config.debugging(seed=run_seed)
    # Seeded environments
    if args["env_seeding_strategy"] == "sequential":
        seed_environments_for_config(config, run_seed)
    elif args["env_seeding_strategy"] == "same":
        seed_environments_for_config(config, args["seed"])
    elif args["env_seeding_strategy"] == "constant":
        seed_environments_for_config(config, 0)
    else:
        seed_environments_for_config(config, None)
    # endregion
    try:
        # new API
        algo = config.build_algo()
    except AttributeError:
        algo = config.build()

    running_reward_updater = create_running_reward_updater()
    running_eval_reward_updater = create_running_reward_updater()
    running_disc_eval_reward_updater = create_running_reward_updater()
    # Prevent unbound variables
    result: StrictAlgorithmReturnData = {}  # type: ignore[assignment]
    metrics: TrainableReturnData | LogMetricsDict = {}  # type: ignore[assignment]
    disc_eval_mean = None
    disc_running_eval_reward = None
    pbar = episode_iterator(args, hparams, use_pbar=use_pbar)
    total_steps = args.get("total_steps", None)
    for _episode in pbar:
        # Train and get results
        result = cast("StrictAlgorithmReturnData", algo.train())

        # Reduce to key-metrics
        metrics = create_log_metrics(result, discrete_eval=discrete_eval, log_stats=args[LOG_STATS])
        # Possibly use if train.get_context().get_local/global_rank() == 0 to save videos
        # Unknown if should save video here and clean from metrics or save in a callback later is faster.

        # Training
        train_reward = metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        running_reward = running_reward_updater(train_reward)

        # Evaluation:
        eval_mean = metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        running_eval_reward = running_eval_reward_updater(eval_mean)

        # Discrete rewards:
        if "discrete" in metrics[EVALUATION_RESULTS]:
            disc_eval_mean = metrics[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]  # pyright: ignore[reportTypedDictNotRequiredAccess]
            disc_running_eval_reward = running_disc_eval_reward_updater(disc_eval_mean)

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

        # Update progress bar
        if not is_pbar(pbar):
            continue
        update_pbar(
            pbar,
            train_results={
                "mean": metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN],
                "max": result["env_runners"].get("episode_return_max", float("nan")),
                "roll": running_reward,
            },
            eval_results={
                "mean": eval_mean,
                "roll": running_eval_reward,
            },
            discrete_eval_results=(
                {
                    "mean": disc_eval_mean,
                    "roll": disc_running_eval_reward,
                }
                if disc_eval_mean is not None and disc_running_eval_reward
                else None
            ),
            total_steps=total_steps,
            current_step=result[ENV_RUNNER_RESULTS].get(NUM_ENV_STEPS_SAMPLED_LIFETIME),
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

    # Postprocess results and return
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
    except Exception:
        logger.exception("Failed to reduce results")
        return final_results
    else:
        return reduced_results
