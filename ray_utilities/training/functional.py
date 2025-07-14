# pyright: enableExperimentalFeatures=true
from __future__ import annotations

import tempfile
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Optional, cast

from ray import tune
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EVALUATION_RESULTS

from ray_utilities.callbacks.progress_bar import update_pbar
from ray_utilities.config.typed_argument_parser import LOG_STATS
from ray_utilities.constants import EVALUATED_THIS_STEP
from ray_utilities.misc import is_pbar
from ray_utilities.postprocessing import create_log_metrics, filter_metrics
from ray_utilities.postprocessing import verify_return as verify_return_type
from ray_utilities.training.helpers import (
    DefaultExperimentSetup,
    episode_iterator,
    get_current_step,
    get_total_steps,
    logger,
    setup_trainable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.rllib.algorithms import Algorithm

    from ray_utilities.config.typed_argument_parser import LogStatsChoices
    from ray_utilities.typing import (
        LogMetricsDict,
        RewardsDict,
        RewardUpdaters,
        StrictAlgorithmReturnData,
        TrainableReturnData,
    )


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
    except Exception:  # noqa: BLE001
        logger.exception("Failed to reduce results")
        return final_results
    else:
        return reduced_results


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
        from ray_utilities.typing import TrainableReturnData

        return verify_return_type(TrainableReturnData)(trainable)
    return wraps(default_trainable)(trainable)


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
