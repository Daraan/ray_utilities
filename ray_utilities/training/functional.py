# pyright: enableExperimentalFeatures=true
"""Functional interface for Ray RLlib training with Ray Tune integration.

This module provides functional-style training utilities that integrate Ray RLlib
algorithms with Ray Tune for distributed training and hyperparameter optimization.
It offers both high-level training functions and lower-level utilities for custom
training loops.

The main function :func:`default_trainable` serves as a drop-in trainable for Ray Tune
that handles the complete training lifecycle including algorithm setup, progress
tracking, evaluation, and result reporting. Additional utilities support custom
training workflows and advanced use cases.

Key Components:
    - :func:`default_trainable`: Main trainable function for Ray Tune integration
    - :func:`training_step`: Core training step implementation with metrics processing
    - Progress bar integration and experiment tracking utilities
    - Comprehensive metrics filtering and logging capabilities

The module is designed to work seamlessly with
:class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase` subclasses
and provides a bridge between experiment configuration and actual training execution.
"""

from __future__ import annotations

import tempfile
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Optional, cast

from ray import tune
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EVALUATION_RESULTS

from ray_utilities.callbacks.progress_bar import update_pbar
from ray_utilities.config.parser.default_argument_parser import LOG_STATS
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

    from ray_utilities.config.parser.default_argument_parser import LogStatsChoices
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
    """Main trainable function for Ray Tune integration with RLlib algorithms.

    This function provides a complete training interface that can be used directly
    with Ray Tune for hyperparameter optimization or distributed training. It handles
    the full training lifecycle including algorithm setup, training loops, progress
    tracking, evaluation, and result reporting.

    The function automatically configures the RLlib algorithm based on the provided
    experiment setup and hyperparameters, runs the training loop with progress
    monitoring, and returns properly formatted results for Ray Tune consumption.

    Args:
        hparams: Hyperparameters dictionary from Ray Tune containing the search space
            selections. Must include an ``args`` key with parsed command-line arguments
            from :class:`~ray_utilities.config.DefaultArgumentParser`.
        use_pbar: Whether to display a progress bar during training. Defaults to ``True``.
        discrete_eval: Whether to enable discrete evaluation metrics processing.
            Defaults to ``False``.
        setup: Pre-configured experiment setup instance. If provided, ``setup_class``
            should not be specified.
        setup_class: Class to instantiate for experiment setup. Used when ``setup``
            is not provided.
        disable_report: Whether to disable Ray Tune result reporting. Useful for
            debugging or custom result handling. Defaults to ``False``.

    Returns:
        Training results dictionary compatible with Ray Tune, containing metrics,
        episode information, and evaluation results.

    Example:
        >>> # Direct usage
        >>> from ray_utilities.setup import PPOSetup
        >>> hparams = {"args": parser.parse_args()}
        >>> results = default_trainable(hparams, setup_class=PPOSetup)

        >>> # With Ray Tune
        >>> tuner = tune.Tuner(
        ...     tune.with_parameters(default_trainable, setup_class=PPOSetup),
        ...     param_space={"args": tune.grid_search([args1, args2])},
        ... )

    Note:
        Best practice is to avoid referring to objects from outer scope within
        this function to ensure proper isolation in distributed environments.

    See Also:
        :func:`training_step`: Core training step implementation
        :func:`~ray_utilities.training.helpers.setup_trainable`: Trainable setup helper
        :class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase`: Base setup class
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
        setup: A setup instance for the experiment.
        setup_class: Alternatively to an instance pass a setup class for the experiment.
        disable_report: Do not log results.
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
        from ray_utilities.typing import TrainableReturnData  # noqa: PLC0415

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
    """Execute a single training step with comprehensive metrics processing.

    This function performs one training iteration of the RLlib algorithm, processes
    the results to extract key metrics, updates running reward trackers, and reports
    results to Ray Tune. It handles both continuous and discrete evaluation metrics.

    The function abstracts the core training loop logic used by :func:`default_trainable`
    and can be used independently for custom training workflows that need fine-grained
    control over the training process.

    Args:
        algo: The RLlib algorithm instance to train. Should be properly configured
            and ready for training.
        reward_updaters: Dictionary of reward tracking functions that maintain running
            averages for training and evaluation rewards. Must include ``"running_reward"``,
            ``"eval_reward"``, and optionally ``"disc_eval_reward"`` for discrete evaluation.
        discrete_eval: Whether to process discrete evaluation metrics. When ``True``,
            expects discrete evaluation results in the algorithm output. Defaults to ``False``.
        disable_report: Whether to skip reporting results to Ray Tune. Useful for
            debugging or when using custom result handling. Defaults to ``False``.
        log_stats: Level of detail for metrics logging. Controls which metrics are
            included in the processed output. Defaults to ``"minimal"``.

    Returns:
        A tuple containing:
            - **result**: Raw algorithm training results with all metrics and metadata
            - **metrics**: Processed and filtered metrics ready for logging/reporting
            - **rewards**: Dictionary with running reward values and current episode means

    Example:
        >>> from ray_utilities.training.helpers import create_running_reward_updater
        >>> reward_updaters = {
        ...     "running_reward": create_running_reward_updater(),
        ...     "eval_reward": create_running_reward_updater(),
        ... }
        >>> result, metrics, rewards = training_step(algo, reward_updaters, log_stats="full")

    Note:
        This function automatically handles Ray Tune result reporting unless disabled.
        Checkpointing is performed selectively based on evaluation timing to optimize
        performance.

    See Also:
        :func:`default_trainable`: High-level trainable that uses this function
        :func:`~ray_utilities.postprocessing.create_log_metrics`: Metrics processing
        :func:`~ray_utilities.training.helpers.create_running_reward_updater`: Reward tracking
    """
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
            import ray.train  # noqa: PLC0415  # Old API

            ray.train.report(report_metrics, checkpoint=None)
    rewards: RewardsDict = {
        "running_reward": running_reward,
        "running_eval_reward": running_eval_reward,
        "eval_mean": eval_mean,
        "disc_eval_mean": disc_eval_mean or 0,
        "disc_eval_reward": disc_running_eval_reward or 0,
    }
    return result, metrics, rewards
