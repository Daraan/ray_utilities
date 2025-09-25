"""Ray Tune stopper for iteration-based early stopping with result tracking.

This module provides :class:`MaximumResultIterationStopper`, an enhanced version
of Ray Tune's MaximumIterationStopper that uses result-based iteration counting
instead of internal restore-based counting.

Key Components:
    - :class:`MaximumResultIterationStopper`: Result-based iteration stopper
    - Integration with Ray RLlib metrics and logging
    - Enhanced stopping criteria with environment step tracking

The stopper provides more accurate iteration counting for distributed training
scenarios where internal iteration counters may not reflect actual training progress.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.tune.result import TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.stopper.maximum_iteration import MaximumIterationStopper as _RayMaximumIterationStopper

from ray_utilities.training.helpers import get_current_step

if TYPE_CHECKING:
    from ray_utilities.typing.metrics import AutoExtendedLogMetricsDict

logger = logging.getLogger(__name__)


class MaximumResultIterationStopper(_RayMaximumIterationStopper):
    """Stop trials after reaching a maximum number of training iterations.

    This stopper differs from Ray Tune's standard MaximumIterationStopper by using
    the training iteration count from results rather than an internal counter that
    tracks iterations since restore. This provides more accurate stopping behavior
    for distributed training scenarios.

    The stopper examines `result[TRAINING_ITERATION]` to determine when to stop trials,
    which reflects the actual training progress rather than restore-relative progress.
    It also provides enhanced logging with environment step information.

    Args:
        max_iter: Maximum number of training iterations before stopping a trial.

    Example:
        >>> stopper = MaximumResultIterationStopper(max_iter=1000)
        >>> # Use with Ray Tune
        >>> tune.run(trainable, stop=stopper)

    Note:
        When a trial is stopped, the stopper logs the trial ID, current iteration,
        maximum iteration limit, and total environment steps sampled for debugging
        and monitoring purposes.

    See Also:
        :class:`ray.tune.stopper.maximum_iteration.MaximumIterationStopper`: Base stopper class
        :data:`ray_utilities.constants.CURRENT_STEP`: Current step metric key
    """

    def __call__(self, trial_id: str, result: AutoExtendedLogMetricsDict | dict[str, Any]):
        """Evaluate whether a trial should be stopped based on iteration count.

        Args:
            trial_id: Unique identifier for the trial being evaluated.
            result: Dictionary containing trial results and metrics.

        Returns:
            True if the trial should be stopped (iteration limit reached),
            False otherwise.

        Note:
            When stopping a trial, logs detailed information including the trial ID,
            current iteration, maximum iteration limit, and environment steps sampled.
        """
        self._iter[trial_id] += 1  # basically training iterations since restore
        stop = result[TRAINING_ITERATION] >= self._max_iter
        if stop:
            logger.info(
                "Stopping trial %s at iteration %s >= max_iter %s, with %s environment steps sampled.",
                trial_id,
                result[TRAINING_ITERATION],
                self._max_iter,
                get_current_step(result),
            )
        return stop
