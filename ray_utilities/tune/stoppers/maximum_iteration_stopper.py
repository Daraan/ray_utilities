from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.tune.result import TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.stopper.maximum_iteration import MaximumIterationStopper as _RayMaximumIterationStopper

from ray_utilities.constants import CURRENT_STEP

if TYPE_CHECKING:
    from ray_utilities.typing.metrics import AutoExtendedLogMetricsDict

logger = logging.getLogger(__name__)


class MaximumResultIterationStopper(_RayMaximumIterationStopper):
    """Stop trials after reaching a maximum number of iterations

    This stopper differs from ray.tune's MaximumIterationStopper
    that is uses result[TRAINING_ITERATION] to count the maximum iterations,
    instead of an internal counter that works like 'iterations_since_restore'.

    Args:
        max_iter: Number of iterations before stopping a trial.
    """

    def __call__(self, trial_id: str, result: AutoExtendedLogMetricsDict | dict[str, Any]):
        self._iter[trial_id] += 1  # basically training iterations since restore
        stop = result[TRAINING_ITERATION] >= self._max_iter
        if stop:
            logger.info(
                "Stopping trial %s at iteration %s >= max_iter %s, with %s environment steps sampled.",
                trial_id,
                result[TRAINING_ITERATION],
                self._max_iter,
                result.get(
                    CURRENT_STEP,
                    result[ENV_RUNNER_RESULTS].get(
                        NUM_ENV_STEPS_SAMPLED_LIFETIME,
                        "unknown (current_step and num_env_steps_sampled_lifetime missing)",
                    ),
                ),
            )
        return stop
