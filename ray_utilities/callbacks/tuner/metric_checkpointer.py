from __future__ import annotations

import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, List, Optional, cast

from ray.air.constants import TRAINING_ITERATION
from ray.rllib.utils.annotations import override
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
from ray.tune.result import SHOULD_CHECKPOINT
from typing_extensions import Self, deprecated

from ray_utilities.constants import CURRENT_STEP, TUNE_RESULT_IS_A_COPY
from ray_utilities.misc import get_current_step, warn_if_slow

if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.tune.experiment import Trial

    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData
    from ray_utilities.typing.metrics import LogMetricsDict

_logger = logging.getLogger(__name__)


# todo: do not derive from RLlibCallback when tuner checkpoint is actually working.
# Need workaround to set `SHOULD_CHECKPOINT` in the actual result dict and not on a copy of
# on_trial_result
@deprecated("Do not use as long as tune passes only a copy of the result dict.", stacklevel=2)
class MetricCheckpointer(Callback):
    """Callbacks that adds ``SHOULD_CHECKPOINT`` to results if a metric condition is met."""

    condition: Optional[Callable[[dict], bool]] = None
    _last_checkpoint_step = -1

    def __init__(self, metric_name: Optional[str] = None, condition: Optional[Callable[[dict], bool]] = None) -> None:
        if self.condition is None and condition is None:
            raise ValueError(
                "Condition must be provided for MetricCheckpointer. Either as class variable or in constructor."
            )
        if self.condition is not None and condition is not None:
            _logger.warning("Both class variable and constructor condition provided. Using constructor condition.")
        super().__init__()
        self.metric_name = metric_name or "Unknown"
        self.condition = self.condition or condition
        assert self.condition
        self._last_checkpoint_iteration = -1
        self._last_checkpoint_value = None

    def _set_checkpoint(self, result: StrictAlgorithmReturnData | LogMetricsDict, trial: Trial | None = None) -> None:
        iteration = result["training_iteration"]
        current_step = get_current_step(result)
        # config available in trial.config
        if self.condition(cast("dict[str, Any]", result)):  # pyright: ignore[reportOptionalCall]
            self._last_checkpoint_iteration = iteration
            self._last_checkpoint_value = result.get(self.metric_name, None)
            self._last_checkpoint_step = current_step
            result[SHOULD_CHECKPOINT] = True  # Needs ray 2.50.0+ to work, else result is a copy.
            _logger.info(
                "Checkpointing trial %s at iteration %s, step %d with: metric '%s' = %s%s",
                trial.trial_id if trial else "",
                iteration,
                current_step,
                self.metric_name,
                self._last_checkpoint_value,
                (
                    ". NOTE: This is only a logging message and does not confirm the checkpoint creation"
                    if TUNE_RESULT_IS_A_COPY
                    else ""
                ),
            )

    def get_state(self) -> dict:
        """Get the state of the callback for checkpointing.

        Returns:
            Dictionary containing checkpoint tracking data.

        Note:
            The condition callable is not saved as it may not be picklable.
            It must be provided again when restoring from checkpoint.
        """
        return {
            "metric_name": self.metric_name,
            "last_checkpoint_iteration": self._last_checkpoint_iteration,
            "last_checkpoint_value": self._last_checkpoint_value,
            "last_checkpoint_step": self._last_checkpoint_step,
        }

    def set_state(self, state: dict) -> None:
        """Set the state of the callback from checkpoint data.

        Args:
            state: State dictionary containing checkpoint tracking data.

        Note:
            The condition callable must be provided during __init__ as it
            cannot be restored from the checkpoint.
        """
        self.metric_name = state.get("metric_name", "Unknown")
        self._last_checkpoint_iteration = state.get("last_checkpoint_iteration", -1)
        self._last_checkpoint_value = state.get("last_checkpoint_value", None)
        self._last_checkpoint_step = state.get("last_checkpoint_step", -1)

        _logger.info(
            "Restored MetricCheckpointer state: last checkpoint at iteration %d, step %d, value %s",
            self._last_checkpoint_iteration,
            self._last_checkpoint_step,
            self._last_checkpoint_value,
        )

    @override(Callback)
    @warn_if_slow
    def on_trial_result(
        self,
        iteration: int,
        trials: List["Trial"],
        trial: "Trial",
        result: dict,
        **info,
    ):
        """Called after receiving a result from a trial.

        The search algorithm and scheduler are notified before this
        hook is called.

        Arguments:
            iteration: Number of iterations of the tuning loop.
            trials: List of trials.
            trial: Trial that just sent a result.
            result: Result that the trial sent.
            **info: Kwargs dict for forward compatibility.
        """
        self._set_checkpoint(
            result,  # pyright: ignore[reportArgumentType]
            trial,
        )


# This can be used from ray 2.50.0 onwards when tune passes the actual result dict
@deprecated("Do not use as long as tune passes only a copy of the result dict.", stacklevel=2)
class StepCheckpointer(MetricCheckpointer):  # type: ignore
    """Checkpoints trials based on a specific metric condition."""

    def _condition(self, result: StrictAlgorithmReturnData | LogMetricsDict | dict) -> bool:
        steps_since_last_checkpoint = get_current_step(result) - self._last_checkpoint_step  # pyright: ignore[reportArgumentType]
        return steps_since_last_checkpoint >= self._checkpoint_frequency and (
            not self._min_iterations
            or result[TRAINING_ITERATION] - self._last_checkpoint_iteration >= self._min_iterations
        )

    def __init__(self, checkpoint_frequency: int = 65_536, min_iterations: Optional[int] = 24) -> None:
        if checkpoint_frequency == 0:
            _logger.info("Checkpoint frequency is set to 0, disabling step checkpointing.")
            checkpoint_frequency = sys.maxsize
        self._checkpoint_frequency = checkpoint_frequency
        self._min_iterations = min_iterations
        super().__init__(CURRENT_STEP, self._condition)

    @classmethod
    def make_callback_class(cls, *, checkpoint_frequency, min_iterations: Optional[int] = 24, **kwargs) -> type[Self]:
        return partial(cls, checkpoint_frequency=checkpoint_frequency, min_iterations=min_iterations, **kwargs)  # pyright: ignore[reportReturnType]
