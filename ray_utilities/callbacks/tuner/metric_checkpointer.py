from __future__ import annotations

from functools import partial
import logging
import sys
from typing import TYPE_CHECKING, Any, List, Optional, cast

from ray.rllib.utils.annotations import override
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
from ray.tune.result import SHOULD_CHECKPOINT
from typing_extensions import deprecated, Self

from ray_utilities.training.helpers import get_current_step

if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.tune.experiment import Trial

    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData
    from ray_utilities.typing.metrics import LogMetricsDict

_logger = logging.getLogger(__name__)


TUNE_RESULT_IS_A_COPY = True  # NOTE: This is not a dynamic or checked value.
"""
Tuner does not allow to modify the result dict as it is a copy. As long as this is True
use the callback on an Algorithm. Or the HACK in the DefaultTrainable class to trigger
checkpointing.

See Also:
    - https://github.com/ray-project/ray/pull/55527
"""


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
            result[SHOULD_CHECKPOINT] = True  # XXX # NOTE: That this is a copy and does NOT WORK
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

    @override(Callback)
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


@deprecated("Do not use as long as tune passes only a copy of the result dict.", stacklevel=2)
class StepCheckpointer(MetricCheckpointer):  # type: ignore
    """Checkpoints trials based on a specific metric condition."""

    def _condition(self, result: StrictAlgorithmReturnData | LogMetricsDict | dict) -> bool:
        steps_since_last_checkpoint = get_current_step(result) - self._last_checkpoint_step  # pyright: ignore[reportArgumentType]
        # _logger.debug(
        #    "StepCheckpointer: steps since last checkpoint: %d, frequency: %d",
        #    steps_since_last_checkpoint,
        #    self._checkpoint_frequency,
        # )
        return steps_since_last_checkpoint >= self._checkpoint_frequency

    def __init__(self, checkpoint_frequency: int = 50_000) -> None:
        if checkpoint_frequency == 0:
            _logger.info("Checkpoint frequency is set to 0, disabling step checkpointing.")
            checkpoint_frequency = sys.maxsize
        self._checkpoint_frequency = checkpoint_frequency
        super().__init__("current_step", self._condition)

    @classmethod
    def make_callback_class(cls, *, checkpoint_frequency, **kwargs) -> type[Self]:
        return partial(cls, checkpoint_frequency=checkpoint_frequency, **kwargs)  # pyright: ignore[reportReturnType]
