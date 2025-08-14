from __future__ import annotations

import logging
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.annotations import override
from ray.tune.result import SHOULD_CHECKPOINT
from typing_extensions import Self

from ray_utilities.training.helpers import get_current_step

if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.tune.experiment import Trial

    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData
    from ray_utilities.typing.metrics import LogMetricsDict

_logger = logging.getLogger(__name__)


class AlgoMetricCheckpointer(RLlibCallback):
    """Callbacks that adds ``SHOULD_CHECKPOINT`` to results if a metric condition is met."""

    condition: Optional[Callable[[dict[str, Any]], bool]] = None

    @classmethod
    def make_callback_class(cls, **kwargs) -> type[Self]:
        return partial(cls, **kwargs)  # pyright: ignore[reportReturnType]

    def __init__(
        self, metric_name: Optional[str] = None, condition: Optional[Callable[[dict[str, Any]], bool]] = None
    ) -> None:
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

    def _set_checkpoint(
        self, result: StrictAlgorithmReturnData | LogMetricsDict, iteration: int, trial: Trial | None = None
    ) -> None:
        current_step = get_current_step(result)
        # config available in trial.config
        if self.condition(cast("dict[str, Any]", result)):  # pyright: ignore[reportOptionalCall]
            self._last_checkpoint_iteration = iteration
            self._last_checkpoint_value = result.get(self.metric_name, None)
            self._last_checkpoint_step = current_step
            result[SHOULD_CHECKPOINT] = True
            _logger.info(
                "Checkpointing trial %s at iteration %s, step %d with: metric %s = %s",
                trial.trial_id if trial else "",
                iteration,
                self._last_checkpoint_step,
                self.metric_name,
                self._last_checkpoint_value,
            )
        # else:
        #    _logger.debug(
        #        "Not checkpointing trial %s at iteration %s, step %d with: metric %s = %s",
        #        trial.trial_id if trial else "",
        #        iteration,
        #        self._last_checkpoint_step,
        #        self.metric_name,
        #        result.get(self.metric_name, None),
        #    )

    @override(RLlibCallback)
    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict[str, Any],
        **kwargs,
    ) -> None:
        # NOTE: Algorithm iteration might be off by 1 as it is logged earlier in the training loop.
        self._set_checkpoint(result, iteration=algorithm.iteration)  # pyright: ignore[reportArgumentType]


class AlgoStepCheckpointer(AlgoMetricCheckpointer):  # type: ignore
    """Checkpoints trials based on a specific metric condition."""

    def _condition(self, result: StrictAlgorithmReturnData | LogMetricsDict | dict) -> bool:
        steps_since_last_checkpoint = get_current_step(result) - self._last_checkpoint_step  # pyright: ignore[reportArgumentType]

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
