from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EVALUATION_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from ray_utilities.constants import (
    EPISODE_RETURN_MEAN_EMA,
    RAY_METRICS_V2,
)

if TYPE_CHECKING:
    from functools import partial

    from ray.rllib.algorithms import Algorithm
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.metrics.stats import Stats


def _compute_adjusted_ema_coeff(
    base_coeff: float,
    current_interval: int,
    base_interval: int = 1,
) -> float:
    """
    Compute time-adjusted EMA coefficient for varying evaluation intervals.

    When evaluation happens less frequently (higher interval), we need a lower
    coefficient to give more weight to new observations. This ensures that the
    effective smoothing window remains consistent across different intervals.

    Args:
        base_coeff: Base EMA coefficient (e.g., 0.8 for base_interval evaluations)
        current_interval: Current evaluation interval (iterations between evals)
        base_interval: Reference interval for which base_coeff was designed

    Returns:
        Adjusted EMA coefficient that accounts for the current evaluation frequency

    Example:
        >>> _compute_adjusted_ema_coeff(0.8, current_interval=4, base_interval=1)
        0.4096  # 0.8^4 - evaluates 4x less often, so heavier smoothing per update
    """
    if current_interval <= 0 or base_interval <= 0:
        return base_coeff
    return base_coeff ** (current_interval / base_interval)


LOG_KEY = (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN_EMA)

if not RAY_METRICS_V2:

    def _log_function(metrics_logger: MetricsLogger, key: str | tuple[str, ...], value: float, ema_coeff: float):
        metrics_logger.log_value(  # pyright: ignore[reportOptionalMemberAccess]
            key, value, ema_coeff=ema_coeff
        )
else:

    def _log_function(metrics_logger: MetricsLogger, key: str | tuple[str, ...], value: float, ema_coeff: float):
        metrics_logger.log_value(  # pyright: ignore[reportOptionalMemberAccess]
            key, value, reduce="ema", ema_coeff=ema_coeff
        )


class _PartialEvalEMAMetaCallback(type):
    """Helper class to create partial EvalEMAMetricCallback with fixed parameters."""

    # We need values here for hash, this is only relevant for cloudpickle
    ema_coeff: float = None
    base_eval_interval: int = None

    def __call__(cls):
        return EvalEMAMetricCallback(ema_coeff=cls.ema_coeff, base_eval_interval=cls.base_eval_interval)

    def __eq__(cls, value: object) -> bool:
        if not isinstance(value, _PartialEvalEMAMetaCallback):
            return False
        return (cls.ema_coeff == value.ema_coeff) and (cls.base_eval_interval == value.base_eval_interval)

    def __hash__(cls) -> int:
        if not hasattr(cls, "ema_coeff") or not hasattr(cls, "base_eval_interval"):
            # For cloudpickle of this class
            return hash(type(cls))
        return hash((type(cls), cls.ema_coeff, cls.base_eval_interval))


class EvalEMAMetricCallback(RLlibCallback):
    base_eval_interval: int = -1

    def __init__(self, ema_coeff: float, base_eval_interval: int = 1):
        super().__init__()
        self.ema_coeff = ema_coeff
        self.base_eval_interval = base_eval_interval
        self._current_interval: int

    def on_algorithm_init(self, *, algorithm: Algorithm, metrics_logger: MetricsLogger | None = None, **kwargs) -> None:
        super().on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
        # NOTE if no partial is used this has to be before DynamicEvalInterval callback in the callback lists
        if self.base_eval_interval == -1:  # restored on set_state
            self.base_eval_interval = algorithm.config.evaluation_interval or 0  # pyright: ignore[reportOptionalMemberAccess]
        self._current_interval = algorithm.config.evaluation_interval or 0  # pyright: ignore[reportOptionalMemberAccess]

    def on_evaluate_end(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        evaluation_metrics: dict,
        **kwargs,
    ):
        """
        Beside the normal evaluation metrics, this also tracks the EMA of the mean return.

        The EMA coefficient is automatically adjusted based on the current evaluation interval
        to ensure consistent smoothing behavior across different evaluation frequencies.

        Args:
            algorithm: The algorithm instance
            metrics_logger: Logger for recording metrics
            evaluation_metrics: Dictionary containing evaluation results
            ema_coeff: Base EMA coefficient for base_eval_interval frequency
            base_eval_interval: Reference evaluation interval for ema_coeff
            **kwargs: Additional arguments

        See Also:
            :attr:`AlgorithmConfig.metrics_num_episodes_for_smoothing`
            :func:`_compute_adjusted_ema_coeff`
        """
        # Get non-ema metric save in new + _ema key
        eval_metric = evaluation_metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        current_interval = algorithm.config.evaluation_interval  # pyright: ignore[reportOptionalMemberAccess]
        if not current_interval:
            return  # we do no evaluation
        if current_interval == self._current_interval:
            _log_function(
                metrics_logger,  # pyright: ignore[reportArgumentType]
                LOG_KEY,
                eval_metric,
                ema_coeff=self.ema_coeff,
            )
            return
        adjusted_coeff = _compute_adjusted_ema_coeff(self.ema_coeff, current_interval, self.base_eval_interval)
        # As the ema_coeff is "frozen" at init we need to check the stat object and modify it as needed
        try:
            eval_ema_stat = metrics_logger._get_key((EVALUATION_RESULTS, ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN_EMA))  # pyright: ignore[reportOptionalMemberAccess]
        except KeyError:
            # not yet logged
            _log_function(
                metrics_logger,  # pyright: ignore[reportArgumentType]
                LOG_KEY,
                eval_metric,
                ema_coeff=adjusted_coeff,
            )
        else:
            if not RAY_METRICS_V2:
                eval_ema_stat = cast("Stats", eval_ema_stat)
            else:
                raise NotImplementedError("RAY_METRICS_V2 path not implemented yet.")
            # Have to modify the stats object directly as we cannot overwrite the value or have public setters
            if eval_ema_stat._ema_coeff != adjusted_coeff:
                eval_ema_stat._ema_coeff = adjusted_coeff
            _log_function(
                metrics_logger,  # pyright: ignore[reportArgumentType]
                LOG_KEY,
                eval_metric,
                ema_coeff=adjusted_coeff,
            )

    def get_state(self) -> dict[str, Any]:
        # Should be no reason to save _current_interval as it is set on algo init
        return {
            "base_eval_interval": self.base_eval_interval,
            "ema_coeff": self.ema_coeff,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.base_eval_interval = state["base_eval_interval"]
        self.ema_coeff = state["ema_coeff"]


def make_eval_ema_metrics_callback(ema_coeff: float, base_eval_interval: int = 1) -> partial[RLlibCallback]:
    """
    Returns a factory for :class:`EvalEMAMetricCallback` with the given parameters.

    Args:
        ema_coeff: Base EMA coefficient
        base_eval_interval: Reference evaluation interval for the base coefficient

    Returns:
        A partial :class:`EvalEMAMetricCallback` with fixed parameters.
    """
    # rename for closure
    ema_coeff_ = ema_coeff
    base_eval_interval_ = base_eval_interval

    class _PartialEvalEMACallback(metaclass=_PartialEvalEMAMetaCallback):
        """Helper class to create partial EvalEMAMetricCallback with fixed parameters."""

        ema_coeff = ema_coeff_
        base_eval_interval = base_eval_interval_

    return _PartialEvalEMACallback
