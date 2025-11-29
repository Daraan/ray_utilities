from __future__ import annotations

import os
from math import floor, log10, isnan
from typing import TYPE_CHECKING, Any, Mapping, TypeVar

import tree
from ray.tune.logger import LoggerCallback

from ray_utilities.constants import RAY_UTILITIES_NEW_LOG_FORMAT
from ray_utilities.postprocessing import log_metrics_to_new_layout

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import AnyAutoExtendedLogMetricsDict, AnyLogMetricsDict


LogMetricsDictT = TypeVar("LogMetricsDictT", bound="dict[str, Any] | AnyAutoExtendedLogMetricsDict | AnyLogMetricsDict")
_M = TypeVar("_M", bound=Mapping[str, Any])


def _round_floats(path, value: float | str | Any):
    if not isinstance(value, float) or isnan(value) or value == 0.0 or "lr" == path[-1]:
        return value
    if abs(value) > 100:
        return round(value, 2)
    if abs(value) > 10:
        return round(value, 3)
    if abs(value) > 1:
        return round(value, 4)
    if abs(value) >= 0.0001:
        return round(value, 6)
    # round to at least two significant digits

    digits = -floor(log10(abs(value))) + 1
    return round(value, digits + 1)


def round_floats(results: _M) -> _M:
    """
    Round all float values to 6 decimal places for better readability or at last two significant digits if lower.
    Ignores learning_rate (lr)
    """
    return tree.map_structure_with_path(
        _round_floats,
        results,
    )  # pyright: ignore[reportReturnType]


class NewStyleLoggerCallback(LoggerCallback):
    """
    If enabled transforms the logged results to the new style layout.

    Replaces:
    - env_runners -> training
    - evaluation/env_runners -> evaluation

    - And merges learner results if there is only one module.

    Subclasses need to be able to handle both LogMetricsDict | NewLogMetricsDict
    for their results dict.
    """

    def on_trial_result(
        self,
        iteration: int,
        trials: list["Trial"],
        trial: "Trial",
        result: dict[str, Any],
        **info,
    ):
        if os.environ.get(RAY_UTILITIES_NEW_LOG_FORMAT, "1").lower() in ("0", "false", "off"):
            super().on_trial_result(iteration, trials, trial, result, **info)
            return
        super().on_trial_result(
            iteration,
            trials,
            trial,
            log_metrics_to_new_layout(result),  # pyright: ignore[reportArgumentType]
            **info,
        )

    if TYPE_CHECKING:

        def log_trial_result(self, iteration: int, trial: "Trial", result: dict[str, Any] | AnyLogMetricsDict): ...
