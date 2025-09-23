from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, TypeVar

from ray.tune.logger import LoggerCallback

from ray_utilities.constants import RAY_UTILITIES_NEW_LOG_FORMAT
from ray_utilities.postprocessing import log_metrics_to_new_layout

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import AnyAutoExtendedLogMetricsDict, AnyLogMetricsDict


LogMetricsDictT = TypeVar("LogMetricsDictT", bound="dict[str, Any] | AnyAutoExtendedLogMetricsDict | AnyLogMetricsDict")


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
        super().on_trial_result(iteration, trials, trial, result, **info)
        if os.environ.get(RAY_UTILITIES_NEW_LOG_FORMAT, "1").lower() in ("0", "false", "off"):
            self.log_trial_result(
                iteration,
                trial,
                result,  # pyright: ignore[reportArgumentType]
            )
            return
        self.log_trial_result(
            iteration,
            trial,
            log_metrics_to_new_layout(result),  # pyright: ignore[reportArgumentType]
        )

    if TYPE_CHECKING:

        def log_trial_result(self, iteration: int, trial: "Trial", result: dict[str, Any] | AnyLogMetricsDict): ...
