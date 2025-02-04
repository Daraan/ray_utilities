"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ray.tune.logger import CSVLoggerCallback

from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import LogMetricsDict


class AdvCSVLoggerCallback(CSVLoggerCallback):
    """Logs trial results in json format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.

    Prefents logging of videos (keys in `DEFAULT_VIDEO_KEYS`) even if they are present
    at the first iteration.
    """

    def log_trial_result(self, iteration: int, trial: "Trial", result: dict | LogMetricsDict):
        if trial not in self._trial_csv:
            # Keys are permanently set; remove videos from the first iteration
            result = remove_videos(result)
        super().log_trial_result(
            iteration,
            trial,
            result,  # type: ignore[arg-type]
        )
