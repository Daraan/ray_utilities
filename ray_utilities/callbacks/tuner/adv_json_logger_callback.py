"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ray.tune.logger import JsonLoggerCallback

from ray_utilities.callbacks.tuner.new_style_logger_callback import NewStyleLoggerCallback
from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import AnyLogMetricsDict


class AdvJsonLoggerCallback(NewStyleLoggerCallback, JsonLoggerCallback):
    """Logs trial results in json format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.

    This updates class does not log videos stored in the DEFAULT_VIDEO_KEYS.
    """

    def log_trial_result(self, iteration: int, trial: Trial, result: dict[str, Any] | AnyLogMetricsDict):
        super().log_trial_result(
            iteration,
            trial,
            remove_videos(result),
        )
