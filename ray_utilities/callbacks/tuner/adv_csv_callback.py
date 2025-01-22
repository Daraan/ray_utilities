"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

from ray.tune.logger import CSVLoggerCallback
from ray_utilities.constants import DEFAULT_VIDEO_KEYS

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial  # noqa: TC002


class AdvCSVLoggerCallback(CSVLoggerCallback):
    """Logs trial results in json format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.

    Prefents logging of videos (keys in `DEFAULT_VIDEO_KEYS`) even if they are present
    at the first iteration.
    """

    _video_keys = DEFAULT_VIDEO_KEYS

    def log_trial_result(self, iteration: int, trial: "Trial", result: dict):
        super().log_trial_result(iteration, trial, {k: v for k, v in result.items() if k not in self._video_keys})
