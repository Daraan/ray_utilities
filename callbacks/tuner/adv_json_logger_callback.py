"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

from ray.tune.logger import JsonLoggerCallback
from interpretable_ddts.runfiles.constants import DEFAULT_VIDEO_KEYS

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial  # noqa: TC002


class AdvJsonLoggerCallback(JsonLoggerCallback):
    """Logs trial results in json format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.

    This updates class does not log videos stored in the DEFAULT_VIDEO_KEYS.
    """

    _video_keys = DEFAULT_VIDEO_KEYS

    def log_trial_result(self, iteration: int, trial: "Trial", result: dict):
        super().log_trial_result(iteration, trial, {k: v for k, v in result.items() if k not in self._video_keys})
