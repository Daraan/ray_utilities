"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ray.air.constants import EXPR_RESULT_FILE
from ray.tune.experiment.trial import Trial
from ray.tune.logger import JsonLoggerCallback

from ray_utilities.callbacks.tuner._file_logger_fork_mixin import FileLoggerForkMixin
from ray_utilities.callbacks.tuner.new_style_logger_callback import NewStyleLoggerCallback
from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from ray_utilities.typing.metrics import AnyLogMetricsDict


logger = logging.getLogger(__name__)


class AdvJsonLoggerCallback(NewStyleLoggerCallback, FileLoggerForkMixin, JsonLoggerCallback):
    """Logs trial results in JSON format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.

    This updated class does not log videos stored in the DEFAULT_VIDEO_KEYS.

    When a trial is forked (has ``FORK_FROM`` in config), creates a new JSON file
    and optionally copies data from the parent trial.
    """

    def _get_file_extension(self) -> str:
        return "json"

    def _get_file_base_name(self) -> str:
        return "result"

    def _get_default_file_name(self) -> str:
        return EXPR_RESULT_FILE

    def _setup_file_handle(self, trial: Trial, local_file_path: Path) -> None:
        """Open the JSON file handle and update config."""
        self.update_config(trial, trial.config)
        self._trial_files[trial] = local_file_path.open("at")

    def _handle_missing_parent_file(self, trial: Trial, local_file_path: Path) -> None:
        """Handle missing parent file for JSON logger."""
        logger.warning(
            "Trial %s forked but found no logfile for parent, starting fresh .json log file: %s",
            trial.trial_id,
            local_file_path,
        )

    def log_trial_result(self, iteration: int, trial: Trial, result: dict[str, Any] | AnyLogMetricsDict):
        super().log_trial_result(
            iteration,
            trial,
            remove_videos(result),
        )
