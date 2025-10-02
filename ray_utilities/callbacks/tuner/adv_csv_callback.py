"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ray.air.constants import EXPR_PROGRESS_FILE
from ray.tune.logger import CSVLoggerCallback

from ray_utilities.callbacks.tuner._file_logger_fork_mixin import FileLoggerForkMixin
from ray_utilities.callbacks.tuner.new_style_logger_callback import NewStyleLoggerCallback
from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from pathlib import Path

    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import AnyLogMetricsDict


logger = logging.getLogger(__name__)


class AdvCSVLoggerCallback(NewStyleLoggerCallback, FileLoggerForkMixin, CSVLoggerCallback):
    """Logs trial results in CSV format.

    Prevents logging of videos (keys in :const:`DEFAULT_VIDEO_DICT_KEYS`) even if they are present
    at the first iteration.

    When a trial is forked (has ``FORK_FROM`` in config), creates a new CSV file
    and tries to copy the data from the parent trial.
    """

    def _get_file_extension(self) -> str:
        return "csv"

    def _get_file_base_name(self) -> str:
        return "progress"

    def _get_default_file_name(self) -> str:
        return EXPR_PROGRESS_FILE

    def _setup_file_handle(self, trial: Trial, local_file_path: Path) -> None:
        """Open the CSV file handle and set CSV-specific state."""
        self._trial_csv[trial] = None  # need to set key # pyright: ignore[reportArgumentType]
        if local_file_path.exists():
            # Check if header needs to be written
            write_header_again = local_file_path.stat().st_size == 0
            # Note: metrics might have changed when loading from checkpoint
            if trial.config.get("cli_args", {}).get("from_checkpoint") or trial.config.get("from_checkpoint"):
                write_header_again = True
            self._trial_continue[trial] = not write_header_again
        else:  # For now this is not called as we enter after a sync from parent
            self._restore_from_remote(local_file_path.name, trial)
            self._trial_continue[trial] = False
        self._trial_files[trial] = local_file_path.open("at")

    def _handle_missing_parent_file(self, trial: Trial, local_file_path: Path) -> None:
        """Handle missing parent file for CSV logger."""
        self._trial_continue[trial] = False
        logger.warning(
            "Trial %s forked but found no logfile for parent, starting fresh .csv log file: %s",
            trial.trial_id,
            local_file_path,
        )

    def log_trial_result(self, iteration: int, trial: "Trial", result: AnyLogMetricsDict):  # pyright: ignore[reportIncompatibleMethodOverride]
        if trial not in self._trial_csv:
            # Keys are permanently set; remove videos from the first iteration.
            # Therefore also need eval metric in first iteration
            result = remove_videos(result)

        super().log_trial_result(
            iteration,
            trial,
            result,
        )


if TYPE_CHECKING:  # Check ABC
    AdvCSVLoggerCallback()
