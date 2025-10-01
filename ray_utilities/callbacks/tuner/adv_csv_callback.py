"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from ray.tune.logger import EXPR_PROGRESS_FILE, CSVLoggerCallback

from ray_utilities.callbacks.tuner.new_style_logger_callback import NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.constants import FORK_FROM
from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import AnyLogMetricsDict


logger = logging.getLogger(__name__)


class AdvCSVLoggerCallback(NewStyleLoggerCallback, TrackForkedTrialsMixin, CSVLoggerCallback):
    """Logs trial results in CSV format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.

    Prevents logging of videos (keys in `DEFAULT_VIDEO_KEYS`) even if they are present
    at the first iteration.

    When a trial is forked (has ``FORK_FROM`` in config), creates a new CSV file
    and optionally copies data from the parent trial.
    """

    def _setup_trial(self, trial: "Trial"):
        """Setup trial logging, handling forked trials by creating new files."""
        if trial in self._trial_files:
            self._trial_files[trial].close()

        # Make sure logdir exists
        trial.init_local_path()
        local_file_path = Path(trial.local_path, EXPR_PROGRESS_FILE)

        # Check if this is a forked trial
        is_forked = FORK_FROM in trial.config

        if is_forked:
            # For forked trials, create a new CSV file and copy parent data if available
            fork_data = trial.config[FORK_FROM]
            parent_id = fork_data["parent_id"]

            # Try to find parent trial's CSV file
            parent_csv_path = self._find_parent_csv(trial, parent_id)

            if parent_csv_path and parent_csv_path.exists():
                logger.info(
                    "Trial %s forked from %s. Copying parent CSV data from %s",
                    trial.trial_id,
                    parent_id,
                    parent_csv_path,
                )
                # Copy parent CSV to new location
                shutil.copy2(parent_csv_path, local_file_path)
                self._trial_continue[trial] = True
            else:
                logger.info(
                    "Trial %s forked from %s. Parent CSV not found, starting fresh.",
                    trial.trial_id,
                    parent_id,
                )
                self._trial_continue[trial] = False
        else:
            # Resume the file from remote storage for non-forked trials
            self._restore_from_remote(EXPR_PROGRESS_FILE, trial)
            self._trial_continue[trial] = local_file_path.exists() and local_file_path.stat().st_size > 0

        self._trial_files[trial] = local_file_path.open("at")
        self._trial_csv[trial] = None

    def _find_parent_csv(self, trial: "Trial", parent_id: str) -> Path | None:
        """Find the CSV file of the parent trial.

        Searches in the trial's experiment directory for a trial with matching ID.
        """
        try:
            experiment_path = Path(trial.local_path).parent
            # Look for parent trial directory
            for trial_dir in experiment_path.iterdir():
                if trial_dir.is_dir() and parent_id in trial_dir.name:
                    parent_csv = trial_dir / EXPR_PROGRESS_FILE
                    if parent_csv.exists():
                        return parent_csv
        except Exception as e:  # noqa: BLE001
            logger.warning("Error finding parent CSV for trial %s: %s", trial.trial_id, e)
        return None

    def log_trial_result(self, iteration: int, trial: "Trial", result: AnyLogMetricsDict):  # pyright: ignore[reportIncompatibleMethodOverride]
        if trial not in self._trial_csv:
            # Keys are permanently set; remove videos from the first iteration
            result = remove_videos(result)
        super().log_trial_result(
            iteration,
            trial,
            result,
        )
