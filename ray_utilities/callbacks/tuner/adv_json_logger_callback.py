"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ray.tune.logger import JsonLoggerCallback, EXPR_RESULT_FILE

from ray_utilities.callbacks.tuner.new_style_logger_callback import NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.constants import FORK_FROM
from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import AnyLogMetricsDict


logger = logging.getLogger(__name__)


class AdvJsonLoggerCallback(NewStyleLoggerCallback, TrackForkedTrialsMixin, JsonLoggerCallback):
    """Logs trial results in JSON format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.

    This updated class does not log videos stored in the DEFAULT_VIDEO_KEYS.

    When a trial is forked (has ``FORK_FROM`` in config), creates a new JSON file
    and optionally copies data from the parent trial.
    """

    def log_trial_start(self, trial: "Trial"):
        """Start logging for a trial, handling forked trials."""
        if trial in self._trial_files:
            self._trial_files[trial].close()

        # Update config
        self.update_config(trial, trial.config)

        # Make sure logdir exists
        trial.init_local_path()
        local_file = Path(trial.local_path, EXPR_RESULT_FILE)

        # Check if this is a forked trial
        is_forked = FORK_FROM in trial.config

        if is_forked:
            # For forked trials, create a new JSON file and copy parent data if available
            fork_data = trial.config[FORK_FROM]
            parent_id = fork_data["parent_id"]

            # Try to find parent trial's JSON file
            parent_json_path = self._find_parent_json(trial, parent_id)

            if parent_json_path and parent_json_path.exists():
                logger.info(
                    "Trial %s forked from %s. Copying parent JSON data from %s",
                    trial.trial_id,
                    parent_id,
                    parent_json_path,
                )
                # Copy parent JSON to new location
                shutil.copy2(parent_json_path, local_file)
            else:
                logger.info(
                    "Trial %s forked from %s. Parent JSON not found, starting fresh.",
                    trial.trial_id,
                    parent_id,
                )
        else:
            # Resume the file from remote storage for non-forked trials
            self._restore_from_remote(EXPR_RESULT_FILE, trial)

        self._trial_files[trial] = local_file.open("at")

    def _find_parent_json(self, trial: "Trial", parent_id: str) -> Path | None:
        """Find the JSON file of the parent trial.

        Searches in the trial's experiment directory for a trial with matching ID.
        """
        try:
            experiment_path = Path(trial.local_path).parent
            # Look for parent trial directory
            for trial_dir in experiment_path.iterdir():
                if trial_dir.is_dir() and parent_id in trial_dir.name:
                    parent_json = trial_dir / EXPR_RESULT_FILE
                    if parent_json.exists():
                        return parent_json
        except Exception as e:  # noqa: BLE001
            logger.warning("Error finding parent JSON for trial %s: %s", trial.trial_id, e)
        return None

    def log_trial_result(self, iteration: int, trial: Trial, result: dict[str, Any] | AnyLogMetricsDict):
        super().log_trial_result(
            iteration,
            trial,
            remove_videos(result),
        )
