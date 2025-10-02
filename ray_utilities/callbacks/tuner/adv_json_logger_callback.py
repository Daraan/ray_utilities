"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ray.air.constants import EXPR_RESULT_FILE
from ray.tune.experiment.trial import Trial
from ray.tune.logger import JsonLoggerCallback

from ray_utilities.callbacks.tuner.new_style_logger_callback import NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.constants import FORK_FROM
from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from ray_utilities.typing import ForkFromData
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

    def _make_forked_trial_file_name(self, trial: "Trial", fork_data: ForkFromData) -> str:
        return f"result-fork-{self.make_forked_trial_id(trial, fork_data)}.json"

    def _setup_forked_trial(self, trial: "Trial", fork_data: ForkFromData):
        """Setup trial logging, handling forked trials by creating new files."""
        # Close current file and clean tracks
        self.log_trial_end(trial)

        # Update config
        self.update_config(trial, trial.config)

        # Make sure logdir exists
        result_file = self._make_forked_trial_file_name(trial, fork_data)
        trial.init_local_path()
        local_file_path = Path(trial.local_path, result_file)  # pyright: ignore[reportArgumentType]
        assert not local_file_path.exists(), "File should not exist yet"

        # Check for parent file
        if (parent_trial := fork_data.get("parent_trial")) or isinstance(
            parent_trial := self.parent_trial_lookup.get(trial), Trial
        ):
            # is EXPR_RESULT_FILE if parent is not a fork
            # else it is result-{fork_id}.json
            parent_fork_id = self._current_fork_ids.get(parent_trial, None)
            parent_file_name = EXPR_RESULT_FILE if parent_fork_id is None else f"result-{parent_fork_id}.json"
            # Sync from parent trial to local path
            # but we also do not want to overwrite result.json
            parent_local_file_path = Path(parent_trial.local_path, parent_file_name)  # pyright: ignore[reportArgumentType]
            if trial.node_ip == parent_trial.node_ip and parent_local_file_path.exists():
                # same node use shutil
                shutil.copy2(parent_local_file_path, local_file_path)
            elif trial.storage and parent_trial.storage:
                try:
                    parent_remote_file_path = (Path(parent_trial.path) / parent_file_name).as_posix()  # pyright: ignore[reportArgumentType]
                    logger.debug("Syncing up parent JSON file to %s", parent_remote_file_path)
                    parent_trial.storage.syncer.sync_up(
                        parent_local_file_path.as_posix(),
                        parent_remote_file_path,
                        exclude=["*/checkpoint_*", "*.pkl", "events.out.tfevents.*", "*.csv"],
                    )
                    parent_trial.storage.syncer.wait()
                    logger.debug("Syncing down parent JSON file to %s", local_file_path.as_posix())
                    trial.storage.syncer.sync_down(
                        parent_remote_file_path,
                        local_file_path.as_posix(),
                        exclude=["*/checkpoint_*", "*.pkl", "events.out.tfevents.*", "*.csv"],
                    )
                    trial.storage.syncer.wait()
                except Exception:
                    logger.exception(
                        "Trial %s forked from %s but could not copy parent JSON data from remote storage.",
                        trial.trial_id,
                        parent_trial.trial_id,
                    )
            else:
                logger.warning(
                    "Trial %s forked from %s but could not copy parent JSON data, no storage backend.",
                    trial.trial_id,
                    parent_trial.trial_id,
                )
        elif (
            checkpoint_path := trial.config.get("cli_args", {}).get("from_checkpoint", None)
            or trial.config.get("from_checkpoint", None)
        ) is not None:
            # loading from checkpoint
            # Sync from checkpoint to local path
            # TODO:
            # Which result file to take when trial was forked? We know the file is in the parent folder of the
            # checkpoint. We can take it ONLY if there is one file, AND need to trim it to the step we are loading.
            parent_dir = Path(checkpoint_path).parent
            if parent_dir.exists():
                result_files = list(parent_dir.glob("result*.json"))
                if len(result_files) == 1:
                    shutil.copy2(result_files[0], local_file_path)
                    # TODO: trim to step that is loaded
                    logger.warning(
                        "(TODO) Could copy result file from checkpoint, but it is not trimmed to loaded step yet."
                    )
                else:
                    logger.info(
                        "(TODO) Trial %s forked but found multiple result files in checkpoint. "
                        "We do not know which to take for the fork, creating a new one. Files: %s",
                        trial.trial_id,
                        [f.name for f in result_files],
                    )

        if not local_file_path.exists():
            # Write fresh file
            logger.warning(
                "Trial %s forked but found no logfile for parent, starting fresh .json log file: %s",
                trial.trial_id,
                local_file_path,
            )

        self._trial_files[trial] = local_file_path.open("at")

    def on_trial_result(self, iteration: int, trials: list[Trial], trial: Trial, result, **info):
        self._trials = trials
        return super().on_trial_result(iteration, trials, trial, result, **info)

    def log_trial_start(self, trial: Trial):
        if trial in self._trial_files and FORK_FROM in trial.config:
            assert self.should_restart_logging(trial)
            self._setup_forked_trial(trial, trial.config[FORK_FROM])
        return super().log_trial_start(trial)

    def log_trial_result(self, iteration: int, trial: Trial, result: dict[str, Any] | AnyLogMetricsDict):
        super().log_trial_result(
            iteration,
            trial,
            remove_videos(result),
        )
