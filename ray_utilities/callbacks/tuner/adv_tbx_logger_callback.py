from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from ray.tune.experiment.trial import Trial
from ray.tune.logger import TBXLoggerCallback

from ray_utilities.callbacks.tuner.new_style_logger_callback import LogMetricsDictT, NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS, FORK_FROM

if TYPE_CHECKING:
    from ray_utilities.typing import ForkFromData
    from ray_utilities.typing.common import VideoTypes
    from ray_utilities.typing.metrics import VideoMetricsDict, _LogMetricsEvalEnvRunnersResultsDict


logger = logging.getLogger(__name__)


class AdvTBXLoggerCallback(NewStyleLoggerCallback, TrackForkedTrialsMixin, TBXLoggerCallback):
    """TensorBoardX Logger.

    Note that hparams will be written only after a trial has terminated.
    This logger automatically flattens nested dicts to show on TensorBoard:

        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}

    Attention:
        To log videos these conditions must hold for the video value

            `isinstance(video, np.ndarray) and video.ndim == 5`
            and have the format "NTCHW"

        Videos will be logged as gif

    When a trial is forked (has ``FORK_FROM`` in config), creates a new TensorBoard
    trial and optionally copies event files from the parent trial.
    """

    _video_keys = DEFAULT_VIDEO_DICT_KEYS

    def _make_forked_trial_subdir_name(self, trial: "Trial", fork_data: ForkFromData) -> str:
        return f"tb-fork-{self.make_forked_trial_id(trial, fork_data)}"

    def _setup_forked_trial(self, trial: "Trial", fork_data: ForkFromData):
        """Setup trial logging, handling forked trials by creating new subdirectory."""
        # Close current writer and clean tracks
        self.log_trial_end(trial)

        # Make sure logdir exists
        trial.init_local_path()
        tb_subdir = self._make_forked_trial_subdir_name(trial, fork_data)
        local_tb_path = Path(trial.local_path, tb_subdir)  # pyright: ignore[reportArgumentType]
        local_tb_path.mkdir(parents=True, exist_ok=True)

        # Check for parent TensorBoard data
        if (parent_trial := fork_data.get("parent_trial")) or isinstance(
            parent_trial := self.parent_trial_lookup.get(trial), Trial
        ):
            # Get parent TB directory: either root or tb-{fork_id}
            parent_fork_id = self._current_fork_ids.get(parent_trial, None)
            if parent_fork_id is None:
                parent_tb_path = Path(parent_trial.local_path)  # pyright: ignore[reportArgumentType]
            else:
                parent_tb_path = Path(parent_trial.local_path, f"tb-{parent_fork_id}")  # pyright: ignore[reportArgumentType]

            # Sync from parent trial to local path
            if trial.node_ip == parent_trial.node_ip and parent_tb_path.exists():
                # same node use shutil - copy all event files
                try:
                    for event_file in parent_tb_path.glob("events.out.tfevents.*"):
                        shutil.copy2(event_file, local_tb_path)
                    logger.info(
                        "Trial %s forked from %s. Copied parent TensorBoard data from %s",
                        trial.trial_id,
                        parent_trial.trial_id,
                        parent_tb_path,
                    )
                except Exception:
                    logger.exception(
                        "Failed to copy parent TensorBoard data for trial %s from %s",
                        trial.trial_id,
                        parent_tb_path,
                    )
            elif trial.storage and parent_trial.storage:
                try:
                    # Sync parent's event files via remote storage
                    parent_remote_tb_path = (
                        (Path(parent_trial.path) / parent_tb_path.name).as_posix()
                        if parent_fork_id
                        else parent_trial.path
                    )  # pyright: ignore[reportArgumentType]
                    logger.debug("Syncing up parent TB files to %s", parent_remote_tb_path)
                    parent_trial.storage.syncer.sync_up(
                        parent_tb_path.as_posix(),
                        parent_remote_tb_path,
                        exclude=["*/checkpoint_*", "*.pkl", "*.csv", "*.json"],
                    )
                    parent_trial.storage.syncer.wait()
                    logger.debug("Syncing down parent TB files to %s", local_tb_path.as_posix())
                    trial.storage.syncer.sync_down(
                        parent_remote_tb_path,
                        local_tb_path.as_posix(),
                        exclude=["*/checkpoint_*", "*.pkl", "*.csv", "*.json"],
                    )
                    trial.storage.syncer.wait()
                except Exception:
                    logger.exception(
                        "Trial %s forked from %s but could not copy parent TensorBoard data from remote storage.",
                        trial.trial_id,
                        parent_trial.trial_id,
                    )
            else:
                logger.warning(
                    "Trial %s forked from %s but could not copy parent TensorBoard data, no storage backend.",
                    trial.trial_id,
                    parent_trial.trial_id,
                )
        elif (
            checkpoint_path := trial.config.get("cli_args", {}).get("from_checkpoint", None)
            or trial.config.get("from_checkpoint", None)
        ) is not None:
            # loading from checkpoint
            # Sync from checkpoint to local path
            # TODO: Handle checkpoint case for TensorBoard
            parent_dir = Path(checkpoint_path).parent
            if parent_dir.exists():
                try:
                    for event_file in parent_dir.glob("**/events.out.tfevents.*"):
                        shutil.copy2(event_file, local_tb_path)
                    logger.info("(TODO) Copied TensorBoard files from checkpoint, but may not be complete.")
                except Exception:
                    logger.exception(
                        "Trial %s forked but could not copy TensorBoard data from checkpoint.",
                        trial.trial_id,
                    )
            else:
                logger.info(
                    "Trial %s forked from checkpoint but parent directory not found, starting fresh.",
                    trial.trial_id,
                )

        if not any(local_tb_path.glob("events.out.tfevents.*")):
            logger.warning(
                "Trial %s forked but found no TensorBoard event files for parent, starting fresh in: %s",
                trial.trial_id,
                local_tb_path,
            )

        # Create new writer for this trial in the forked subdirectory
        self._trial_writer[trial] = self._summary_writer_cls(local_tb_path.as_posix(), flush_secs=30)
        self._trial_result[trial] = {}

    def on_trial_result(self, iteration: int, trials: list[Trial], trial: Trial, result, **info):
        self._trials = trials
        return super().on_trial_result(iteration, trials, trial, result, **info)

    def log_trial_start(self, trial: Trial):
        if trial in self._trial_writer and FORK_FROM in trial.config:
            assert self.should_restart_logging(trial)
            self._setup_forked_trial(trial, trial.config[FORK_FROM])
        return super().log_trial_start(trial)

    @staticmethod
    def preprocess_videos(result: LogMetricsDictT) -> LogMetricsDictT:
        """
        For tensorboard it must hold that:

        `isinstance(video, np.ndarray) and video.ndim == 5`
        """
        did_copy = False
        for keys in DEFAULT_VIDEO_DICT_KEYS:
            subdir = result
            # See if leaf is present
            for key in keys[:-1]:
                if key not in subdir:
                    break
                # key is present we can access it
                subdir = subdir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
            else:
                # Perform a selective deep copy on the modified items
                subdir = cast("dict[str, VideoMetricsDict]", subdir)
                # keys[-1] is best or worst
                if keys[-1] in subdir and "video" in subdir[keys[-1]]:
                    if not did_copy:
                        result = result.copy()  # pyright: ignore[reportAssignmentType]
                        did_copy = True
                    parent_dir = result
                    for key in keys[:-1]:
                        parent_dir[key] = parent_dir[key].copy()  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]  # fmt: skip
                        parent_dir = parent_dir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]  # fmt: skip
                    parent_dir = cast("_LogMetricsEvalEnvRunnersResultsDict", parent_dir)
                    video = subdir[keys[-1]]["video"]
                    if isinstance(video, list):
                        if len(video) > 1:
                            video = np.stack(video).squeeze()
                        else:
                            video = video[0]
                    assert isinstance(video, np.ndarray) and video.ndim == 5
                    parent_dir[keys[-1]] = cast("VideoTypes.Array5D", video)
        return result

    def log_trial_result(self, iteration: int, trial: Trial, result):
        super().log_trial_result(
            iteration,
            trial,
            self.preprocess_videos(result),
        )
