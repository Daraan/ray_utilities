from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from ray.tune.logger import TBXLoggerCallback

from ray_utilities.callbacks.tuner.new_style_logger_callback import LogMetricsDictT, NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS, FORK_FROM

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

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

    def log_trial_start(self, trial: "Trial"):
        """Start logging for a trial, handling forked trials."""
        if trial in self._trial_writer:
            self._trial_writer[trial].close()

        trial.init_local_path()

        # Check if this is a forked trial
        is_forked = FORK_FROM in trial.config

        if is_forked:
            # For forked trials, optionally copy parent's TensorBoard data
            fork_data = trial.config[FORK_FROM]
            parent_id = fork_data["parent_id"]

            # Try to find parent trial's TensorBoard directory
            parent_tb_path = self._find_parent_tb_dir(trial, parent_id)

            if parent_tb_path and parent_tb_path.exists():
                logger.info(
                    "Trial %s forked from %s. Copying parent TensorBoard data from %s",
                    trial.trial_id,
                    parent_id,
                    parent_tb_path,
                )
                # Copy all event files from parent to new trial directory
                try:
                    for event_file in parent_tb_path.glob("events.out.tfevents.*"):
                        shutil.copy2(event_file, trial.local_path)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to copy parent TensorBoard data for trial %s: %s",
                        trial.trial_id,
                        e,
                    )
            else:
                logger.info(
                    "Trial %s forked from %s. Parent TensorBoard data not found, starting fresh.",
                    trial.trial_id,
                    parent_id,
                )

        # Create new writer for this trial
        self._trial_writer[trial] = self._summary_writer_cls(trial.local_path, flush_secs=30)
        self._trial_result[trial] = {}

    def _find_parent_tb_dir(self, trial: "Trial", parent_id: str) -> Path | None:
        """Find the TensorBoard directory of the parent trial.

        Searches in the trial's experiment directory for a trial with matching ID.
        """
        try:
            experiment_path = Path(trial.local_path).parent
            # Look for parent trial directory
            for trial_dir in experiment_path.iterdir():
                if trial_dir.is_dir() and parent_id in trial_dir.name:
                    return trial_dir
        except Exception as e:  # noqa: BLE001
            logger.warning("Error finding parent TensorBoard dir for trial %s: %s", trial.trial_id, e)
        return None

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
