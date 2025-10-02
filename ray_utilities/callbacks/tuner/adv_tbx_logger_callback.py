from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from ray.tune.logger import TBXLoggerCallback

from ray_utilities.callbacks.tuner.new_style_logger_callback import LogMetricsDictT, NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS, FORK_FROM

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

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

    def _make_forked_trial_suffix(self, trial: "Trial", fork_data: ForkFromData | str) -> str:
        return fork_data if isinstance(fork_data, str) else self.make_forked_trial_id(trial, fork_data)

    def _setup_forked_trial(self, trial: "Trial", fork_data: ForkFromData):
        """Setup trial logging, handling forked trials by creating new subdirectory."""
        # Close current writer and clean tracks
        self.log_trial_end(trial)
        # Cannot continue files, add suffix information to file name
        if trial in self._trial_writer:
            self._trial_writer[trial].close()
        trial.init_local_path()
        self._trial_writer[trial] = self._summary_writer_cls(
            trial.local_path,
            flush_secs=30,
            filename_suffix=self._make_forked_trial_suffix(trial, fork_data),
        )
        self._trial_result[trial] = {}

    def log_trial_start(self, trial: Trial):
        if trial in self._trial_writer and FORK_FROM in trial.config:
            assert self.should_restart_logging(trial)
        if FORK_FROM in trial.config:
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


if TYPE_CHECKING:  # Check ABC
    AdvTBXLoggerCallback()
