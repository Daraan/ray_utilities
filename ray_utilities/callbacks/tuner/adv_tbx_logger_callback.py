from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ray.tune.logger import TBXLoggerCallback

from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import (
        AutoExtendedLogMetricsDict,
        VideoMetricsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
    )


logger = logging.getLogger(__name__)


class AdvTBXLoggerCallback(TBXLoggerCallback):
    """TensorBoardX Logger.

    Note that hparams will be written only after a trial has terminated.
    This logger automatically flattens nested dicts to show on TensorBoard:

        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}

    Attention:
        To log videos these conditions must holdf for the video value

            `isinstance(video, np.ndarray) and video.ndim == 5`
            and have the format "NTCHW"

        Videos will be logged as gif
    """

    _video_keys = DEFAULT_VIDEO_DICT_KEYS

    @staticmethod
    def preprocess_videos(result: dict[str, Any] | AutoExtendedLogMetricsDict) -> dict[Any, Any]:
        """
        For tensorboard it must hold that:

        `isinstance(video, np.ndarray) and video.ndim == 5`
        """
        did_copy = False
        for keys in DEFAULT_VIDEO_DICT_KEYS:
            subdir = result
            for key in keys[:-1]:
                if key not in subdir:
                    break
                subdir = subdir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
            else:
                # Perform a selective deep copy on the modified items
                subdir = cast("dict[str, VideoMetricsDict]", subdir)
                if keys[-1] in subdir and "video" in subdir[keys[-1]]:
                    if not did_copy:
                        result = result.copy()
                        did_copy = True
                    parent_dir = result
                    for key in keys[:-1]:
                        parent_dir[key] = parent_dir[key].copy()  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
                        parent_dir = parent_dir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
                    parent_dir = cast("_LogMetricsEvalEnvRunnersResultsDict", parent_dir)
                    video = subdir[keys[-1]]["video"]
                    if isinstance(video, list):
                        if len(video) > 1:
                            video = np.stack(video).squeeze()
                        else:
                            video = video[0]
                    assert isinstance(video, np.ndarray) and video.ndim == 5
                    parent_dir[keys[-1]] = video  # pyright: ignore[reportGeneralTypeIssues]
        return result  # type: ignore[return-value]

    def log_trial_result(self, iteration: int, trial: "Trial", result: dict[str, Any] | AutoExtendedLogMetricsDict):
        super().log_trial_result(
            iteration,
            trial,
            self.preprocess_videos(result),
        )
