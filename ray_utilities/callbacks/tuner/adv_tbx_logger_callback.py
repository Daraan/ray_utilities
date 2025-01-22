from __future__ import annotations

import numpy as np
from ray.tune.logger import TBXLoggerCallback
from ray_utilities.constants import DEFAULT_VIDEO_KEYS

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial  # noqa: TC002


class AdvTBXLoggerCallback(TBXLoggerCallback):
    """TensorBoardX Logger.

    Note that hparams will be written only after a trial has terminated.
    This logger automatically flattens nested dicts to show on TensorBoard:

        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}

    Note:
        To log videos these conditions must hold:

            `isinstance(video, np.ndarray) and video.ndim == 5`
            and have the format "NTCHW"
    """

    _video_keys = DEFAULT_VIDEO_KEYS

    def log_trial_result(self, iteration: int, trial: "Trial", result: dict):
        # Check valid video
        has_videos = any(k in result for k in self._video_keys)
        if not has_videos:
            super().log_trial_result(iteration, trial, result)
            return
        result = result.copy()
        for k in self._video_keys:
            if k in result:
                video = result[k]["video"]
                if not (isinstance(video, np.ndarray) and video.ndim == 5):
                    # assume it is a list of videos; likely length 1
                    if len(video) != 1:
                        print("unexpected video shape", np.shape(video))
                    video = video[0]
                if not (isinstance(video, np.ndarray) and video.ndim == 5):
                    print("WARNING - video will not be logged as video to TBX because it is not a 5D numpy array")
                result[k] = video  # place ndarray in result dict
        super().log_trial_result(iteration, trial, result)
