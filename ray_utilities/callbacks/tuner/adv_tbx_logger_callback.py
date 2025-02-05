from __future__ import annotations
from typing import TYPE_CHECKING
import logging

from ray.tune.logger import TBXLoggerCallback

from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS
from ray_utilities.postprocessing import strip_videos_metadata

if TYPE_CHECKING:
    from ray_utilities.typing.metrics import LogMetricsDict
    from ray.tune.experiment.trial import Trial


logger = logging.getLogger(__name__)


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

    _video_keys = DEFAULT_VIDEO_DICT_KEYS

    def log_trial_result(self, iteration: int, trial: "Trial", result: dict | LogMetricsDict):
        super().log_trial_result(
            iteration,
            trial,
            strip_videos_metadata(result),  # pyright: ignore[reportArgumentType]
        )
