from __future__ import annotations

import logging
import math

# pyright: enableExperimentalFeatures=true
import os
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from ray.air.integrations.comet import CometLoggerCallback
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    EVALUATION_RESULTS,
)
from wandb import Video

from ray_utilities.constants import (
    DEFAULT_VIDEO_DICT_KEYS,
    EPISODE_BEST_VIDEO,
    EPISODE_WORST_VIDEO,
)
from ray_utilities.temp import TEMP_DIR_PATH
from ray_utilities.video.numpy_to_video import create_temp_video

if TYPE_CHECKING:
    from typing_extensions import TypeForm

    from ray_utilities.typing import LogMetricsDict, StrictAlgorithmReturnData
    from ray_utilities.typing.algorithm_return import EvaluationResultsDict
    from ray_utilities.typing.metrics import (
        AutoExtendedLogMetricsDict,
        VideoMetricsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
    )

__all__ = ["RESULTS_TO_KEEP", "filter_metrics"]

# NOTE: This should not overlap!
RESULTS_TO_KEEP = {
    (ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    # (ENV_RUNNER_RESULTS, NUM_EPISODES),
    (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    ("comment",),
}
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._other_results)
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._system_results)
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._exclude_results)
assert all(isinstance(key, (tuple, list)) for key in RESULTS_TO_KEEP)

RESULTS_TO_REMOVE = {"fault_tolerance", "num_agent_steps_sampled_lifetime", "learners", "timers"}

_MISSING: Any = object()

_M = TypeVar("_M", bound=Mapping[Any, Any])
_T = TypeVar("_T")

_MetricDict = TypeVar("_MetricDict", "AutoExtendedLogMetricsDict", "LogMetricsDict")

_logger = logging.getLogger(__name__)


def _find_item(obj: Mapping[str, Any], keys: list[str]) -> Any:
    if len(keys) == 1:
        return obj.get(keys[0], _MISSING)
    value = obj.get(keys[0], _MISSING)
    if isinstance(value, dict):
        return _find_item(value, keys[1:])
    if value is not _MISSING and len(keys) > 0:
        raise TypeError(f"Expected dict at {keys[0]} but got {value}")
    return value


@overload
def remove_unwanted_metrics(results: _M) -> _M: ...


@overload
def remove_unwanted_metrics(results: Mapping[Any, Any], *, cast_to: TypeForm[_T]) -> _T: ...


def remove_unwanted_metrics(results: _M, *, cast_to: TypeForm[_T] = _MISSING) -> _T | _M:  # noqa: ARG001
    """
    Removes unwanted top-level keys from the results.

    See:
    - `RESULTS_TO_REMOVE`
    """
    return {k: v for k, v in results.items() if k not in RESULTS_TO_REMOVE}  # type: ignore[return-type]


@overload
def filter_metrics(results: _M, extra_keys_to_keep: Optional[list[tuple[str, ...]]] = None) -> _M: ...


@overload
def filter_metrics(
    results: Mapping[str, Any], extra_keys_to_keep: Optional[list[tuple[str, ...]]] = None, *, cast_to: TypeForm[_T]
) -> _T: ...


def filter_metrics(
    results: _M,
    extra_keys_to_keep: Optional[list[tuple[str, ...]]] = None,
    *,
    cast_to: TypeForm[_T] = _MISSING,  # noqa: ARG001
) -> _M | _T:
    """Reduces the metrics to only keep `RESULTS_TO_KEEP` and `extra_keys_to_keep`."""
    reduced: dict[str, Any] = {}
    _count = 0
    if extra_keys_to_keep:
        keys_to_keep = RESULTS_TO_KEEP.copy()
        keys_to_keep.update(extra_keys_to_keep)
    else:
        keys_to_keep = RESULTS_TO_KEEP

    for keys in keys_to_keep:
        value = _find_item(results, keys if not isinstance(keys, str) else [keys])
        if value is not _MISSING:
            sub_dir = reduced
            for key in keys[:-1]:
                sub_dir = sub_dir.setdefault(key, {})
            if keys[-1] in sub_dir:
                raise ValueError(f"Key {keys[-1]} already exists in {sub_dir}")
            sub_dir[keys[-1]] = value
            _count += 1
    if _count != len(RESULTS_TO_KEEP):
        _logger.warning("Reduced results do not match the expected amount of keys: %s", reduced)

    return reduced  # type: ignore[return-type]


@overload
def remove_videos(metrics: _MetricDict) -> _MetricDict: ...


@overload
def remove_videos(metrics: dict[Any, Any]) -> dict: ...


# Caching not needed yet, this is especially for the json logger
# @cached(cache=FIFOCache(maxsize=1), key=cachetools.keys.methodkey, info=True)
def remove_videos(
    metrics: dict[Any, Any] | LogMetricsDict,
) -> dict | LogMetricsDict:
    """
    Removes video keys from the metrics

    This is especially for the json logger
    """
    did_copy = False
    for keys in DEFAULT_VIDEO_DICT_KEYS:
        subdir = metrics
        for key in keys[:-1]:
            if key not in subdir:
                break
            subdir = subdir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
        else:
            # Perform a selective deep copy on the modified items
            if keys[-1] in subdir:
                # subdir = cast("dict[str, VideoMetricsDict]", subdir)
                if not did_copy:
                    metrics = metrics.copy()
                    did_copy = True
                parent_dir = metrics
                for key in keys[:-1]:
                    parent_dir[key] = parent_dir[key].copy()  # pyright: ignore[reportGeneralTypeIssues]
                    parent_dir = parent_dir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
                # parent_dir = cast("_LogMetricsEvaluationResultsDict", parent_dir)
                del parent_dir[keys[-1]]  # pyright: ignore[reportGeneralTypeIssues]
    return metrics


def save_videos(
    metrics: LogMetricsDict | AutoExtendedLogMetricsDict,
    dir=TEMP_DIR_PATH,
) -> None:
    """
    Attention:
        This modifies the metrics in place! If you want to keep the video as numpy array extract it first.

        Note that tensorboard uses gifs. WandB and Comet support multiple formats.
    """
    if EVALUATION_RESULTS not in metrics:
        return
    eval_dict = metrics[EVALUATION_RESULTS]
    discrete_results = eval_dict.get("discrete", None)
    video_dicts = (
        [eval_dict[ENV_RUNNER_RESULTS], discrete_results[ENV_RUNNER_RESULTS]]
        if discrete_results
        else [eval_dict[ENV_RUNNER_RESULTS]]
    )
    for video_dict in video_dicts:
        for key in (EPISODE_BEST_VIDEO, EPISODE_WORST_VIDEO):
            if (
                key in video_dict
                # skip if we already have a video path
                and "video_path" not in video_dict[key]  # pyright: ignore[reportTypedDictNotRequiredAccess]
            ):
                value = video_dict[key]  # pyright: ignore[reportTypedDictNotRequiredAccess]
                if (
                    isinstance(value, dict)
                    and not value.get("video_path", False)
                    and not isinstance(value["video"], str)
                ):
                    # Set VideoPath
                    value["video_path"] = create_temp_video(value["video"], dir=dir)
                elif not isinstance(value, (str, dict)):
                    # No VideoMetricsDict present and not yet a video
                    _logger.warning(
                        "Overwritting video with path. Consider moving the video to a subkey %s : {'video': video}", key
                    )
                    video_dict[key] = create_temp_video(value, dir=dir)
                # else already str or VideoMetricsDict with a str


@overload
def check_if_video(  # pyright: ignore[reportOverlappingOverload]
    video: list[Any], video_name: str = ...
) -> TypeGuard[list[np.ndarray[tuple[int, int, int, int, int], np.dtype[np.uint8]]]]: ...


@overload
def check_if_video(
    video: Any, video_name: str = ...
) -> TypeGuard[np.ndarray[tuple[int, int, int, int, int], np.dtype[np.uint8]]]: ...


def check_if_video(
    video: Any, video_name: str = ""
) -> TypeGuard[
    np.ndarray[tuple[int, int, int, int, int], np.dtype[np.uint8]]
    | list[np.ndarray[tuple[int, int, int, int, int], np.dtype[np.uint8]]]
]:
    if isinstance(video, list):
        if len(video) != 1:
            _logger.warning("unexpected video shape %s", np.shape(video))
        video = video[0]
    if not (isinstance(video, np.ndarray) and video.ndim == 5):
        _logger.error("%s Video will not be logged as video to TBX because it is not a 5D numpy array", video_name)
        return False
    return True


def _old_strip_metadata_from_flat_metrics(result: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    result = result.copy()
    for k in DEFAULT_VIDEO_DICT_KEYS:
        if k in result:
            video = result[k]["video"]
            if not (isinstance(video, np.ndarray) and video.ndim == 5):
                # assume it is a list of videos; likely length 1
                if len(video) != 1:
                    _logger.warning("unexpected video shape %s", np.shape(video))
                video = video[0]
            check_if_video(video)
            result[k] = video  # place ndarray in result dict
    return result


def create_log_metrics(result: StrictAlgorithmReturnData, *, save_video=False) -> LogMetricsDict:
    """
    Filters the result of the Algorithm training step to only keep the relevant metrics.

    Args:
        result: The result dictionary from the algorithm
        save_video: If True the video will be saved to a temporary directory
            A new key "video_path" will be added to the video dict, or if the video is a numpy array
            the array will be replaced by the path.
    """
    # NOTE: The csv logger will only log keys that are present in the first result,
    #       i.e. the videos will not be logged if they are added later; but overtwise everytime!
    if EVALUATION_RESULTS in result:
        evaluation_results: EvaluationResultsDict = result[EVALUATION_RESULTS]
        eval_mean: float = evaluation_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]

        if "discrete" in evaluation_results:
            disc_eval_mean = evaluation_results["discrete"][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        else:
            disc_eval_mean = float("nan")
    else:
        eval_mean = float("nan")
        disc_eval_mean = float("nan")

    metrics: LogMetricsDict = {
        ENV_RUNNER_RESULTS: {
            EPISODE_RETURN_MEAN: result[ENV_RUNNER_RESULTS].get(
                EPISODE_RETURN_MEAN,
                float("nan"),
            )
        },
        EVALUATION_RESULTS: {
            ENV_RUNNER_RESULTS: {
                EPISODE_RETURN_MEAN: eval_mean,
            },
            "discrete": {
                ENV_RUNNER_RESULTS: {
                    EPISODE_RETURN_MEAN: disc_eval_mean,
                },
            },
        },
        "training_iteration": result["training_iteration"],
    }

    if EVALUATION_RESULTS in result:
        # Check for NaN values, if they are not the evaluation metrics warn.
        if any(isinstance(value, float) and math.isnan(value) for value in metrics.values()):
            logging.warning("NaN values in metrics: %s", metrics)

        # Store videos
        if evaluation_videos_best := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(
            EPISODE_BEST_VIDEO,
        ):
            metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_BEST_VIDEO] = {
                "video": evaluation_videos_best,
                "reward": result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MAX],
            }
        if evaluation_videos_worst := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(
            EPISODE_WORST_VIDEO,
        ):
            metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_WORST_VIDEO] = {
                "video": evaluation_videos_worst,
                "reward": result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN],
            }
        if discrete_evaluation_results := result[EVALUATION_RESULTS].get("discrete"):
            if discrete_evaluation_videos_best := discrete_evaluation_results[ENV_RUNNER_RESULTS].get(
                EPISODE_BEST_VIDEO
            ):
                metrics[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_BEST_VIDEO] = {
                    "video": discrete_evaluation_videos_best,
                    "reward": discrete_evaluation_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MAX],
                }
            if discrete_evaluation_videos_worst := discrete_evaluation_results[ENV_RUNNER_RESULTS].get(
                EPISODE_WORST_VIDEO
            ):
                metrics[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_WORST_VIDEO] = {
                    "video": discrete_evaluation_videos_worst,
                    "reward": discrete_evaluation_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN],
                }
            if discrete_evaluation_videos_best:
                check_if_video(discrete_evaluation_videos_best, "discrete" + EPISODE_BEST_VIDEO)
            if discrete_evaluation_videos_worst:
                check_if_video(discrete_evaluation_videos_worst, "discrete" + EPISODE_WORST_VIDEO)
        if evaluation_videos_best:
            check_if_video(evaluation_videos_best, EPISODE_BEST_VIDEO)
        if evaluation_videos_worst:
            check_if_video(evaluation_videos_worst, EPISODE_WORST_VIDEO)
        if save_video:
            save_videos(metrics)
    return metrics


def update_running_reward(new_reward: float, reward_array: list[float]) -> float:
    if not math.isnan(new_reward):
        reward_array.append(new_reward)
    running_reward = sum(reward_array[-100:]) / (min(100, len(reward_array)) or float("nan"))  # nan for 0
    return running_reward


def create_running_reward_updater() -> Callable[[float], float]:
    return partial(update_running_reward, reward_array=[])
