from __future__ import annotations

# pyright: enableExperimentalFeatures=true
from functools import partial
import logging
import math
from typing import Any, Callable, Mapping, Optional, TypeVar, overload, TYPE_CHECKING

from ray.air.integrations.comet import CometLoggerCallback
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    EVALUATION_RESULTS,
)

from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS, EPISODE_BEST_VIDEO, EPISODE_WORST_VIDEO

if TYPE_CHECKING:
    from ray_utilities.typing import StrictAlgorithmReturnData, LogMetricsDict
    from ray_utilities.typing.algorithm_return import EvaluationResultsDict
    from typing_extensions import TypeForm

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
        logging.warning("Reduced results do not match the expected amount of keys: %s", reduced)

    return reduced  # type: ignore[return-type]


@overload
def remove_videos(metrics: LogMetricsDict) -> LogMetricsDict: ...


@overload
def remove_videos(metrics: dict[Any, Any]) -> dict: ...


# Caching not needed yet, this is especially for the json logger
# @cached(cache=FIFOCache(maxsize=1), key=cachetools.keys.methodkey, info=True)
def remove_videos(
    metrics: dict[Any, Any] | LogMetricsDict,
) -> dict | LogMetricsDict:
    """Removes video keys from the metrics"""
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
                if not did_copy:
                    metrics = metrics.copy()
                    did_copy = True
                parent_dir = metrics
                for key in keys[:-1]:
                    parent_dir[key] = parent_dir[key].copy()  # pyright: ignore[reportGeneralTypeIssues]
                    parent_dir = parent_dir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
                del parent_dir[keys[-1]]  # pyright: ignore[reportGeneralTypeIssues]
    return metrics


def strip_videos_metadata(metrics: dict[Any, Any] | LogMetricsDict) -> dict | LogMetricsDict:
    """
    Pops the video from `episode_video_best : {video: np.ndarray, reward: float}`
    to `episode_video_best : np.ndarray`
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
            if (
                keys[-1] in subdir and "video" in subdir[keys[-1]]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
            ):
                if not did_copy:
                    metrics = metrics.copy()
                    did_copy = True
                parent_dir = metrics
                for key in keys[:-1]:
                    parent_dir[key] = parent_dir[key].copy()  # pyright: ignore[reportGeneralTypeIssues]
                    parent_dir = parent_dir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
                parent_dir[keys[-1]] = subdir[keys[-1]]["video"]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]  # fmt: skip
    return metrics


def _old_strip_metadata_from_flat_metrics(result: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    result = result.copy()
    for k in DEFAULT_VIDEO_DICT_KEYS:
        if k in result:
            video = result[k]["video"]
            if not (isinstance(video, np.ndarray) and video.ndim == 5):
                # assume it is a list of videos; likely length 1
                if len(video) != 1:
                    logging.warning("unexpected video shape %s", np.shape(video))
                video = video[0]
            if not (isinstance(video, np.ndarray) and video.ndim == 5):
                logging.error("Video will not be logged as video to TBX because it is not a 5D numpy array")
            result[k] = video  # place ndarray in result dict
    return result


def create_log_metrics(result: dict[str, Any] | StrictAlgorithmReturnData) -> LogMetricsDict:
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
    }

    if EVALUATION_RESULTS in result:
        # Store videos
        if evaluation_videos_best := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(
            "episode_videos_best",
        ):
            metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_BEST_VIDEO] = {
                "video": evaluation_videos_best,
                "reward": result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MAX],
            }
        if evaluation_videos_worst := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(
            "episode_videos_worst",
        ):
            metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_WORST_VIDEO] = {
                "video": evaluation_videos_worst,
                "reward": result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN],
            }
        if discrete_evaluation_results := result[EVALUATION_RESULTS].get("discrete"):
            if discrete_evaluation_videos_best := discrete_evaluation_results[ENV_RUNNER_RESULTS].get(
                "episode_videos_best"
            ):
                metrics[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_BEST_VIDEO] = {
                    "video": discrete_evaluation_videos_best,
                    "reward": discrete_evaluation_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MAX],
                }
            if discrete_evaluation_videos_worst := discrete_evaluation_results[ENV_RUNNER_RESULTS].get(
                "episode_videos_worst"
            ):
                metrics[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_WORST_VIDEO] = {
                    "video": discrete_evaluation_videos_worst,
                    "reward": discrete_evaluation_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN],
                }
        # Check for NaN values, if they are not the evaluation metrics warn.
        if any(isinstance(value, float) and math.isnan(value) for value in metrics.values()):
            logging.warning("NaN values in metrics: %s", metrics)
    return metrics


def update_running_reward(new_reward: float, reward_array: list[float]) -> float:
    if not math.isnan(new_reward):
        reward_array.append(new_reward)
    running_reward = sum(reward_array[-100:]) / (min(100, len(reward_array)) or float("nan"))  # nan for 0
    return running_reward


def create_running_reward_updater() -> Callable[[float], float]:
    return partial(update_running_reward, reward_array=[])
