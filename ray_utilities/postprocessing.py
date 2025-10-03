"""Post-processing utilities for Ray RLlib training results and metrics.

This module provides comprehensive utilities for processing, filtering, and formatting
Ray RLlib training results. It includes functions for metric filtering, video handling,
and result validation that are essential for clean experiment logging and analysis.

The module handles the complex nested structure of Ray RLlib results and provides
tools to extract only the relevant metrics while properly handling video data,
timers, and other specialized result types.

Key Features:
    - **Metric Filtering**: Extract only essential metrics from verbose RLlib results
    - **Video Processing**: Handle video recording and temporary file creation
    - **Result Validation**: Ensure result dictionaries match expected schemas
    - **Flexible Logging**: Support different levels of metric detail for various loggers

Main Functions:
    :func:`filter_metrics`: Extract essential metrics from RLlib results
    :func:`remove_unwanted_metrics`: Remove top-level unwanted keys
    :func:`create_log_metrics`: Create structured metrics for logging
    :func:`save_videos`: Convert video arrays to temporary files
    :func:`remove_videos`: Strip video data for JSON logging

Constants:
    :data:`RESULTS_TO_KEEP`: Core metrics to preserve during filtering
    :data:`RESULTS_TO_REMOVE`: Top-level keys to exclude from results

Example:
    Basic metric filtering for clean logging::

        from ray_utilities.postprocessing import filter_metrics, create_log_metrics

        # Get clean metrics from RLlib result
        filtered = filter_metrics(algorithm_result)

        # Create structured log metrics with video handling
        log_metrics = create_log_metrics(
            algorithm_result,
            save_video=True,
            log_stats="minimal",
        )

    Custom metric filtering::

        # Keep additional metrics beyond defaults
        extra_keys = [("custom_metric",), ("learner_results", "loss")]
        filtered = filter_metrics(result, extra_keys_to_keep=extra_keys)

See Also:
    :mod:`ray.rllib.utils.metrics`: Ray RLlib metrics definitions
    :mod:`ray_utilities.video.numpy_to_video`: Video file creation utilities
    :mod:`ray_utilities.constants`: Metric key constants and video configurations
"""
# pyright: enableExperimentalFeatures=true
# ruff: noqa: PLC0415  # imports at top level of file; safe import time if not needed.

from __future__ import annotations

import logging
import math
from copy import deepcopy
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    ParamSpec,
    TypedDict,
    TypeGuard,
    cast,
    overload,
)

import numpy as np
from ray.air.integrations.comet import CometLoggerCallback
from ray.rllib.core import DEFAULT_AGENT_ID, DEFAULT_MODULE_ID
from ray.rllib.utils.metrics import (
    ALL_MODULES,  # pyright: ignore[reportPrivateImportUsage]
    CONNECTOR_PIPELINE_TIMER,
    # CONNECTOR_TIMERS,  # subkey of env_to_module_connector
    ENV_RESET_TIMER,
    ENV_RUNNER_RESULTS,
    ENV_STEP_TIMER,
    ENV_TO_MODULE_CONNECTOR,
    ENV_TO_MODULE_SUM_EPISODES_LENGTH_IN,
    ENV_TO_MODULE_SUM_EPISODES_LENGTH_OUT,
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    EVALUATION_RESULTS,
    LEARNER_CONNECTOR,
    LEARNER_CONNECTOR_SUM_EPISODES_LENGTH_IN,
    LEARNER_CONNECTOR_SUM_EPISODES_LENGTH_OUT,
    LEARNER_RESULTS,
    MODULE_TO_ENV_CONNECTOR,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_TRAINED_LIFETIME,
    NUM_MODULE_STEPS_SAMPLED,
    NUM_MODULE_STEPS_SAMPLED_LIFETIME,
    NUM_MODULE_STEPS_TRAINED,
    RLMODULE_INFERENCE_TIMER,
    SAMPLE_TIMER,  # OldAPI
    TIME_BETWEEN_SAMPLING,
    TIMERS,
    WEIGHTS_SEQ_NO,  # Sequence Number of weights currently used. Increased +1 per update.
)
from typing_extensions import TypeVar

from ray_utilities.constants import (
    DEFAULT_VIDEO_DICT_KEYS,
    ENVIRONMENT_RESULTS,
    EPISODE_BEST_VIDEO,
    EPISODE_WORST_VIDEO,
)
from ray_utilities.misc import deep_update
from ray_utilities.temp_dir import TEMP_DIR_PATH
from ray_utilities.training.helpers import get_current_step, get_evaluation_results
from ray_utilities.typing.trainable_return import TrainableReturnData
from ray_utilities.video.numpy_to_video import create_temp_video

try:
    from wandb import Video  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    if TYPE_CHECKING:
        from wandb import Video  # pyright: ignore[reportMissingImports]
    else:

        class Video:  # need a type for isinstance
            pass


if TYPE_CHECKING:
    from collections import deque as Deque
    from collections.abc import Sequence

    from typing_extensions import TypeForm

    from ray_utilities.config.parser.default_argument_parser import LogStatsChoices
    from ray_utilities.typing.algorithm_return import (
        EvalEnvRunnersResultsDict,
        EvaluationResultsDict,
        StrictAlgorithmReturnData,
    )
    from ray_utilities.typing.common import VideoTypes
    from ray_utilities.typing.metrics import (
        AnyAutoExtendedLogMetricsDict,
        AnyLogMetricsDict,
        LogMetricsDict,
        NewLogMetricsDict,
        VideoMetricsDict,
        _LogMetricsEnvRunnersResultsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
        _LogMetricsEvaluationResultsDict,
        _LogMetricsEvaluationResultsWithoutDiscreteDict,
        _NewLogMetricsEvaluationResultsDict,
        _NewLogMetricsEvaluationResultsWithoutDiscreteDict,
    )

__all__ = ["RESULTS_TO_KEEP", "filter_metrics"]

# NOTE: This should not overlap!
RESULTS_TO_KEEP: set[tuple[str, ...]] = {
    (ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    # (ENV_RUNNER_RESULTS, NUM_EPISODES),
    (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN),
    ("comment",),
    ("trial_id",),
    ("training_iteration",),
    # Steps taken
    (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME),
}
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._other_results)
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._system_results)
RESULTS_TO_KEEP.update((key,) for key in CometLoggerCallback._exclude_results)
RESULTS_TO_KEEP.update((key,) for key in TrainableReturnData.__required_keys__)
assert all(isinstance(key, (tuple, list)) for key in RESULTS_TO_KEEP)
"""set[tuple[str, ...]]: Essential metric key paths to preserve during filtering.

This set defines the hierarchical paths of metrics that should be retained when
filtering Ray RLlib results. It includes:

- Core training and evaluation metrics (episode returns, step counts)
- System and metadata fields required by logging callbacks
- Trial identification and iteration tracking
- Required keys from trainable return data structures

The tuple format represents nested dictionary paths, e.g.,
``(EVALUATION_RESULTS, ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN)`` corresponds
to accessing ``result["evaluation"]["env_runner_results"]["episode_return_mean"]``.

Note:
    These keys should not overlap with :data:`RESULTS_TO_REMOVE` to avoid
    conflicting filtering behavior.

See Also:
    :func:`filter_metrics`: Uses this set for metric extraction
    :class:`ray.air.integrations.comet.CometLoggerCallback`: Defines additional required keys
"""

RESULTS_TO_REMOVE = {"fault_tolerance", NUM_AGENT_STEPS_SAMPLED_LIFETIME, LEARNER_RESULTS, TIMERS}
"""set[str]: Top-level result keys to remove when calling :func:`remove_unwanted_metrics`.

These keys contain internal Ray RLlib state or verbose debugging information that
is typically not needed for experiment analysis and logging. Removing them keeps
the result dictionaries clean and focused on essential metrics.
"""

_MISSING: Any = object()

_M = TypeVar("_M", bound=Mapping[Any, Any])
_D = TypeVar("_D", bound=dict[Any, Any])
_T = TypeVar("_T")
_TD = TypeVar("_TD", bound=TypedDict, default="TrainableReturnData")  # pyright: ignore[reportInvalidTypeForm]
_P = ParamSpec("_P")


_MetricDict = TypeVar("_MetricDict", "AnyAutoExtendedLogMetricsDict", "AnyLogMetricsDict")
"""Any LogMetricsDict or extensions thereof."""

_LogT = TypeVar("_LogT", bound="dict[str, dict[str, Any | dict[str, Any]]] | LogMetricsDict")
"""A nested dict structure or LogMetricsDict - not in the new layout."""

_logger = logging.getLogger(__name__)


def _find_item(obj: Mapping[str, Any], keys: Sequence[str]) -> Any:
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
    """Remove unwanted top-level keys from Ray RLlib results.

    This function filters out internal Ray RLlib keys that are typically not needed
    for experiment analysis, such as fault tolerance information, internal learner
    state, and verbose debugging data.

    Args:
        results: The results dictionary to filter.
        cast_to: Optional type to cast the result to. Used for type hinting only.

    Returns:
        A new dictionary with unwanted keys removed.

    Example:
        >>> results = {
        ...     "episode_return_mean": 150.0,
        ...     "fault_tolerance": {...},  # Will be removed
        ...     "training_iteration": 100,
        ... }
        >>> clean_results = remove_unwanted_metrics(results)
        >>> "fault_tolerance" in clean_results
        False

    See Also:
        :data:`RESULTS_TO_REMOVE`: Defines which keys are removed
        :func:`filter_metrics`: More comprehensive metric filtering
    """
    return {k: v for k, v in results.items() if k not in RESULTS_TO_REMOVE}  # type: ignore[return-type]


@overload
def filter_metrics(results: _D, extra_keys_to_keep: Optional[list[tuple[str, ...]]] = None) -> _D: ...


@overload
def filter_metrics(results: _M, extra_keys_to_keep: Optional[list[tuple[str, ...]]] = None) -> _M: ...


@overload
def filter_metrics(
    results: Mapping[str, Any],
    extra_keys_to_keep: Optional[list[tuple[str, ...]]] = None,
    *,
    cast_to: TypeForm[_T],
) -> _T: ...


def filter_metrics(
    results: _D | Mapping[Any, Any],
    extra_keys_to_keep: Optional[list[tuple[str, ...]]] = None,
    *,
    cast_to: TypeForm[_T] = _MISSING,  # noqa: ARG001
) -> _T | _D:
    """Extract essential metrics from Ray RLlib results using hierarchical key filtering.

    This function filters complex nested Ray RLlib result dictionaries to extract
    only the essential metrics defined in :data:`RESULTS_TO_KEEP` plus any additional
    keys specified. It's particularly useful for cleaning verbose RLlib results
    before logging or analysis.

    Args:
        results: The Ray RLlib results dictionary to filter. Can be the full
            algorithm training result or any nested metrics dictionary.
        extra_keys_to_keep: Additional hierarchical key paths to preserve beyond
            the default :data:`RESULTS_TO_KEEP`. Each tuple represents a path
            through the nested dictionary structure.
        cast_to: Optional type to cast the result to. Used for type hinting only.

    Returns:
        A new dictionary containing only the specified metrics. The structure
        matches the original nested hierarchy for the preserved keys.

    Example:
        Basic filtering with defaults only::

            >>> results = algorithm.train()  # Large nested dict
            >>> clean_metrics = filter_metrics(results)
            >>> # Contains only essential metrics like episode returns, step counts

        Adding custom metrics::

            >>> extra_keys = [
            ...     ("custom_metric",),  # Top-level custom metric
            ...     ("learner_results", "default_policy", "loss"),  # Nested loss metric
            ... ]
            >>> filtered = filter_metrics(results, extra_keys_to_keep=extra_keys)

        Type-safe filtering::

            >>> from ray_utilities.typing import LogMetricsDict
            >>> typed_metrics = filter_metrics(results, cast_to=LogMetricsDict)

    Raises:
        ValueError: If a key path already exists in the result structure with
            a different value (indicating conflicting metric definitions).

    Note:
        - The function preserves the nested structure of the original dictionary
        - Missing keys are silently ignored (no error raised)
        - The original results dictionary is not modified
        - Key paths are followed in order, creating nested dictionaries as needed

    See Also:
        :data:`RESULTS_TO_KEEP`: Default set of essential metric paths
        :func:`remove_unwanted_metrics`: Simpler top-level key removal
        :func:`create_log_metrics`: Higher-level metric structuring for logging
    """
    reduced = {}
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
    return reduced  # type: ignore[return-type]


@overload
def remove_videos(metrics: _MetricDict) -> _MetricDict: ...


@overload
def remove_videos(metrics: dict[Any, Any]) -> dict[Any, Any]: ...


# Caching not needed yet, this is especially for the json logger
# @cached(cache=FIFOCache(maxsize=1), key=cachetools.keys.methodkey, info=True)
def remove_videos(
    metrics: dict[str, Any] | AnyLogMetricsDict,
) -> dict[str, Any] | AnyLogMetricsDict:
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
                    parent_dir[key] = parent_dir[key].copy()  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
                    parent_dir = parent_dir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
                # parent_dir = cast("_LogMetricsEvaluationResultsDict", parent_dir)
                del parent_dir[keys[-1]]  # pyright: ignore[reportGeneralTypeIssues]
    return metrics


def save_videos(
    metrics: AnyLogMetricsDict | AnyAutoExtendedLogMetricsDict | StrictAlgorithmReturnData,
    dir=TEMP_DIR_PATH,
) -> None:
    """
    Attention:
        This modifies the metrics in place! If you want to keep the video as numpy array extract it first.

        Note that tensorboard uses gifs. WandB and Comet support multiple formats.
    """
    eval_dict = get_evaluation_results(metrics)
    if eval_dict is None:
        return
    discrete_results = cast(
        "_NewLogMetricsEvaluationResultsWithoutDiscreteDict | _LogMetricsEvaluationResultsWithoutDiscreteDict | None",
        eval_dict.get("discrete", None),
    )
    if discrete_results and ENV_RUNNER_RESULTS in discrete_results:
        discrete_results = discrete_results[ENV_RUNNER_RESULTS]
    video_dicts: list[
        EvalEnvRunnersResultsDict | _LogMetricsEvalEnvRunnersResultsDict | _NewLogMetricsEvaluationResultsDict
    ] = [eval_dict, discrete_results] if discrete_results else [eval_dict]
    video_value: VideoTypes.LogVideoTypes | VideoMetricsDict | None
    for video_dict in video_dicts:
        for key in (EPISODE_BEST_VIDEO, EPISODE_WORST_VIDEO):
            if (
                (video_value := video_dict.get(key, None))
                # skip if we already have a video path
                and not isinstance(video_value, Video)
                and "video_path" not in video_value
            ):
                if (
                    isinstance(video_value, dict)
                    and not video_value.get("video_path", False)
                    and not isinstance(video_value["video"], (str, Video))
                ):
                    # Set VideoPath
                    video_value["video_path"] = create_temp_video(video_value["video"], dir=dir)
                elif not isinstance(video_value, (str, dict)):
                    if isinstance(video_value, list) and len(video_value) == 0:
                        _logger.warning("Empty video list %s - skipping to save video", key)
                        continue
                    # No VideoMetricsDict present and not yet a video
                    _logger.warning(
                        "Overwritting video with path. Consider moving the video to a subkey %s : {'video': video}", key
                    )
                    assert key in video_dict
                    # Assign video to log dict (no list of numpy array anymore)
                    video_dict[key] = create_temp_video(video_value, dir=dir)  # pyright: ignore[reportGeneralTypeIssues]
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


def log_metrics_to_new_layout(metrics: LogMetricsDict) -> NewLogMetricsDict:
    """
    Changes:
        env_runners -> training
        evaluation.env_runners -> evaluation

        if only "__all_modules__" and "default_policy" are present in learners,
        merges them.
    """
    new_metrics = deepcopy(metrics)  # pyright: ignore[reportAssignmentType]
    new_metrics["training"] = cast("_LogMetricsEnvRunnersResultsDict", new_metrics.pop(ENV_RUNNER_RESULTS, {}))  # pyright: ignore[reportGeneralTypeIssues]
    if "current_step" in new_metrics:
        new_metrics.pop(NUM_ENV_STEPS_SAMPLED_LIFETIME, None)  # superseeded by current_step
    if EVALUATION_RESULTS in new_metrics:
        new_eval = new_metrics[EVALUATION_RESULTS]
        new_eval.update(new_eval.pop(ENV_RUNNER_RESULTS, {}))  # pyright: ignore[reportArgumentType, reportCallIssue]
        if "discrete" in new_eval:
            new_eval["discrete"] = new_eval["discrete"]
            new_eval["discrete"].update(new_eval["discrete"].pop(ENV_RUNNER_RESULTS, {}))
    if LEARNER_RESULTS in new_metrics and {ALL_MODULES, DEFAULT_MODULE_ID} == set(new_metrics[LEARNER_RESULTS].keys()):
        # merge the two
        merged = new_metrics[LEARNER_RESULTS][ALL_MODULES]
        merged.update(new_metrics[LEARNER_RESULTS][DEFAULT_MODULE_ID])
        new_metrics[LEARNER_RESULTS] = merged
    return cast("NewLogMetricsDict", new_metrics)


def create_log_metrics(
    result: StrictAlgorithmReturnData,
    *,
    save_video=False,
    discrete_eval: bool = False,
    log_stats: LogStatsChoices = "minimal",
) -> LogMetricsDict:
    """
    Filters the result of the Algorithm training step to only keep the relevant metrics.

    Args:
        result: The result dictionary from the algorithm
        save_video: If True the video will be saved to a temporary directory
            A new key "video_path" will be added to the video dict, or if the video is a numpy array
            the array will be replaced by the path.
        log_stats: Define how much the metrics should be reduced. See :obj:`LogStatsChoices`.
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
    environment_results = result[ENV_RUNNER_RESULTS].get(ENVIRONMENT_RESULTS, {})
    if environment_results:  # turn to a list
        s_seq: Deque = environment_results["seeds"]["seed_sequence"]
        environment_results["seeds"]["seed_sequence"] = list(s_seq)

    current_step = get_current_step(result)
    metrics: LogMetricsDict = {
        ENV_RUNNER_RESULTS: {
            EPISODE_RETURN_MEAN: result[ENV_RUNNER_RESULTS].get(
                EPISODE_RETURN_MEAN,
                float("nan"),
            ),
            NUM_ENV_STEPS_SAMPLED_LIFETIME: result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME],
        },
        EVALUATION_RESULTS: {
            ENV_RUNNER_RESULTS: {
                EPISODE_RETURN_MEAN: eval_mean,
            },
        },
        "training_iteration": result["training_iteration"],
        "done": result["done"],
        "current_step": current_step,
        "batch_size": result["config"]["_train_batch_size_per_learner"],
    }
    if discrete_eval:
        metrics[EVALUATION_RESULTS]["discrete"] = {
            ENV_RUNNER_RESULTS: {
                EPISODE_RETURN_MEAN: disc_eval_mean,
            },
        }

    if EVALUATION_RESULTS in result:
        # Check for NaN values, if they are not the evaluation metrics warn.
        if any(isinstance(value, float) and math.isnan(value) for value in metrics.values()):
            _logger.warning("NaN values in metrics: %s", metrics)

        # Store videos
        if evaluation_videos_best := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(
            EPISODE_BEST_VIDEO,
        ):
            metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_BEST_VIDEO] = {
                "video": evaluation_videos_best,
                # if we have a video but no max reward, use the mean as the guaranteed metric present
                "reward": result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(EPISODE_RETURN_MAX, eval_mean),
            }
        if evaluation_videos_worst := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(
            EPISODE_WORST_VIDEO,
        ):
            metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_WORST_VIDEO] = {
                "video": evaluation_videos_worst,
                "reward": result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(EPISODE_RETURN_MIN, eval_mean),
            }
        if discrete_evaluation_results := result[EVALUATION_RESULTS].get("discrete"):
            if discrete_evaluation_videos_best := discrete_evaluation_results[ENV_RUNNER_RESULTS].get(
                EPISODE_BEST_VIDEO
            ):
                metrics[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_BEST_VIDEO] = {  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    "video": discrete_evaluation_videos_best,
                    "reward": discrete_evaluation_results[ENV_RUNNER_RESULTS].get(EPISODE_RETURN_MAX, disc_eval_mean),
                }  # fmt: skip
            if discrete_evaluation_videos_worst := discrete_evaluation_results[ENV_RUNNER_RESULTS].get(
                EPISODE_WORST_VIDEO
            ):
                metrics[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][EPISODE_WORST_VIDEO] = {  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    "video": discrete_evaluation_videos_worst,
                    "reward": discrete_evaluation_results[ENV_RUNNER_RESULTS].get(EPISODE_RETURN_MIN, disc_eval_mean),
                }  # fmt: skip
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
    if log_stats == "minimal":
        if not math.isnan(eval_mean) and EVALUATION_RESULTS in result:
            # Add min/max
            for key in (EPISODE_RETURN_MIN, EPISODE_RETURN_MAX):
                if (
                    eval_min_max := result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(key, None)
                ) is not None and eval_min_max != eval_mean:
                    metrics[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][key] = eval_min_max
            if discrete_eval and "discrete" in result[EVALUATION_RESULTS]:
                disc_eval_results = result[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS]  # pyright: ignore[reportTypedDictNotRequiredAccess]
                for key in (EPISODE_RETURN_MIN, EPISODE_RETURN_MAX):
                    if (
                        eval_min_max := disc_eval_results.get(key, None)
                    ) is not None and eval_min_max != disc_eval_mean:
                        metrics[EVALUATION_RESULTS]["discrete"][ENV_RUNNER_RESULTS][key] = eval_min_max  # pyright: ignore[reportTypedDictNotRequiredAccess]
        # reorganize some keys to align with other log_stats
        metrics = _reorganize_timer_logs(metrics)  # pyright: ignore[reportArgumentType, reportAssignmentType]
        metrics.pop(TIMERS, None)
        _remove_less_interesting_keys(metrics)
        return metrics
    merged_result: LogMetricsDict = deep_update(result, metrics)  # type: ignore[return-value]
    # clean videos
    if EVALUATION_RESULTS in result:
        if not evaluation_videos_best:  # pyright: ignore[reportPossiblyUnboundVariable]
            merged_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(EPISODE_BEST_VIDEO, None)
        if not evaluation_videos_worst:  # pyright: ignore[reportPossiblyUnboundVariable]
            merged_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(EPISODE_WORST_VIDEO, None)
        if not discrete_eval:
            merged_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(EPISODE_BEST_VIDEO, None)
            merged_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(EPISODE_WORST_VIDEO, None)
    merged_result = _reorganize_connector_logs(merged_result)
    merged_result = _reorganize_timer_logs(merged_result)
    if log_stats == "all":
        return merged_result  # type: ignore[return-value]
    # Remove other unwanted metrics
    # log_stats in ("more", "timers", "learners", "timers+learners", "most")
    # Additional Keys:
    # {'date', 'num_env_steps_sampled_lifetime', 'config', 'node_ip', 'hostname',
    # 'iterations_since_restore', 'timestamp', 'time_this_iter_s', 'pid', 'time_since_restore',
    # 'time_total_s', 'num_training_step_calls_per_iteration', 'trial_id'}
    # Remove System stats:
    if log_stats != "most":
        for k in (
            # "time_since_restore",  # moved to timers
            "num_training_step_calls_per_iteration",  # if using algo.train more often before logging
            "iterations_since_restore",
            # "node_ip",  # autofilled
            # "hostname",  # autofilled
            # "pid",  # autofilled
            # "date",
            # "timestamp",  # autofilled
            # "time_this_iter_s",  # will be re-added
            # "time_total_s",  # will be re-added
            "num_env_steps_sampled_lifetime",  # superseded by current_step
        ):
            merged_result.pop(k)

    merged_result.pop("fault_tolerance")
    merged_result.pop(NUM_ENV_STEPS_SAMPLED_LIFETIME + "_throughput", None)  # Lifetime key has same information
    # merged_result[ENV_RUNNER_RESULTS].pop("num_healthy_workers", )
    # merged_result[ENV_RUNNER_RESULTS].pop("num_remote_worker_restarts", None)
    merged_result[ENV_RUNNER_RESULTS].pop(ENV_TO_MODULE_SUM_EPISODES_LENGTH_IN)
    merged_result[ENV_RUNNER_RESULTS].pop(ENV_TO_MODULE_SUM_EPISODES_LENGTH_OUT)
    merged_result[ENV_RUNNER_RESULTS].pop(WEIGHTS_SEQ_NO)
    if EVALUATION_RESULTS in result:
        merged_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(NUM_ENV_STEPS_SAMPLED_LIFETIME + "_throughput", None)
        merged_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(ENV_TO_MODULE_SUM_EPISODES_LENGTH_IN)
        merged_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(ENV_TO_MODULE_SUM_EPISODES_LENGTH_OUT)
        merged_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(WEIGHTS_SEQ_NO)
        merged_result[EVALUATION_RESULTS].pop("num_healthy_workers", None)
        merged_result[EVALUATION_RESULTS].pop("num_remote_worker_restarts", None)
        merged_result[EVALUATION_RESULTS].pop("actor_manager_num_outstanding_async_reqs", None)
    # most currently equivalent to timers+learners
    if "timers" not in log_stats and log_stats != "most":  # timers and timers+learners
        merged_result.pop(TIMERS)
    elif "timers" not in log_stats:
        _remove_less_interesting_timers(merged_result)

    if "learners" not in log_stats and log_stats != "most":  # learners and timers+learners
        merged_result.pop("learners", None)
    elif "learners" not in log_stats and LEARNER_RESULTS in merged_result:
        merged_result[LEARNER_RESULTS][ALL_MODULES].pop(LEARNER_CONNECTOR_SUM_EPISODES_LENGTH_IN)
        merged_result[LEARNER_RESULTS][ALL_MODULES].pop(LEARNER_CONNECTOR_SUM_EPISODES_LENGTH_OUT)
        merged_result[LEARNER_RESULTS][ALL_MODULES].pop(NUM_ENV_STEPS_TRAINED_LIFETIME + "_throughput")
        merged_result[LEARNER_RESULTS][ALL_MODULES].pop(NUM_MODULE_STEPS_TRAINED + "_throughput")
    _remove_less_interesting_keys(merged_result)
    return merged_result  # type: ignore[return-value]


def _reorganize_connector_logs(results: _LogT) -> _LogT:
    """Move timer results listed in env_runner to the timers key"""
    results.setdefault(TIMERS, {})
    assert TIMERS in results
    results[TIMERS][ENV_TO_MODULE_CONNECTOR] = results[ENV_RUNNER_RESULTS].pop(ENV_TO_MODULE_CONNECTOR)
    results[TIMERS][MODULE_TO_ENV_CONNECTOR] = results[ENV_RUNNER_RESULTS].pop(MODULE_TO_ENV_CONNECTOR)
    results[TIMERS][RLMODULE_INFERENCE_TIMER] = results[ENV_RUNNER_RESULTS].pop(RLMODULE_INFERENCE_TIMER)
    results[TIMERS][ENV_RESET_TIMER] = results[ENV_RUNNER_RESULTS].pop(ENV_RESET_TIMER)
    results[TIMERS][ENV_STEP_TIMER] = results[ENV_RUNNER_RESULTS].pop(ENV_STEP_TIMER)
    if LEARNER_RESULTS in results:
        results[TIMERS][LEARNER_CONNECTOR] = results[LEARNER_RESULTS][ALL_MODULES].pop(LEARNER_CONNECTOR)
    results[TIMERS][LEARNER_CONNECTOR].update(results[TIMERS][LEARNER_CONNECTOR].pop(TIMERS))  # pyright: ignore[reportAttributeAccessIssue]
    if EVALUATION_RESULTS in results and results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get(ENV_TO_MODULE_CONNECTOR):
        evaluation_timers: dict[str, Any] = results[TIMERS].setdefault(EVALUATION_RESULTS, {})  # pyright: ignore[reportAssignmentType]
        evaluation_timers[ENV_TO_MODULE_CONNECTOR] = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(
            ENV_TO_MODULE_CONNECTOR
        )
        evaluation_timers[MODULE_TO_ENV_CONNECTOR] = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(
            MODULE_TO_ENV_CONNECTOR
        )
        evaluation_timers[RLMODULE_INFERENCE_TIMER] = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(
            RLMODULE_INFERENCE_TIMER
        )
        evaluation_timers[ENV_STEP_TIMER] = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(ENV_STEP_TIMER)
        evaluation_timers[ENV_RESET_TIMER] = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(ENV_RESET_TIMER)
    return results


def _reorganize_timer_logs(results: _LogT) -> _LogT:
    results.setdefault(TIMERS, {})
    assert TIMERS in results
    for key in ("time_since_restore", "time_total_s"):
        if key in results:
            results[TIMERS][key] = results.pop(key)
    if "time_this_iter_s" in results:
        results[TIMERS]["time_this_iter_s"] = results["time_this_iter_s"]  # autofilled
    results[TIMERS].setdefault(ENV_RUNNER_RESULTS, {})
    # if sample amount is very low, e.g. during debugging, _done_episodes_for_metrics is empty
    # this results in keys missings for episodes
    for key in (EPISODE_DURATION_SEC_MEAN, TIME_BETWEEN_SAMPLING, SAMPLE_TIMER):
        if key in results[ENV_RUNNER_RESULTS]:
            results[TIMERS][ENV_RUNNER_RESULTS][key] = results[ENV_RUNNER_RESULTS].pop(key)
    if EVALUATION_RESULTS in results and len(results[EVALUATION_RESULTS]) > 1:
        evaluation_timers: dict[str, Any] = results[TIMERS].setdefault(EVALUATION_RESULTS, {})
        assert EVALUATION_RESULTS not in evaluation_timers
        evaluation_timers[ENV_RUNNER_RESULTS] = {}
        # NOTE: Earlier we kept EPISODE_DURATION_SEC_MEAN at nan for first step
        for key in (EPISODE_DURATION_SEC_MEAN, TIME_BETWEEN_SAMPLING, SAMPLE_TIMER):
            if key in results[ENV_RUNNER_RESULTS]:
                evaluation_timers[ENV_RUNNER_RESULTS][key] = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].pop(key)
        if not evaluation_timers[ENV_RUNNER_RESULTS]:
            results[TIMERS].pop(EVALUATION_RESULTS)
    if not results[TIMERS][ENV_RUNNER_RESULTS]:
        results[TIMERS].pop(ENV_RUNNER_RESULTS)
    if not results[TIMERS]:
        results.pop(TIMERS)
    return results


def _remove_small_values(d: dict[Any, Any], threshold: float = 1e-4) -> dict[Any, Any]:
    """
    Recursively remove all key-value pairs from a nested dictionary where the value is a float
    and less than the given threshold.

    Args:
        d: The dictionary to process.
        threshold: The minimum value to keep.

    Returns:
        A new dictionary with small float values removed.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = _remove_small_values(v, threshold)
            if nested:
                result[k] = nested
        elif isinstance(v, float):
            if abs(v) >= threshold:
                result[k] = v
        else:
            result[k] = v
    return result


def _remove_less_interesting_timers(metrics: dict[str, Any] | LogMetricsDict, remove_below=1e-4) -> None:
    # Assume reorganized
    if TIMERS not in metrics:
        return
    clean_metrics = [metrics[TIMERS]]
    if EVALUATION_RESULTS in metrics[TIMERS]:
        clean_metrics.append(metrics[TIMERS][EVALUATION_RESULTS])

    timer_dict: dict[str, dict]
    for timer_dict in clean_metrics:
        for key in (ENV_TO_MODULE_CONNECTOR, MODULE_TO_ENV_CONNECTOR):
            try:
                pipeline_timer = timer_dict[key][CONNECTOR_PIPELINE_TIMER]
            except KeyError as e:
                _logger.warning("Cannot access %s in %s: %s", CONNECTOR_PIPELINE_TIMER, key, e)
                continue
            timer_dict[key].clear()
            timer_dict[key][CONNECTOR_PIPELINE_TIMER] = pipeline_timer

    learner_connector_timer = metrics[TIMERS][LEARNER_CONNECTOR][CONNECTOR_PIPELINE_TIMER]
    remove_masked_samples_timer = metrics[TIMERS][LEARNER_CONNECTOR].get("remove_masked_samples_connector", None)
    metrics[TIMERS][LEARNER_CONNECTOR].clear()
    metrics[TIMERS][LEARNER_CONNECTOR][CONNECTOR_PIPELINE_TIMER] = learner_connector_timer
    if remove_masked_samples_timer:
        metrics[TIMERS][LEARNER_CONNECTOR]["remove_masked_samples_connector"] = remove_masked_samples_timer
    metrics[TIMERS] = _remove_small_values(metrics[TIMERS], threshold=remove_below)


def _remove_less_interesting_keys(metrics: LogMetricsDict | _LogMetricsEvaluationResultsDict) -> None:
    """Remove some duplicated and less interesting keys."""
    metrics.pop("env_runner_group", None)  # just env_runner_group/actor_manager_num_outstanding_async_reqs
    if metrics.get("num_training_step_calls_per_iteration", 0) == 1:
        metrics.pop("num_training_step_calls_per_iteration", None)

    env_runner_results = metrics.get(ENV_RUNNER_RESULTS, None)
    if env_runner_results:
        for env_key, mod_key, id_key in (
            (NUM_ENV_STEPS_SAMPLED, NUM_MODULE_STEPS_SAMPLED, DEFAULT_MODULE_ID),
            (NUM_ENV_STEPS_SAMPLED_LIFETIME, NUM_MODULE_STEPS_SAMPLED_LIFETIME, DEFAULT_MODULE_ID),
            (NUM_ENV_STEPS_SAMPLED_LIFETIME, NUM_AGENT_STEPS_SAMPLED_LIFETIME, DEFAULT_AGENT_ID),
            (NUM_ENV_STEPS_SAMPLED, NUM_AGENT_STEPS_SAMPLED, DEFAULT_AGENT_ID),
        ):
            if (
                env_key in env_runner_results
                and mod_key in env_runner_results
                and len(env_runner_results[mod_key]) == 1
                and env_runner_results[mod_key][id_key] == env_runner_results[env_key]
            ):
                env_runner_results.pop(mod_key)
        for metric_key, id_key in (
            ("module_episode_return_mean", DEFAULT_MODULE_ID),
            ("agent_episode_return_mean", DEFAULT_AGENT_ID),
        ):
            if (
                metric_key in env_runner_results
                and len(env_runner_results[metric_key]) == 1
                and math.isclose(
                    env_runner_results[metric_key].get(id_key, 0.0),
                    env_runner_results[EPISODE_RETURN_MEAN],
                )
            ):
                env_runner_results.pop(metric_key)
    if (evaluation_results := metrics.get(EVALUATION_RESULTS, None)) and len(
        evaluation_results[ENV_RUNNER_RESULTS]
    ) > 1:
        _remove_less_interesting_keys(evaluation_results)
    learner_results = metrics.get(LEARNER_RESULTS, None)
    if learner_results:
        # remove __all_modules if duplicated
        modules: dict[str, dict[str, Any]] = {
            k: v for k, v in learner_results.items() if isinstance(v, dict)
        }  # best guess
        for module_id, module_results in modules.items():
            if module_id == ALL_MODULES or not isinstance(module_id, dict):
                continue
            for key in module_results.keys():
                if key in learner_results[ALL_MODULES] and (
                    module_results[key] == learner_results[ALL_MODULES][key]
                    or (
                        key.endswith("_throughput")
                        and math.isclose(module_results[key], learner_results[ALL_MODULES][key], abs_tol=0.2)
                    )
                ):
                    module_results.pop(key, None)


def verify_keys(metrics: Mapping[Any, Any], typ: type[_TD], *, test_optional: bool = True) -> TypeGuard[_TD]:
    if not all(k in metrics for k in typ.__required_keys__):
        missing = set(typ.__required_keys__) - set(metrics.keys())
        _logger.error("Required keys missing from %r: %s", typ, missing)
        return False
    if test_optional:
        if not all(k in metrics for k in typ.__optional_keys__):
            missing = set(typ.__optional_keys__) - set(metrics.keys())
    return True


def verify_return(return_type: type[_TD]):
    """
    Verify the required keys of the return type are present in the return value.

    Attention:
        It is not guranteed that all required keys are present at runtime
        in the __required_keys__ attribute.
        `TypedDict`s that are checked should prefer using `total=True`
        over `total=False` with `NotRequired`.
    """

    def decorator(func: Callable[_P, _TD]) -> Callable[_P, _TD]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            verify_keys(result, return_type)
            return result

        return wrapper

    return decorator
