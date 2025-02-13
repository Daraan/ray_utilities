from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, TypeGuard

from typing_extensions import Never, NotRequired, Required, TypedDict

from .algorithm_return import EvaluationResultsDict, _EvaluationNoDiscreteDict

# pyright: enableExperimentalFeatures=true


if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "LogMetricsDict",
]

_LOG_METRICS_VIDEO_TYPES: TypeAlias = "list[NDArray] | str"


class VideoMetricsDict(TypedDict, closed=True):
    video: _LOG_METRICS_VIDEO_TYPES
    """
    A 5D numpy array representing a video; or a string pointing to a video file to upload.

    Should be a list of a 5D numpy video array representing a video.
    """
    reward: float
    video_path: NotRequired[str]
    """If a video file already exists for re-use, this is the path to the video file."""


class _WarnVideosToEnvRunners(TypedDict):
    episode_videos_best: NotRequired[Annotated[Never, "needs to be in env_runners"]]
    episode_videos_worst: NotRequired[Annotated[Never, "needs to be in env_runners"]]


class _LogMetricsEnvRunnersResultsDict(TypedDict):
    episode_return_mean: float
    epioode_return_max: NotRequired[float]
    epioode_return_min: NotRequired[float]


class _LogMetricsEvalEnvRunnersResultsDict(_LogMetricsEnvRunnersResultsDict, total=False):
    """
    Either a 5D Numpy array representing a video, or a dict with "video" and "reward" keys,
    representing the video, or a string pointing to a video file to upload.
    """

    episode_videos_best: _LOG_METRICS_VIDEO_TYPES | VideoMetricsDict
    episode_videos_worst: _LOG_METRICS_VIDEO_TYPES | VideoMetricsDict


class _LogMetricsEvaluationResultsWithoutDiscreteDict(_EvaluationNoDiscreteDict, _WarnVideosToEnvRunners):
    env_runners: Required[_LogMetricsEvalEnvRunnersResultsDict]  # pyright: ignore[reportIncompatibleVariableOverride]
    discrete: NotRequired[Never]


class _LogMetricsEvaluationResultsDict(EvaluationResultsDict, _WarnVideosToEnvRunners):
    env_runners: _LogMetricsEvalEnvRunnersResultsDict  # pyright: ignore[reportIncompatibleVariableOverride]
    discrete: NotRequired[_LogMetricsEvaluationResultsWithoutDiscreteDict]  # pyright: ignore[reportIncompatibleVariableOverride]


class LogMetricsDict(TypedDict):
    env_runners: _LogMetricsEnvRunnersResultsDict
    evaluation: _LogMetricsEvaluationResultsDict
    training_iteration: int
    """The number of times train.report() has been called"""

    done: bool


class AutoExtendedLogMetricsDict(LogMetricsDict):
    """
    Auto filled in keys after train.report.

    Use this in Callbacks

    See Also:
        - https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        - Trial.last_result
    """

    done: bool
    training_iteration: int  # auto filled in
    """The number of times train.report() has been called"""
    trial_id: int | str  # auto filled in


FlatLogMetricsDict = TypedDict(
    "FlatLogMetricsDict",
    {
        "training_iteration": int,
        "env_runners/episode_return_mean": float,
        "evaluation/env_runners/episode_return_mean": float,
        "evaluation/env_runners/episode_videos_best": NotRequired["str | NDArray"],
        "evaluation/env_runners/episode_videos_best/reward": NotRequired[float],
        "evaluation/env_runners/episode_videos_best/video": NotRequired["NDArray"],
        "evaluation/env_runners/episode_videos_best/video_path": NotRequired["str"],
        "evaluation/env_runners/episode_videos_worst": NotRequired["NDArray | str"],
        "evaluation/env_runners/episode_videos_worst/reward": NotRequired[float],
        "evaluation/env_runners/episode_videos_worst/video": NotRequired["NDArray"],
        "evaluation/env_runners/episode_videos_worst/video_path": NotRequired["str"],
        "evaluation/discrete/env_runners/episode_return_mean": float,
        "evaluation/discrete/env_runners/episode_videos_best": NotRequired["str | NDArray"],
        "evaluation/discrete/env_runners/episode_videos_best/reward": NotRequired[float],
        "evaluation/discrete/env_runners/episode_videos_best/video": NotRequired["NDArray"],
        "evaluation/discrete/env_runners/episode_videos_best/video_path": NotRequired["str"],
        "evaluation/discrete/env_runners/episode_videos_worst": NotRequired["str | NDArray"],
        "evaluation/discrete/env_runners/episode_videos_worst/reward": NotRequired[float],
        "evaluation/discrete/env_runners/episode_videos_worst/video": NotRequired["NDArray"],
        "evaluation/discrete/env_runners/episode_videos_worst/video_path": NotRequired["str"],
    },
    closed=False,
)

## TypeGuards


def has_video_key(
    dir: dict | _LogMetricsEvalEnvRunnersResultsDict, video_key: Literal["episode_videos_best", "episode_videos_worst"]
) -> TypeGuard[_LogMetricsEvalEnvRunnersResultsDict]:
    return video_key in dir
