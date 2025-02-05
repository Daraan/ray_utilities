from __future__ import annotations

# pyright: enableExperimentalFeatures=true

from typing import TYPE_CHECKING, Annotated
from typing_extensions import TypedDict, Required, Never, NotRequired

from . import _PEP_728_AVAILABLE
from .algorithm_return import _EvaluationNoDiscreteDict, EvaluationResultsDict

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "LogMetricsDict",
]

if _PEP_728_AVAILABLE or TYPE_CHECKING:

    class VideoMetricsDict(TypedDict, closed=True):
        video: NDArray
        reward: float

    class _VideosToEnvRunners(TypedDict):
        episode_videos_best: NotRequired[Annotated[Never, "needs to be in env_runners"]]
        episode_videos_worst: NotRequired[Annotated[Never, "needs to be in env_runners"]]

    class _LogMetricsEnvRunnersResultsDict(TypedDict):
        episode_return_mean: float

    class _LogMetricsEvalEnvRunnersResultsDict(_LogMetricsEnvRunnersResultsDict, total=False):
        episode_videos_best: NDArray | VideoMetricsDict
        episode_videos_worst: NDArray | VideoMetricsDict

    class _LogMetricsEvaluationResultsWithoutDiscreteDict(_EvaluationNoDiscreteDict, _VideosToEnvRunners):
        env_runners: Required[_LogMetricsEvalEnvRunnersResultsDict]  # pyright: ignore[reportIncompatibleVariableOverride]
        discrete: NotRequired[Never]

    class _LogMetricsEvaluationResultsDict(EvaluationResultsDict, _VideosToEnvRunners):
        env_runners: Required[_LogMetricsEvalEnvRunnersResultsDict]  # pyright: ignore[reportIncompatibleVariableOverride]
        discrete: _LogMetricsEvaluationResultsWithoutDiscreteDict  # pyright: ignore[reportIncompatibleVariableOverride]

    class LogMetricsDict(TypedDict):
        env_runners: _LogMetricsEnvRunnersResultsDict
        evaluation: _LogMetricsEvaluationResultsDict
        training_iteration: int

    FlatLogMetricsDict = TypedDict(
        "FlatLogMetricsDict",
        {
            "training_iteration": int,
            "evaluation/env_runners/episode_return_mean": float,
            "evaluation/env_runners/episode_videos_best/reward": NotRequired[float],
            "evaluation/env_runners/episode_videos_best/video": NotRequired["NDArray"],
            "evaluation/env_runners/episode_videos_best": NotRequired["VideoMetricsDict | NDArray"],
            "evaluation/env_runners/episode_videos_worst/reward": NotRequired[float],
            "evaluation/env_runners/episode_videos_worst/video": NotRequired["NDArray"],
            "evaluation/env_runners/episode_videos_worst": NotRequired["VideoMetricsDict | NDArray"],
            "env_runners/episode_return_mean": float,
            "evaluation/discrete/env_runners/episode_return_mean": float,
            "evaluation/discrete/env_runners/episode_videos_best/reward": NotRequired[float],
            "evaluation/discrete/env_runners/episode_videos_best/video": NotRequired["NDArray"],
            "evaluation/discrete/env_runners/episode_videos_best": NotRequired["VideoMetricsDict | NDArray"],
            "evaluation/discrete/env_runners/episode_videos_worst/reward": NotRequired[float],
            "evaluation/discrete/env_runners/episode_videos_worst/video": NotRequired["NDArray"],
            "evaluation/discrete/env_runners/episode_videos_worst": NotRequired["VideoMetricsDict | NDArray"],
        },
    )
else:
    LogMetricsDict = dict
    FlatLogMetricsDict = dict
