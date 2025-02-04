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

    class _VideoMetricsDict(TypedDict, closed=True):
        video: NDArray
        reward: float

    class _VideosToEnvRunners(TypedDict):
        episode_videos_best: NotRequired[Annotated[Never, "needs to be in env_runners"]]
        episode_videos_worst: NotRequired[Annotated[Never, "needs to be in env_runners"]]

    class _LogMetricsEnvRunnersResultsDict(TypedDict):
        episode_return_mean: float

    class _LogMetricsEvalEnvRunnersResultsDict(_LogMetricsEnvRunnersResultsDict, total=False):
        episode_videos_best: NDArray | _VideoMetricsDict
        episode_videos_worst: NDArray | _VideoMetricsDict

    class _LogMetricsEvaluationResultsWithoutDiscreteDict(_EvaluationNoDiscreteDict, _VideosToEnvRunners):
        env_runners: Required[_LogMetricsEvalEnvRunnersResultsDict]  # pyright: ignore[reportIncompatibleVariableOverride]
        discrete: NotRequired[Never]

    class _LogMetricsEvaluationResultsDict(EvaluationResultsDict, _VideosToEnvRunners):
        env_runners: Required[_LogMetricsEvalEnvRunnersResultsDict]  # pyright: ignore[reportIncompatibleVariableOverride]
        discrete: _LogMetricsEvaluationResultsWithoutDiscreteDict  # pyright: ignore[reportIncompatibleVariableOverride]

    class LogMetricsDict(TypedDict):
        env_runners: _LogMetricsEnvRunnersResultsDict
        evaluation: _LogMetricsEvaluationResultsDict
else:
    LogMetricsDict = dict
