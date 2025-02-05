# Currently cannot use variables, even if final or literal, with TypedDict
# Uses PEP 728 not yet released in typing_extensions
# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Never, NotRequired, Required, TypedDict

from . import ExtraItems, _PEP_728_AVAILABLE

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "AlgorithmReturnData",
    "EnvRunnersResultsDict",
    "EvaluationResultsDict",
    "StrictAlgorithmReturnData",
]

if _PEP_728_AVAILABLE or TYPE_CHECKING:

    class EnvRunnersResultsDict(TypedDict, closed=False):
        episode_return_mean: float
        episode_return_max: float
        episode_return_min: float

    class EvalEnvRunnersResultsDict(EnvRunnersResultsDict, total=False, extra_items=ExtraItems):
        episode_videos_best: NDArray
        episode_videos_worst: NDArray

    class _EvaluationNoDiscreteDict(TypedDict, extra_items=ExtraItems):
        env_runners: EvalEnvRunnersResultsDict
        discrete: NotRequired[Never]

    class EvaluationResultsDict(TypedDict, total=False, extra_items=ExtraItems):
        env_runners: Required[EvalEnvRunnersResultsDict]
        discrete: _EvaluationNoDiscreteDict

    class _RequiredEnvRunners(TypedDict, total=False, closed=False):
        env_runners: Required[EnvRunnersResultsDict]

    class _NotRequiredEnvRunners(TypedDict, total=False, closed=False):
        env_runners: NotRequired[EnvRunnersResultsDict]

    class _AlgoReturnDataWithoutEnvRunners(TypedDict, total=False, extra_items=ExtraItems):
        done: Required[bool]
        evaluation: EvaluationResultsDict
        env_runners: Required[EnvRunnersResultsDict] | NotRequired[EnvRunnersResultsDict]
        # Present in rllib results
        training_iteration: int

        should_checkpoint: bool

        comment: str
        trial_id: int | str
        episodes_total: int
        episodes_this_iter: int
        learners: dict[str, dict[str, float | int]]

        # Times
        timers: dict[str, float]
        timestamp: int
        time_total_s: float
        time_this_iter_s: float

        # System results
        date: str
        node_ip: str
        hostname: str
        pid: int

    class AlgorithmReturnData(
        _AlgoReturnDataWithoutEnvRunners, _NotRequiredEnvRunners, total=False, extra_items=ExtraItems
    ): ...

    class StrictAlgorithmReturnData(  # pyright: ignore[reportIncompatibleVariableOverride]
        _AlgoReturnDataWithoutEnvRunners, _RequiredEnvRunners, total=False, extra_items=ExtraItems
    ):
        """Return data with env_runners present"""

else:
    # PEP 728 not yet released in typing_extensions
    AlgorithmReturnData = dict
    StrictAlgorithmReturnData = dict
    EvaluationResultsDict = dict
    EnvRunnersResultsDict = dict
    _EvaluationNoDiscreteDict = dict
