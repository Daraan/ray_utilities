# Currently cannot use variables, even if final or literal, with TypedDict
# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Never, NotRequired, Required, TypedDict

if TYPE_CHECKING:
    from typing import type_check_only
    _ExtraItems = float | int | str | bool | None | dict[str, "_ExtraItems"]

    @type_check_only
    class EnvRunnersResultsDict(TypedDict, extra_items=_ExtraItems):
        episode_return_mean: float
        episode_return_max: float
        episode_return_min: float

    @type_check_only
    class _EvaluationNoDiscreteDict(TypedDict, extra_items=_ExtraItems):
        env_runners: EnvRunnersResultsDict
        discrete: Never

    @type_check_only
    class EvaluationResultsDict(TypedDict, total=False, extra_items=_ExtraItems):
        env_runners: Required[EnvRunnersResultsDict]
        discrete: _EvaluationNoDiscreteDict

    class _RequiredEnvRunners(TypedDict, total=False, closed=False):
        env_runners: Required[EnvRunnersResultsDict]

    class _NotRequiredEnvRunners(TypedDict, total=False, closed=False):
        env_runners: NotRequired[EnvRunnersResultsDict]

    class _AlgoReturnDataWithoutEnvRunners(TypedDict, total=False, closed=False, extra_items=_ExtraItems):
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

    @type_check_only
    class AlgorithmReturnData(
        _AlgoReturnDataWithoutEnvRunners, _NotRequiredEnvRunners, total=False, closed=False, extra_items=_ExtraItems
    ):
        ...

    @type_check_only
    class StrictAlgorithmReturnData(
        _AlgoReturnDataWithoutEnvRunners, _RequiredEnvRunners, total=False, closed=False, extra_items=_ExtraItems
    ):
        """Return data with env_runners present"""
