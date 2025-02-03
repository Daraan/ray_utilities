# Currently cannot use variables, even if final or literal, with TypedDict
# pyright: enableExperimentalFeatures=true
from __future__ import annotations
from typing_extensions import TypedDict, Required, Never, NotRequired
from typing import TYPE_CHECKING

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

    class _TrainableReturnDataWOEnvRunners(TypedDict, total=False, closed=False, extra_items=_ExtraItems):
        done: Required[bool]
        evaluation: EvaluationResultsDict
        env_runners: Required[EnvRunnersResultsDict] | NotRequired[EnvRunnersResultsDict]
        # Present in rllib results
        training_iteration: int
        timestamp: int

        should_checkpoint: bool

        comment: str
        trial_id: int | str
        time_this_iter_s: float
        episodes_total: int
        episodes_this_iter: int
        learners: dict[str, dict[str, float | int]]
        timers: dict[str, float]

        # System results
        date: str
        node_ip: str
        hostname: str
        pid: int

    @type_check_only
    class TrainableReturnData(
        _TrainableReturnDataWOEnvRunners, _NotRequiredEnvRunners, total=False, closed=False, extra_items=_ExtraItems
    ):
        ...

    @type_check_only
    class StrictTrainableReturnData(
        _TrainableReturnDataWOEnvRunners, _RequiredEnvRunners, total=False, closed=False, extra_items=_ExtraItems
    ):
        ...
