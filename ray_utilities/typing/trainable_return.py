from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from typing_extensions import NotRequired, Required, TypedDict

from .algorithm_return import _NotRequiredEnvRunners

if TYPE_CHECKING:
    from .metrics import _LogMetricsEvaluationResultsDict

# pyright: enableExperimentalFeatures=true


class TrainableReturnData(_NotRequiredEnvRunners, total=False, closed=False):
    """The return type of the trainable's step method."""

    evaluation: Required[_LogMetricsEvaluationResultsDict]
    training_iteration: int
    done: Required[bool]
    current_step: Required[int]
    comment: str
    trial_id: int | str


# Further


class RewardsDict(TypedDict):
    """A dictionary containing the rewards for the current training step."""

    running_reward: float
    running_eval_reward: float
    eval_mean: float
    disc_eval_reward: NotRequired[float]
    disc_eval_mean: NotRequired[float]
    disc_running_eval_reward: NotRequired[float]


class RewardUpdater(Protocol):
    """A protocol for a partial function that updates rewards."""

    def __call__(self, new_reward: float) -> float: ...

    keywords: Final[TypedDict[{"reward_array": list[float]}]]


class RewardUpdaters(TypedDict, extra_items=RewardUpdater):
    """A dictionary containing the reward updaters for the current training step."""

    # The keys are the names of the reward updaters, and the values are the functions that update the rewards.
    # The functions take a float as input and return a float.

    running_reward: RewardUpdater
    eval_reward: RewardUpdater
    disc_eval_reward: NotRequired[RewardUpdater]
