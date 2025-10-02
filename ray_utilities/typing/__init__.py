"""Type definitions and aliases for Ray Utilities components.

Provides type definitions for Ray RLlib return data, metrics dictionaries,
trainable configurations, and other commonly used types throughout the package.

Main Type Aliases:
    - :data:`AlgorithmReturnData`: Ray RLlib algorithm training results
    - :data:`TrainableReturnData`: Ray Tune trainable return structure
    - :data:`LogMetricsDict`: Structured metrics for logging frameworks
    - :data:`RewardsDict`: Episode reward tracking structure

Example:
    >>> from ray_utilities.typing import AlgorithmReturnData
    >>> def process_results(results: AlgorithmReturnData) -> float:
    ...     return results["env_runner_results"]["episode_return_mean"]

See Also:
    :mod:`ray_utilities.typing.algorithm_return`: Algorithm-specific return types
    :mod:`ray_utilities.typing.trainable_return`: Trainable return structures
    :mod:`ray_utilities.typing.metrics`: Metrics and logging type definitions
    :mod:`typing_extensions`: Extended typing capabilities
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, NamedTuple

import typing_extensions
from typing_extensions import NotRequired, TypeAliasType, TypedDict, TypeVar

from .algorithm_return import (
    AlgorithmReturnData,
    EnvRunnersResultsDict,
    EvaluationResultsDict,
    StrictAlgorithmReturnData,
)
from .common import (
    BaseEnvRunnersResultsDict,
    BaseEvaluationResultsDict,
    VideoTypes,
)
from .metrics import FlatLogMetricsDict, LogMetricsDict
from .trainable_return import RewardsDict, RewardUpdaters, TrainableReturnData

if TYPE_CHECKING:
    import gymnasium as gym
    import ray.tune
    from gymnasium.envs.registration import EnvSpec as _EnvSpec
    from gymnax.environments import environment as environment_gymnax
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.tune.experiment import Trial
    from ray.tune.search.sample import Domain

    from ray_utilities.setup.experiment_base import ExperimentSetupBase


__all__ = [
    "AlgorithmReturnData",
    "BaseEnvRunnersResultsDict",
    "BaseEvaluationResultsDict",
    "CometStripedVideoFilename",
    "EnvRunnersResultsDict",
    "EvaluationResultsDict",
    "FlatLogMetricsDict",
    "FunctionalTrainable",
    "LogMetricsDict",
    "RewardUpdaters",
    "RewardsDict",
    "StrictAlgorithmReturnData",
    "TestModeCallable",
    "TrainableReturnData",
    "VideoTypes",
]


CometStripedVideoFilename = Literal[
    "evaluation_best_video",
    "evaluation_discrete_best_video",
    "evaluation_worst_video",
    "evaluation_discrete_worst_video",
]

FunctionalTrainable = typing_extensions.TypeAliasType(
    "FunctionalTrainable", Callable[[dict[str, Any]], TrainableReturnData]
)

_ESB_co = TypeVar("_ESB_co", bound="ExperimentSetupBase[Any, AlgorithmConfig, Algorithm]", covariant=True)

TestModeCallable = typing_extensions.TypeAliasType(
    "TestModeCallable", Callable[[FunctionalTrainable, _ESB_co], TrainableReturnData], type_params=(_ESB_co,)
)

EnvSpec = TypeAliasType("EnvSpec", "str | _EnvSpec")
EnvType = TypeAliasType("EnvType", "gym.Env | environment_gymnax.Environment | gym.vector.VectorEnv")

_Domain = TypeVar("_Domain", bound="Domain", default="Domain")
_AnyT = TypeVar("_AnyT", default=Any)
ParameterSpace = TypeAliasType(
    "ParameterSpace",
    dict[Literal["grid_search"] | str, Iterable[_AnyT]] | _Domain,  # noqa: PYI051
    type_params=(_AnyT, _Domain),
)
"""Describes a tune.Domain or grid_search for a parameter sampling by tune"""

StopperType = TypeAliasType("StopperType", "Mapping | ray.tune.Stopper | Callable[[str, Mapping], bool] | None")


class Forktime(NamedTuple):
    time_attr: str
    """String to describe what the :attr:`time` tracks, e.g. ``training_iteration`` or ``current_step``"""
    time: int | float
    """Current time of the fork, tracked in :attr:`time_attr` units"""


class ForkFromData(TypedDict):
    parent_id: str
    """Trial id of the run to fork"""

    parent_training_iteration: int
    """Training iteration the fork is at. This is needed for example for WandB's fork_from feature"""

    parent_time: Forktime
    """
    Current time the fork is at.

    First element of the :class:`Forktime` is a string that describes the unit,
    second the numerical value of the time
    """

    parent_trial: NotRequired[Trial]
    """
    If available, the actual Trial object of the run to fork

    Attention:
        The trial might be modified during the tuning process
        as itself could be forked at some time.
    """

    parent_fork_id: NotRequired[str]
    """If available, the fork_id of the parent trial, if it was forked itself"""

    fork_id_this_trial: NotRequired[str]
    """The fork_id this trial will get. It is constructed from the other fields."""

    controller: NotRequired[str | Any]
    """A field to inform who decided the forking, e.g. scheduler name"""
