import importlib.metadata
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import typing_extensions
from packaging.version import Version
from typing_extensions import TypeAliasType

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium.envs.registration import EnvSpec as _EnvSpec
    from gymnax.environments import environment as environment_gymnax
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig

_PEP_728_AVAILABLE = getattr(typing_extensions, "_PEP_728_IMPLEMENTED", False) or Version(
    importlib.metadata.version("typing-extensions")
) >= Version("4.13")

ExtraItems = Any  # float | int | str | bool | None | dict[str, "_ExtraItems"] | NDArray[Any] | Never
"""ExtraItems for TypedDict"""

# Below requires _PEP_728_AVAILABLE
# ruff: noqa: E402
from .algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData
from .metrics import FlatLogMetricsDict, LogMetricsDict
from .trainable_return import TrainableReturnData, RewardsDict, RewardUpdaters

if typing_extensions.TYPE_CHECKING:
    from ray_utilities.setup.experiment_base import ExperimentSetupBase

__all__ = [
    "AlgorithmReturnData",
    "CometStripedVideoFilename",
    "FlatLogMetricsDict",
    "FunctionalTrainable",
    "LogMetricsDict",
    "RewardUpdaters",
    "RewardsDict",
    "StrictAlgorithmReturnData",
    "TestModeCallable",
    "TrainableReturnData",
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
