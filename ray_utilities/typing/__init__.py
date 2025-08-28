"""Type definitions and aliases for Ray Utilities components.

This module provides comprehensive type definitions used throughout the Ray Utilities
package. It includes type aliases for Ray RLlib return data, metrics dictionaries,
trainable configurations, and other commonly used types.

The types are designed to provide better IDE support, type checking, and
documentation for the complex data structures used in Ray Tune and RLlib experiments.

**Key Type Categories:**

- **Return Data Types**: Structured return values from algorithms and trainables
- **Metrics Types**: Dictionary structures for logging and metrics
- **Configuration Types**: Experiment and algorithm configuration types  
- **Callable Types**: Function signatures for trainables and callbacks

**Main Type Aliases:**
    :data:`AlgorithmReturnData`: Ray RLlib algorithm training results
    :data:`TrainableReturnData`: Ray Tune trainable return structure
    :data:`LogMetricsDict`: Structured metrics for logging frameworks
    :data:`RewardsDict`: Episode reward tracking structure

Example:
    Using type annotations for better code clarity::
    
        from ray_utilities.typing import AlgorithmReturnData, LogMetricsDict
        
        def process_training_results(results: AlgorithmReturnData) -> LogMetricsDict:
            # Type checker knows the structure of results
            episode_return = results["env_runner_results"]["episode_return_mean"]
            return {"training_return": episode_return}
            
    Type-safe trainable definition::
    
        from ray_utilities.typing import TrainableReturnData
        from ray.tune import Trainable
        
        class MyTrainable(Trainable):
            def step(self) -> TrainableReturnData:
                # Return structure is type-checked
                return {
                    "episode_return_mean": 100.0,
                    "training_iteration": self.iteration,
                    "done": False
                }

Compatibility:
    This module requires modern typing features and may use experimental
    typing extensions for the latest type annotation capabilities.

See Also:
    :mod:`ray_utilities.typing.algorithm_return`: Algorithm-specific return types
    :mod:`ray_utilities.typing.trainable_return`: Trainable return structures
    :mod:`ray_utilities.typing.metrics`: Metrics and logging type definitions
    :mod:`typing_extensions`: Extended typing capabilities
"""

import importlib.metadata
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Iterable, Literal

import typing_extensions
from packaging.version import Version
from typing_extensions import TypeAliasType, TypeVar

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium.envs.registration import EnvSpec as _EnvSpec
    from gymnax.environments import environment as environment_gymnax
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig

_PEP_728_AVAILABLE = getattr(typing_extensions, "_PEP_728_IMPLEMENTED", False) or Version(
    importlib.metadata.version("typing-extensions")
) >= Version("4.13")

ExtraItems = Any  # float | int | str | bool | None | dict[str, "_ExtraItems"] | NDArray[Any] | Never
"""type: Type alias for additional items that can be included in TypedDict structures.

This flexible type allows for various data types that might be included in
experiment results, metrics, or configuration dictionaries beyond the
strictly typed required fields.
"""

# Below requires _PEP_728_AVAILABLE
# ruff: noqa: E402
from .algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData
from .metrics import FlatLogMetricsDict, LogMetricsDict
from .trainable_return import RewardsDict, RewardUpdaters, TrainableReturnData

if typing_extensions.TYPE_CHECKING:
    from ray.tune.search.sample import Domain

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

_Domain = TypeVar("_Domain", bound="Domain", default="Domain")
_AnyT = TypeVar("_AnyT", default=Any)
ParameterSpace = TypeAliasType(
    "ParameterSpace",
    dict[Literal["grid_search"] | str, Iterable[_AnyT]] | _Domain,  # noqa: PYI051
    type_params=(_AnyT, _Domain),
)
"""Describes a tune.Domain or grid_search for a parameter sampling by tune"""
