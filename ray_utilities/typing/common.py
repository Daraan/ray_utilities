"""Common base types for unified type hierarchy across Ray Utilities.

This module provides base types that unify common structures between
algorithm return data and metrics logging types, eliminating redundancy
while maintaining compatibility with existing code.

The base types defined here are inherited by both:
- `ray_utilities.typing.algorithm_return`: Types for raw algorithm results
- `ray_utilities.typing.metrics`: Types for processed logging metrics

Base Types:
    :class:`BaseEnvRunnersResultsDict`: Core environment runner metrics
    :class:`BaseEvaluationResultsDict`: Core evaluation structure
"""

# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ray.rllib.utils.typing import AgentID, ModuleID
    from wandb import Video  # pyright: ignore[reportMissingImports]

__all__ = [
    "BaseEnvRunnersResultsDict",
    "BaseEvaluationResultsDict",
    "CommonVideoTypes",
]


class BaseEnvRunnersResultsDict(TypedDict):
    """Base type for environment runner results.

    Contains core metrics shared between algorithm return and metrics types.
    """

    episode_return_mean: float
    """Always required - primary performance metric"""


class CommonVideoTypes:
    """Common video type definitions used across the type hierarchy."""

    # Basic video types for algorithm returns
    BasicVideoList: TypeAlias = "list[NDArray]"
    """Simple list of video arrays for algorithm return data"""

    # Advanced video types for logging metrics
    Shape4D = tuple[int, int, int, int]  # (B, C, H, W)
    Array4D: TypeAlias = "NDArray[Shape4D]"  # shape=(B, C, H, W)
    Shape5D = tuple[int, int, int, int, int]  # (N, T, C, H, W)
    Array5D: TypeAlias = "NDArray[Shape5D]"  # shape=(N, T, C, H, W)

    LogVideoTypes: TypeAlias = "list[Array4D | Array5D] | Array5D | str | Video"
    """Advanced video types for metrics logging including Video objects and file paths"""


class BaseEvaluationResultsDict(TypedDict, total=False):
    """Base evaluation results structure."""

    evaluated_this_step: NotRequired[bool]
    """Whether evaluation was performed in this training step"""
