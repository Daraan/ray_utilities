"""Common base types for unified type hierarchy across Ray Utilities.

This module provides base types that unify common structures between
algorithm return data and metrics logging types, eliminating redundancy
while maintaining compatibility with existing code.

The base types defined here are inherited by both:
- `ray_utilities.typing.algorithm_return`: Types for raw algorithm results
- `ray_utilities.typing.metrics`: Types for processed logging metrics

Key Benefits:
    - Eliminates duplicate type definitions
    - Provides clear inheritance hierarchy
    - Maintains backward compatibility
    - Enables type-safe transformations between raw and processed data

Base Types:
    :class:`BaseEnvRunnersResultsDict`: Core environment runner metrics
    :class:`BaseEvaluationResultsDict`: Core evaluation structure
    :class:`BaseVideoTypes`: Video data type definitions
"""

# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from enum import Enum
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
    "VideoArrayShape",
]


class VideoArrayShape(Enum):
    """Enum for video array shape dimensions used across the type hierarchy."""

    SHAPE_4D = (4, "B", "C", "H", "W")  # (Batch, Channels, Height, Width)
    SHAPE_5D = (5, "N", "T", "C", "H", "W")  # (N_videos, Time, Channels, Height, Width)

    def __init__(self, ndim: int, *dim_names: str):
        self.ndim = ndim
        self.dim_names = dim_names

    @property
    def description(self) -> str:
        """Get a human-readable description of the shape."""
        dims = ", ".join(self.dim_names)
        return f"{self.ndim}D array ({dims})"


class BaseEnvRunnersResultsDict(TypedDict):
    """Base type for environment runner results shared between algorithm return and metrics.

    Contains only the core metrics that are consistently Required in both raw algorithm
    results and processed logging metrics. Derived types add their specific Required vs
    NotRequired fields as appropriate.

    This eliminates duplication between EnvRunnersResultsDict and
    _LogMetricsEnvRunnersResultsDict while maintaining their specific behaviors.
    """

    episode_return_mean: float
    """Always required - primary performance metric"""


class CommonVideoTypes:
    """Common video type definitions used across the type hierarchy.

    Provides video-related type aliases that can be specialized
    by different modules for their specific use cases.

    See Also:
        :class:`VideoArrayShape`: Enum for video dimensions
    """

    # Basic video types for algorithm returns
    BasicVideoList: TypeAlias = "list[NDArray]"
    """Simple list of video arrays for algorithm return data"""

    # Advanced video types for logging metrics using enum for dimensions
    Shape4D = tuple[int, int, int, int]  # (B, C, H, W)
    Array4D: TypeAlias = "NDArray[Shape4D]"  # shape=(B, C, H, W)
    Shape5D = tuple[int, int, int, int, int]  # (N, T, C, H, W)
    Array5D: TypeAlias = "NDArray[Shape5D]"  # shape=(N, T, C, H, W)

    LogVideoTypes: TypeAlias = "list[Array4D | Array5D] | Array5D | str | Video"
    """Advanced video types for metrics logging including Video objects and file paths"""


class BaseEvaluationResultsDict(TypedDict, total=False):
    """Base evaluation results structure shared between algorithm return and metrics.

    Defines the core evaluation structure that both raw algorithm results
    and processed metrics use, with specialized implementations extending
    this base as needed.
    """

    evaluated_this_step: NotRequired[bool]
    """Whether evaluation was performed in this training step"""
