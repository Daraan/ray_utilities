import importlib.metadata
from typing import Any, Literal

import typing_extensions
from packaging.version import Version

_PEP_728_AVAILABLE = getattr(typing_extensions, "_PEP_728_IMPLEMENTED", False) or Version(
    importlib.metadata.version("typing-extensions")
) >= Version("4.13")

ExtraItems = Any  # float | int | str | bool | None | dict[str, "_ExtraItems"] | NDArray[Any] | Never
"""ExtraItems for TypedDict"""

# Below requires _PEP_728_AVAILABLE
# ruff: noqa: E402
from .algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData
from .metrics import LogMetricsDict, FlatLogMetricsDict

__all__ = [
    "AlgorithmReturnData",
    "CometStripedVideoFilename",
    "FlatLogMetricsDict",
    "LogMetricsDict",
    "StrictAlgorithmReturnData",
]


CometStripedVideoFilename = Literal[
    "evaluation_best_video",
    "evaluation_discrete_best_video",
    "evaluation_worst_video",
    "evaluation_discrete_worst_video",
]
