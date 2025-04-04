"""Utilities for ray and ray tune to be used across projects."""

# ruff: noqa: PLC0415  # imports at top level of file; safe import time if not needed.

from __future__ import annotations

from typing import Any

# fmt: off
try:
    # Import comet early for its monkey patch
    import comet_ml  # noqa: F401
except ImportError:
    pass
# fmt: on

from ray_utilities.default_trainable import create_default_trainable, default_trainable, episode_iterator  # noqa: E402
from ray_utilities.misc import get_trainable_name, is_pbar, trial_name_creator
from ray_utilities.nice_logging import nicer_logging
from ray_utilities.random import seed_everything
from ray_utilities.runfiles.run_tune import run_tune
from ray_utilities.typing.algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData

logger = nicer_logging(__name__, level="DEBUG")
logger.info("Ray utilities imported")
logger.debug("Ray utilities logger debug level set")


__all__ = [
    "AlgorithmReturnData",
    "StrictAlgorithmReturnData",
    "create_default_trainable",
    "default_trainable",
    "episode_iterator",
    "get_trainable_name",
    "is_pbar",
    "run_tune",
    "seed_everything",
    "trial_name_creator",
]

def flat_dict_to_nested(metrics: dict[str, Any]) -> dict[str, Any | dict[str, Any]]:
    nested_metrics = metrics.copy()
    for key_orig, v in metrics.items():
        k = key_orig
        subdict = nested_metrics
        while "/" in k:
            parent, k = k.split("/", 1)
            subdict = subdict.setdefault(parent, {})
        subdict[k] = v
        if key_orig != k:
            del nested_metrics[key_orig]
    return nested_metrics
