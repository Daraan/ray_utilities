"""Ray Utilities: Advanced utilities for Ray Tune and RLlib experiments.

Provides a comprehensive set of utilities, classes, and functions to streamline
Ray Tune hyperparameter optimization and Ray RLlib reinforcement learning experiments.

Main Components:
    - :class:`DefaultTrainable`: Base trainable class with checkpoint/restore functionality
    - :func:`run_tune`: Enhanced Ray Tune experiment runner with advanced logging
    - :func:`nice_logger`: Colored logging setup for better debugging
    - :func:`seed_everything`: Comprehensive seeding for reproducible experiments
    - :data:`AlgorithmReturnData`: Type definitions for algorithm return values

Example:
    >>> import ray_utilities as ru
    >>> logger = ru.nice_logger(__name__)
    >>> ru.seed_everything(env=None, seed=42)
    >>> trainable = ru.create_default_trainable(config_class=PPOConfig)
    >>> ru.run_tune(trainable, param_space=config, num_samples=10)
"""

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

from ray_utilities.misc import get_trainable_name, is_pbar, trial_name_creator
from ray_utilities.nice_logger import nice_logger
from ray_utilities.random import seed_everything
from ray_utilities.runfiles.run_tune import run_tune
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.functional import create_default_trainable, default_trainable
from ray_utilities.training.helpers import episode_iterator
from ray_utilities.typing.algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData

logger = nice_logger(__name__, level="DEBUG")
logger.info("Ray utilities imported")
logger.debug("Ray utilities logger debug level set")


__all__ = [
    "AlgorithmReturnData",
    "DefaultTrainable",
    "StrictAlgorithmReturnData",
    "create_default_trainable",
    "default_trainable",
    "episode_iterator",
    "get_trainable_name",
    "is_pbar",
    "nice_logger",
    "run_tune",
    "seed_everything",
    "trial_name_creator",
]


def flat_dict_to_nested(metrics: dict[str, Any]) -> dict[str, Any | dict[str, Any]]:
    """Convert a flat dictionary with slash-separated keys to a nested dictionary structure.

    This function transforms dictionary keys containing forward slashes into nested
    dictionary structures, useful for organizing Ray Tune/RLlib metrics that are
    typically logged with hierarchical key names.

    Args:
        metrics: A dictionary with potentially slash-separated keys (e.g.,
            ``{"eval/return_mean": 100, "train/loss": 0.5}``).

    Returns:
        A nested dictionary structure where slash-separated keys become nested levels
        (e.g., ``{"eval": {"return_mean": 100}, "train": {"loss": 0.5}}``).

    Example:
        >>> metrics = {
        ...     "train/episode_return_mean": 150.0,
        ...     "eval/env_runner_results/episode_return_mean": 200.0,
        ...     "timesteps_total": 10000,
        ... }
        >>> nested = flat_dict_to_nested(metrics)
        >>> nested["train"]["episode_return_mean"]
        150.0
        >>> nested["eval"]["env_runner_results"]["episode_return_mean"]
        200.0

    Note:
        This is particularly useful when working with Ray Tune's result dictionaries
        which often contain hierarchical metrics with slash-separated key names.
    """
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
