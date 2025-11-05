"""Ray Tune extensions and utilities for hyperparameter optimization.

Provides custom schedulers, stoppers, and utilities that extend Ray Tune's
capabilities for hyperparameter optimization of reinforcement learning experiments.

Key Components:
    - Custom schedulers for adaptive training schedule management
    - Specialized stoppers for RL-specific stopping criteria
    - Integration utilities for Ray Tune and RLlib workflows
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def update_hyperparameters(
    param_space: dict[str, Any],
    hyperparameters: dict[str, Any],
    tune_parameters: Sequence[str],
    *,
    num_grid_samples: Optional[int] = None,
    train_batch_size_per_learner: Optional[int] = None,
):
    """Process and validate hyperparameter configurations.

    When using grid_search and the sample of grid search values is less than
    num_grid_samples, the grid search values are repeated to match num_grid_samples.
    This is required for the OptunaSearch which samples a grid value only once, but may repeat
    the value of non-finished runs randomly.

    Args:
        hyperparameters (dict): A dictionary of hyperparameter configurations.
    """
    hyperparameters = deepcopy(hyperparameters)
    if "batch_size" in tune_parameters:  # convenience key
        hyperparameters["train_batch_size_per_learner"] = hyperparameters.pop("batch_size")
        param_space.pop("batch_size", None)
    # Check grid search length and fix minibatch_size
    if (
        len(hyperparameters) == 1
        and isinstance(param := next(iter(hyperparameters.values())), dict)
        and "grid_search" in param
    ):
        if num_grid_samples is None:
            raise ValueError(
                "num_grid_samples must be provided when only tuning a single hyperparameter with grid search."
            )
        # If only tuning batch size with cyclic mutation, also tune minibatch size accordingly
        if "minibatch_size" in hyperparameters:
            # Limit grid to be <= train_batch_size_per_learner
            # TODO: But what is batch size is also being tuned?
            if train_batch_size_per_learner is None:
                raise ValueError("train_batch_size_per_learner must be provided when tuning minibatch_size.")
            param["grid_search"] = [v for v in param["grid_search"] if v <= train_batch_size_per_learner]
        if len(param["grid_search"]) < num_grid_samples:
            # enlarge cyclic grid search values, Optuna shuffles
            param["grid_search"] = (list(param["grid_search"]) * ((num_grid_samples // len(param["grid_search"])) + 1))[
                :num_grid_samples
            ]
    if "batch_size" in hyperparameters:  # convenience key
        hyperparameters["train_batch_size_per_learner"] = hyperparameters.pop("batch_size")
    for matching_keys in set(tune_parameters) & param_space.keys() & hyperparameters.keys():
        logger.warning(
            "Overwriting parameter '%s' in param_space with value from hyperparameters: %s -> %s",
            matching_keys,
            param_space[matching_keys],
            hyperparameters[matching_keys],
        )
    param_space.update({k: v for k, v in hyperparameters.items() if k in tune_parameters})
    return hyperparameters
