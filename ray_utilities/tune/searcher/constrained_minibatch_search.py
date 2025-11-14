from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.sample import Categorical, Float, Integer

if TYPE_CHECKING:
    from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase
    from ray_utilities.typing import ParameterSpace

logger = logging.getLogger(__name__)


def constrained_minibatch_search(
    setup: ExperimentSetupBase[DefaultArgumentParser, Any, Any],
) -> None | BasicVariantGenerator:
    """
    Constrains the minibatch search space based on the train batch size.

    This function modifies the parameter space for "minibatch_size" in the given experiment setup,
    ensuring that minibatch sizes do not exceed the corresponding train_batch_size_per_learner.
    It supports both dictionary-based grid searches and Ray Tune domain objects (Categorical, Integer, Float).
    If both "minibatch_size" and "train_batch_size_per_learner" are grid searches, it generates all valid
    combinations where the minibatch size does not exceed the train_batch_size_per_learner.

    Args:
        setup (ExperimentSetupBase[DefaultArgumentParser, Any, Any]):
            The experiment setup containing parameter spaces and arguments.

    Returns:
        None | BasicVariantGenerator:
            Returns a BasicVariantGenerator with constrained points to evaluate if possible,
            otherwise returns None if constraints cannot be applied or parameter types are unsupported.

    Notes:
        - Modifies the parameter space in-place.
        - Logs warnings if parameter spaces are not as expected or constraints cannot be applied.
        - If "train_batch_size_per_learner" is not in the parameter space, uses the value from setup.args.
    """
    # Check if minibatch_size is in the parameter space
    if "minibatch_size" not in setup.param_space:
        logger.debug("minibatch_size not in param_space, no constraints to apply")
        return None
    # limit search space
    minibatch_size_param: ParameterSpace = setup.param_space["minibatch_size"]
    train_batch_size_param: ParameterSpace | None = setup.param_space.get("train_batch_size_per_learner", None)
    if train_batch_size_param is None:
        batch_size = setup.args.train_batch_size_per_learner
        if isinstance(minibatch_size_param, dict):
            if "grid_search" not in minibatch_size_param:
                logger.warning(
                    "minibatch_size param space is a dict but does not contain grid_search: %s",
                    minibatch_size_param,
                )
                return None
            minibatch_size_param["grid_search"] = [v for v in minibatch_size_param["grid_search"] if v <= batch_size]
            return BasicVariantGenerator(
                points_to_evaluate=[
                    {"minibatch_size": v}
                    for _ in range(setup.args.num_samples)
                    for v in minibatch_size_param["grid_search"]
                ],
                max_concurrent=0 if setup.args.not_parallel else setup.args.num_jobs,
            )
        # Otherwise it is a Domain
        if isinstance(minibatch_size_param, Categorical):
            # limit_categories
            logger.info("Limiting minibatch_size categories to be <= train_batch_size_per_learner %s", batch_size)
            minibatch_size_param.categories = [v for v in minibatch_size_param.categories if v <= batch_size]
        elif isinstance(minibatch_size_param, (Integer, Float)):
            logger.info("Limiting minibatch_size upper bound to be <= train_batch_size_per_learner %s", batch_size)
            minibatch_size_param.upper = min(minibatch_size_param.upper, batch_size)
        # else not sure what to do
        return None
    # train_batch_size_per_learner is also in the sample space
    if isinstance(train_batch_size_param, dict):
        if "grid_search" not in train_batch_size_param:
            logger.warning(
                "train_batch_size_per_learner param space is a dict "
                "but does not contain grid_search cannot limit minibatch_size: %s",
                train_batch_size_param,
            )
            return None
        valid_batch_sizes = set(train_batch_size_param["grid_search"])
        max_batch_size = max(valid_batch_sizes)
    elif isinstance(train_batch_size_param, Categorical):
        valid_batch_sizes = set(train_batch_size_param.categories)
        max_batch_size = max(valid_batch_sizes)
    elif isinstance(train_batch_size_param, (Integer, Float)):
        max_batch_size = train_batch_size_param.upper
    else:
        logger.warning(
            "train_batch_size_per_learner param space is of unknown type %s cannot limit minibatch_size: %s",
            type(train_batch_size_param),
            train_batch_size_param,
        )
        max_batch_size = float("inf")
    if isinstance(minibatch_size_param, dict):
        if "grid_search" not in minibatch_size_param:
            logger.warning(
                "minibatch_size param space is a dict but does not contain grid_search: %s",
                minibatch_size_param,
            )
            return None
        minibatch_size_param["grid_search"] = [v for v in minibatch_size_param["grid_search"] if v <= max_batch_size]
    # Both grid search
    if isinstance(train_batch_size_param, dict) and isinstance(minibatch_size_param, dict):
        # Build grid search combinations
        return BasicVariantGenerator(
            points_to_evaluate=[
                {"minibatch_size": v, "train_batch_size_per_learner": bs}
                for _ in range(setup.args.num_samples)
                for bs in train_batch_size_param["grid_search"]
                for v in minibatch_size_param["grid_search"]
                if v <= bs
            ],
            max_concurrent=0 if setup.args.not_parallel else setup.args.num_jobs,
        )
    # Otherwise it is a Domain
    if isinstance(minibatch_size_param, (Integer, Float)):
        logger.info("Limiting minibatch_size categories to be <= train_batch_size_per_learner %s", max_batch_size)
        minibatch_size_param.upper = min(minibatch_size_param.upper, max_batch_size)
    elif isinstance(minibatch_size_param, Categorical):
        logger.info("Limiting minibatch_size categories to be <= train_batch_size_per_learner %s", max_batch_size)
        minibatch_size_param.categories = [v for v in minibatch_size_param.categories if v <= max_batch_size]
    # else not sure what do do
    return None
