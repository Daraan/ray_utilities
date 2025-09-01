"""Port of Catalog._get_dist_cls_from_action_space"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional, Sequence

import gymnasium as gym
import numpy as np
import tree
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space, get_base_struct_from_space

from ray_utilities.jax.distributions.jax_distributions import (
    Categorical,
    Deterministic,
    MultiCategorical,
    MultiDistribution,
    Normal,
    RLlibToJaxDistribution,
)

if TYPE_CHECKING:
    from ray.rllib.core.distribution.distribution import Distribution
    from ray.rllib.core.models.catalog import Catalog

__all__ = ["get_jax_dist_cls_from_action_space"]


def get_jax_dist_cls_from_action_space(
    cls: type[Catalog] | Catalog,
    action_space: gym.Space,
    *,
    framework: Optional[str] = None,
) -> type[RLlibToJaxDistribution]:
    """Returns a distribution class for the given action space.

    You can get the required input dimension for the distribution by calling
    `action_dict_cls.required_input_dim(action_space)`
    on the retrieved class. This is useful, because the Catalog needs to find out
    about the required input dimension for the distribution before the model that
    outputs these inputs is configured.

    Args:
        action_space: Action space of the target gym env.
        framework: The framework to use.

    Returns:
        The distribution class for the given action space.
    """
    # If no framework provided, return no action distribution class (None).
    if framework is None:
        return None  # type: ignore[return-value]
    # This method is structured in two steps:
    # Firstly, construct a dictionary containing the available distribution classes.
    # Secondly, return the correct distribution class for the given action space.

    # Step 1: Construct the dictionary.

    class DistEnum(enum.Enum):
        Categorical = "Categorical"
        DiagGaussian = "Gaussian"
        Deterministic = "Deterministic"
        MultiDistribution = "MultiDistribution"
        MultiCategorical = "MultiCategorical"

    distribution_dicts: dict[DistEnum, type[RLlibToJaxDistribution]] = {
        DistEnum.Deterministic: Deterministic,
        DistEnum.DiagGaussian: Normal,  # DiagGaussian
        DistEnum.Categorical: Categorical,
    }

    # Only add a MultiAction distribution class to the dict if we can compute its
    # components (we need a Tuple/Dict space for this).
    if isinstance(action_space, (spaces.Tuple, spaces.Dict)):
        partial_multi_action_distribution_cls = _multi_action_dist_partial_helper(
            catalog_cls=cls,
            action_space=action_space,
            framework=framework,
        )

        distribution_dicts[DistEnum.MultiDistribution] = partial_multi_action_distribution_cls

    # Only add a MultiCategorical distribution class to the dict if we can compute
    # its components (we need a MultiDiscrete space for this).
    if isinstance(action_space, spaces.MultiDiscrete):
        partial_multi_categorical_distribution_cls = _multi_categorical_dist_partial_helper(
            action_space=action_space,
            framework=framework,
        )

        distribution_dicts[DistEnum.MultiCategorical] = partial_multi_categorical_distribution_cls

    # Step 2: Return the correct distribution class for the given action space.

    # Box space -> DiagGaussian OR Deterministic.
    if isinstance(action_space, Box):
        if action_space.dtype.char in np.typecodes["AllInteger"]:
            raise ValueError(
                "Box(..., `int`) action spaces are not supported. Use MultiDiscrete  or Box(..., `float`)."
            )
        if len(action_space.shape) > 1:
            raise UnsupportedSpaceException(
                f"Action space has multiple dimensions {action_space.shape}. "
                f"Consider reshaping this into a single dimension, using a "
                f"custom action distribution, using a Tuple action space, "
                f"or the multi-agent API."
            )
        return distribution_dicts[DistEnum.DiagGaussian]

    # Discrete Space -> Categorical.
    if isinstance(action_space, Discrete):
        return distribution_dicts[DistEnum.Categorical]

    # Tuple/Dict Spaces -> MultiAction.
    if isinstance(action_space, (spaces.Tuple, spaces.Dict)):
        return distribution_dicts[DistEnum.MultiDistribution]

    # Simplex -> Dirichlet.
    if isinstance(action_space, Simplex):
        raise NotImplementedError("Simplex action space not yet supported.")

    # MultiDiscrete -> MultiCategorical.
    if isinstance(action_space, spaces.MultiDiscrete):
        return distribution_dicts[DistEnum.MultiCategorical]

    # Unknown type -> Error.
    raise NotImplementedError(f"Unsupported action space: `{action_space}`")


def _multi_categorical_dist_partial_helper(
    action_space: gym.spaces.MultiDiscrete,
    framework: str,  # noqa: ARG001
) -> type[RLlibToJaxDistribution]:
    """Helper method to get a partial of a MultiCategorical Distribution.

    This is useful for when we want to create MultiCategorical Distribution from
    logits only (!) later, but know the action space now already.

    Args:
        action_space: The action space to get the child distribution classes for.
        framework: The framework to use.

    Returns:
        A partial of the MultiCategorical class.
    """
    partial_dist_cls = MultiCategorical.get_partial_dist_cls(  # pyright: ignore[reportAttributeAccessIssue]
        space=action_space, input_lens=list(action_space.nvec)
    )

    return partial_dist_cls


def _multi_action_dist_partial_helper(
    catalog_cls: type[Catalog] | Catalog, action_space: gym.Space, framework: str
) -> type[RLlibToJaxDistribution]:
    """Helper method to get a partial of a MultiActionDistribution.

    This is useful for when we want to create MultiActionDistributions from
    logits only (!) later, but know the action space now already.

    Args:
        catalog_cls: The ModelCatalog class to use.
        action_space: The action space to get the child distribution classes for.
        framework: The framework to use.

    Returns:
        A partial of the TorchMultiActionDistribution class.
    """
    action_space_struct = get_base_struct_from_space(action_space)
    flat_action_space = flatten_space(action_space)
    child_distribution_cls_struct = tree.map_structure(
        lambda s: catalog_cls._get_dist_cls_from_action_space(
            action_space=s,
            framework=framework,
        ),
        action_space_struct,
    )
    flat_distribution_clses: Sequence[Distribution] = tree.flatten(child_distribution_cls_struct)

    logit_lens = [
        int(dist_cls.required_input_dim(space)) for dist_cls, space in zip(flat_distribution_clses, flat_action_space)
    ]

    partial_dist_cls = MultiDistribution.get_partial_dist_cls(  # assignment to cls fails # pyright: ignore[reportAttributeAccessIssue]  # noqa: E501
        space=action_space,
        child_distribution_cls_struct=child_distribution_cls_struct,
        input_lens=logit_lens,
    )
    return partial_dist_cls
