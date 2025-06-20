from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ray.rllib.core.models.catalog import Catalog

from ray_utilities.jax.distributions.get_jax_action_distribution import get_jax_dist_cls_from_action_space

if TYPE_CHECKING:
    import gymnasium as gym

    from ray_utilities.jax.distributions.jax_distributions import RLlibToJaxDistribution


class JaxCatalog(Catalog):
    """Provides _get_dist_cls_from_action_space to return JAX based action distributions."""

    @classmethod
    def _get_dist_cls_from_action_space(  # pyright: ignore[reportIncompatibleMethodOverride] # ray is wrong should return type
        cls,
        action_space: gym.Space,
        *,
        framework: Optional[str] = None,
    ) -> type[RLlibToJaxDistribution]:
        return get_jax_dist_cls_from_action_space(cls, action_space, framework=framework)
