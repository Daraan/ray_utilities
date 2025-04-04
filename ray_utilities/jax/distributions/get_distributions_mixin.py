from __future__ import annotations

import gymnasium as gym
from ray_utilities.jax.distributions.jax_distributions import Categorical, Normal, RLlibToJaxDistribution

__all__ = ["GetJaxDistributionsMixin"]


class GetJaxDistributionsMixin:
    """
    Mixin providing methods for RLModules

    Methods:
        - get_train_action_dist_cls
        - get_exploration_action_dist_cls
        - get_inference_action_dict_cls
    """

    action_dist_cls: type[RLlibToJaxDistribution]
    action_space: gym.Space

    def get_train_action_dist_cls(self) -> type[RLlibToJaxDistribution]:
        return self.get_inference_action_dist_cls()

    def get_exploration_action_dist_cls(self) -> type[RLlibToJaxDistribution]:
        return self.get_inference_action_dist_cls()

    def get_inference_action_dist_cls(self) -> type[RLlibToJaxDistribution]:
        if self.action_dist_cls is not None:  # pyright: ignore[reportUnnecessaryComparison]
            return self.action_dist_cls
        if isinstance(self.action_space, gym.spaces.Discrete):
            return Categorical
        if isinstance(self.action_space, gym.spaces.Box):
            return Normal
        raise ValueError(
            f"Default action distribution for action space "
            f"{self.action_space} not supported! Either set the "
            f"`self.action_dist_cls` property in your RLModule's `setup()` method "
            f"to a subclass of `ray.rllib.models.torch.torch_distributions."
            f"TorchDistribution` or - if you need different distributions for "
            f"inference and training - override the three methods: "
            f"`get_inference_action_dist_cls`, `get_exploration_action_dist_cls`, "
            f"and `get_train_action_dist_cls` in your RLModule."
        )
