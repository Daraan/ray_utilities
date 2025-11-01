"""JAX-based catalog for DQN model creation.

This module provides the JaxDQNCatalog class for creating JAX-compatible
DQN models. The catalog builds Torch encoder and heads (as required by RLlib),
and the JaxDQNModule handles tensor conversion between Torch and JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ray.rllib.algorithms.dqn.dqn_catalog import DQNCatalog
from ray.rllib.utils.annotations import override

from ray_utilities.jax.catalog.jax_catalog import JaxCatalog

if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.core.models.configs import MLPHeadConfig


class JaxDQNCatalog(DQNCatalog, JaxCatalog):
    """JAX-based catalog for DQN Rainbow models.

    JaxDQNCatalog extends DQNCatalog to provide JAX-specific model creation
    for DQN algorithms. It builds JAX/Flax models for:

    - Encoder: JAX-based encoder for observations
    - AF Head: Advantage or Q-function head (action_space.n * num_atoms outputs)
    - VF Head: Value function head for dueling architecture (optional, 1 output)

    The catalog supports:
    - Standard Q-networks
    - Dueling architecture (separate advantage and value streams)
    - Distributional Q-learning (C51/Rainbow with multiple atoms)

    All models are built using the JAX framework and are compatible with
    JAX's JIT compilation for efficient training.

    Attributes:
        num_atoms: Number of atoms for distributional Q-learning (1 for standard DQN)
        af_head_config: Configuration for advantage/Q-function head
        vf_head_config: Configuration for value function head (dueling only)
    """

    @override(DQNCatalog)
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
        view_requirements: dict | None = None,
    ):
        """Initialize the JaxDQNCatalog.

        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            model_config_dict: Model configuration dictionary containing:
                - num_atoms: Number of atoms for distributional Q-learning
                - head_fcnet_hiddens: Hidden layer dimensions for heads
                - head_fcnet_activation: Activation function for heads
                - Various weight/bias initializer configs
            view_requirements: Deprecated, should be None. Use ConnectorV2 API instead.

        Raises:
            AssertionError: If view_requirements is not None.
            TypeError: If action_space is not Discrete.
        """
        if not hasattr(action_space, "n"):
            raise TypeError("JaxDQNCatalog only supports Discrete action spaces.")
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
            view_requirements=view_requirements,  # type: ignore[arg-type]
        )


    @override(DQNCatalog)
    def get_action_dist_cls(self, framework: str | None = None):
        # RLlib expects framework="torch" for DQN, even for JAX modules.
        if framework is None:
            framework = "torch"
        return super().get_action_dist_cls(framework=framework)


    @override(DQNCatalog)
    def _get_head_config(self, output_layer_dim: int) -> MLPHeadConfig:
        """Get configuration for an MLP head.

        This is called internally to create af_head_config and vf_head_config.

        Args:
            output_layer_dim: Output dimension (1 for value, action_space.n * num_atoms
                for advantage/Q-function).

        Returns:
            MLPHeadConfig for the specified output dimension.
        """
        # Call parent implementation which handles all the config details
        return super()._get_head_config(output_layer_dim)
