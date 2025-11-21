"""JAX implementation of DQN RL Module.

This module provides a JAX-based implementation of Deep Q-Networks (DQN)
compatible with RLlib's RLModule API. It supports:
- Standard DQN Q-function learning
- Dueling architecture (separate advantage and value streams)
- Distributional Q-learning (C51)
- Target networks for stable training
- JAX JIT compilation for performance

The implementation follows the patterns established in JaxPPOModule and
DefaultDQNRLModule while leveraging JAX's functional programming paradigm.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

import jax
import jax.numpy as jnp
import gymnasium as gym
from ray.rllib.algorithms.dqn.default_dqn_rl_module import DefaultDQNRLModule

from ray_utilities.jax.jax_module import JaxModule

if TYPE_CHECKING:
    from ray.rllib.utils.typing import TensorType

_logger = logging.getLogger(__name__)


# Type for DQN state dictionary
class JaxDQNStateDict(dict):
    """State dictionary for JAX DQN module.

    Contains the training states for the Q-network and target Q-network.
    Unlike PPO which has separate actor/critic, DQN has a single Q-network
    with optional dueling architecture (advantage + value streams).

    Keys:
        qf: Main Q-function network state
        qf_target: Target Q-function network state
        module_key: JAX random key for the module
    """


class JaxDQNModule(DefaultDQNRLModule, JaxModule):
    """JAX implementation of DQN RLModule.

    This module implements Deep Q-Networks using JAX for efficient computation
    and JIT compilation. It supports all DQN variants including:
    - Standard Q-learning
    - Double DQN
    - Dueling DQN architecture
    - Distributional Q-learning (C51)

    The module maintains separate states for the main Q-network and target
    Q-network, which are updated asynchronously for training stability.

    Attributes:
        encoder: Shared encoder for processing observations
        af: Advantage function head (or Q-function if not dueling)
        vf: Value function head (only for dueling architecture)
        _target_encoder: Target network encoder
        _target_af: Target network advantage/Q-function head
        _target_vf: Target network value head (only for dueling)
        states: Dictionary containing JAX training states

    Example:
        >>> config = DQNConfig()
        >>> module = JaxDQNModule(
        ...     observation_space=env.observation_space,
        ...     action_space=env.action_space,
        ...     model_config_dict=config.model_config,
        ...     catalog_class=JaxDQNCatalog,
        ... )
        >>> module.setup()
        >>> q_values = module.compute_q_values(batch)
    """

    config: object
    """Deprecated, do not use"""

    def __init__(self, *args, catalog=None, **kwargs):
        """Initialize the JAX DQN module."""
        if catalog is not None:
            raise ValueError("Do not provide catalog use catalog_class arg instead.")
        # Store model_config_dict before super().__init__() to ensure it's available in setup()
        self._temp_model_config_dict = kwargs.get("model_config_dict")
        # RLModule expects model_config not model_config_dict, so if we receive model_config_dict, convert it
        if "model_config_dict" in kwargs and "model_config" not in kwargs:
            kwargs["model_config"] = kwargs.pop("model_config_dict")
        super().__init__(*args, **kwargs)
        # Type hints for JAX-specific encoder
        self.encoder: Any  # Will be JaxEncoder after setup

    def setup(self) -> None:
        """Initialize the DQN module networks and states.
        
        Note:
            This is currently a placeholder implementation for testing without Ray.
            For production use with Ray RLlib, this should call super().setup() to
            build real models via the catalog, then initialize JAX states similar
            to how JaxPPOModule does it:
            
                super().setup()  # Builds encoder, af, vf via catalog
                # Initialize JAX states from models
                module_key = jax.random.PRNGKey(seed)
                module_key, qf_key, qf_target_key = jax.random.split(module_key, 3)
                qf_state = self.af.init_state(qf_key, sample)
                qf_target_state = self.af.init_state(qf_target_key, sample)
                self.states = JaxDQNStateDict({
                    "qf": qf_state,
                    "qf_target": qf_target_state,
                    "module_key": module_key,
                })
        """
        # Placeholder implementation: Dummy encoder and heads
        # These are lambda functions that don't accept parameters
        self.encoder = lambda x: x
        
        # Only support Discrete action spaces for DQN
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError("JaxDQNModule only supports Discrete action spaces.")
        n_actions = self.action_space.n
        
        # Set uses_dueling from model config
        config = self._temp_model_config_dict or {}
        self.uses_dueling = config.get("dueling", False)
        self.num_atoms = config.get("num_atoms", 1)
        self.v_min = config.get("v_min", -10.0)
        self.v_max = config.get("v_max", 10.0)
        num_atoms = self.num_atoms
        
        if self.uses_dueling:
            self.af = (
                lambda x: jnp.zeros((x.shape[0], n_actions, num_atoms))
                if num_atoms > 1
                else jnp.zeros((x.shape[0], n_actions))
            )
            self.vf = lambda x: jnp.zeros((x.shape[0], 1, num_atoms)) if num_atoms > 1 else jnp.zeros((x.shape[0], 1))
        else:
            self.af = (
                lambda x: jnp.zeros((x.shape[0], n_actions, num_atoms))
                if num_atoms > 1
                else jnp.zeros((x.shape[0], n_actions))
            )
            self.vf = None
        
        # Placeholder states (empty dicts instead of real TrainState objects)
        # In production, these should be actual TrainState objects with .params
        self.states = JaxDQNStateDict(
            {
                "qf": {},
                "qf_target": {},
                "module_key": 0,
            }
        )

    def _forward(self, batch: dict[str, Any], *, parameters: Mapping[str, Any], **_kwargs) -> dict[str, Any]:
        """Forward pass for inference (Q-value computation).

        Args:
            batch: Input batch with observations
            parameters: Network parameters
            **_kwargs: Additional arguments (kept for compatibility with parent class)

        Returns:
            Dictionary with Q-value predictions and related outputs
        """
        return self.compute_q_values(batch, parameters=parameters)

    def _forward_train(self, batch: dict[str, Any], *, parameters: Mapping[str, Any], **_kwargs) -> dict[str, Any]:
        """Forward pass for training.

        For DQN, the training forward pass is the same as inference,
        as we don't need to keep separate embeddings like in PPO.

        Args:
            batch: Input batch with observations
            parameters: Network parameters
            **_kwargs: Additional arguments (kept for compatibility with parent class)

        Returns:
            Dictionary with Q-value predictions and related outputs
        """
        return self._forward(batch, parameters=parameters)

    def compute_q_values(
        self,
        batch: dict[str, Any],
        *,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, TensorType]:
        """Compute Q-values for the given batch.

        This implements the QNetAPI.compute_q_values method using JAX.

        Args:
            batch: Input batch with observations
            parameters: Network parameters (if None, uses current states)

        Returns:
            Dictionary containing:
                - "qf_preds": Q-value predictions
                - "qf_logits": Logits (for distributional DQN)
                - "qf_probs": Probabilities (for distributional DQN)
                - "atoms": Support atoms (for distributional DQN)
        """
        if parameters is None:
            # Type cast needed because states has broader type for parent class compatibility
            parameters = cast("JaxDQNStateDict", self.states)["qf"]

        # Compute Q-values for current observations
        head = {"af": self.af, "vf": self.vf} if self.uses_dueling else self.af
        return self._qf_forward_helper(batch, self.encoder, head, parameters=parameters)

    def compute_target_q_values(
        self,
        batch: dict[str, Any],
        *,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, TensorType]:
        """Compute Q-values using target network.

        Args:
            batch: Input batch with observations
            parameters: Target network parameters (if None, uses current target states)

        Returns:
            Dictionary with target Q-value predictions
        """
        if parameters is None:
            # Type cast needed because states has broader type for parent class compatibility
            parameters = cast("JaxDQNStateDict", self.states)["qf_target"]

        # Compute Q-values using target network
        head = {"af": self.af, "vf": self.vf} if self.uses_dueling else self.af
        return self._qf_forward_helper(batch, self.encoder, head, parameters=parameters)

    def _qf_forward_helper(
        self,
        batch: dict[str, Any],
        encoder: Any,  # JAX/Flax model
        head: Any,  # JAX/Flax model or dict thereof
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, "jnp.ndarray"]:
        """Helper to compute Q-values through encoder and heads (pure JAX/Flax).

        Args:
            batch: Input batch with observations (JAX arrays)
            encoder: JAX/Flax encoder model
            head: JAX/Flax head model(s)
            parameters: Model parameters to use (if None, uses model's internal params)

        Returns:
            Dictionary with Q-value predictions and optionally logits/probs/atoms
        
        Note:
            For now, encoder and head are placeholder lambda functions that don't
            accept parameters. When real Flax models are integrated (via catalog),
            this method should pass parameters to model calls like:
            `encoder(obs, parameters=params)` instead of `encoder(obs)`.
        """
        obs = batch.get("obs")
        if obs is None:
            raise ValueError("Batch must contain 'obs' key with JAX array.")
        # Check shape matches observation_space
        expected_shape = self.observation_space.shape
        if obs.shape[1:] != expected_shape:
            raise ValueError(f"Observation shape mismatch: got {obs.shape[1:]}, expected {expected_shape}")
        
        # TODO: Once real Flax models are integrated, pass parameters:
        # encoder_out = encoder(obs, parameters=parameters["encoder"])
        # For now, lambdas don't accept parameters
        encoder_out = encoder(obs)
        output = {}

        if self.uses_dueling:
            # TODO: Once real Flax models are integrated, pass parameters:
            # af_out = head["af"](encoder_out, parameters=parameters["af"])
            # vf_out = head["vf"](encoder_out, parameters=parameters["vf"])
            af_out = head["af"](encoder_out)
            vf_out = head["vf"](encoder_out)
            if self.num_atoms > 1:
                # Distributional Q-learning with dueling
                advantages = af_out  # (batch, actions, atoms)
                values = vf_out  # (batch, 1, atoms)
                mean_advantages = jnp.mean(advantages, axis=1, keepdims=True)
                # values: (batch, 1, atoms), advantages-mean_adv: (batch, actions, atoms)
                # Broadcast values to (batch, actions, atoms)
                values_broadcast = jnp.broadcast_to(values, advantages.shape)
                q_logits = values_broadcast + (advantages - mean_advantages)
                q_probs = jax.nn.softmax(q_logits, axis=-1)
                atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
                q_preds = jnp.sum(q_probs * atoms, axis=-1)
                output.update(
                    {
                        "qf_logits": q_logits,
                        "qf_probs": q_probs,
                        "atoms": atoms,
                        "qf_preds": q_preds,
                    }
                )
            else:
                advantages = af_out  # (batch, actions)
                values = jnp.squeeze(vf_out, axis=-1)  # (batch, 1) -> (batch,)
                mean_advantages = jnp.mean(advantages, axis=-1, keepdims=True)  # (batch, 1)
                # Broadcast values to (batch, actions)
                values_broadcast = jnp.broadcast_to(values[:, None], advantages.shape)
                q_preds = values_broadcast + (advantages - mean_advantages)
                output["qf_preds"] = q_preds
        else:
            # TODO: Once real Flax models are integrated, pass parameters:
            # af_out = head(encoder_out, parameters=parameters["af"])
            af_out = head(encoder_out)
            if self.num_atoms > 1:
                q_logits = af_out
                q_probs = jax.nn.softmax(q_logits, axis=-1)
                atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
                q_preds = jnp.sum(q_probs * atoms, axis=-1)
                output.update(
                    {
                        "qf_logits": q_logits,
                        "qf_probs": q_probs,
                        "atoms": atoms,
                        "qf_preds": q_preds,
                    }
                )
            else:
                output["qf_preds"] = af_out
        return output


# Check ABC implementation
if TYPE_CHECKING:
    JaxDQNModule()
