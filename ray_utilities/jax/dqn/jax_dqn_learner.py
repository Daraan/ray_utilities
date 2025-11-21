"""JAX-based learner for DQN with JIT support.

This module provides the JaxDQNLearner class for training DQN agents using JAX.
It includes support for target network updates, double DQN, and distributional Q-learning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from ray.rllib.algorithms.dqn.dqn_learner import DQNLearner as RayDQNLearner
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY
from ray.rllib.core.learner.utils import update_target_network
from ray.rllib.core.rl_module.apis import TargetNetworkAPI
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_TARGET_UPDATES,
)

from ray_utilities.jax.dqn.compute_dqn_loss import make_jax_compute_dqn_loss_function
from ray_utilities.jax.jax_learner import JaxLearner

if TYPE_CHECKING:
    from collections.abc import Mapping

    import chex
    from ray.rllib.algorithms.dqn.dqn import DQNConfig
    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.rllib.utils.typing import ModuleID, TensorType

    from ray_utilities.jax.dqn.jax_dqn_module import JaxDQNModule, JaxDQNStateDict
    from ray_utilities.typing.jax import type_grad_and_value

logger = logging.getLogger(__name__)


# Metric keys for DQN
QF_LOSS_KEY = "qf_loss"
QF_MEAN_KEY = "qf_mean"
QF_MAX_KEY = "qf_max"
QF_MIN_KEY = "qf_min"
TD_ERROR_MEAN_KEY = "td_error_mean"


class JaxDQNLearner(RayDQNLearner, JaxLearner):
    """JAX-based learner for DQN algorithm.

    This learner implements DQN training using JAX for efficient JIT-compiled
    computation. It supports:
    - Target network updates (soft or hard)
    - Double DQN
    - Distributional Q-learning (C51/Rainbow)
    - Gradient accumulation
    - JIT-compiled forward and backward passes

    The learner manages Q-function states and target network states, performing
    periodic target network updates based on the configured update frequency.

    Attributes:
        _states: Mapping from module IDs to JaxDQNStateDict containing Q-function states
        _compute_loss_for_modules: Mapping from module IDs to jitted loss functions
        _forward_with_grads: JIT-compiled forward pass with gradient computation
        _update_jax: JIT-compiled parameter update function
    """

    def build(self, **kwargs) -> None:
        """Build the learner and initialize JIT-compiled functions.

        This method:
        1. Calls parent build() to set up modules and connectors
        2. Initializes JAX PRNG key from config
        3. Creates JIT-compiled loss functions for each module
        4. Sets up JIT-compiled gradient and update functions

        Args:
            **kwargs: Additional keyword arguments passed to parent build()
        """
        super().build(**kwargs)

        # Initialize JAX-specific state
        self._states: Mapping[ModuleID, JaxDQNStateDict]
        self._rng_key = self.config.learner_config_dict.get("rng_key", jax.random.PRNGKey(0))

        # Create JIT-compiled loss functions for each module
        self._compute_loss_for_modules = {
            module_id: make_jax_compute_dqn_loss_function(
                module,  # type: ignore[arg-type]
                self.config,  # type: ignore[arg-type]
            )
            for module_id, module in self.module.items()
        }

        # Create JIT-compiled forward pass with gradients
        if TYPE_CHECKING:
            # Prevent JIT during type checking
            self._forward_with_grads = type_grad_and_value(self._jax_forward_pass)
        else:
            self._forward_with_grads = jax.jit(jax.value_and_grad(self._jax_forward_pass, has_aux=True, argnums=(0,)))

        # JIT-compile the update function
        self._update_jax = jax.jit(self._update_jax)

    @staticmethod
    def _get_state_parameters(
        states: Mapping[ModuleID, JaxDQNStateDict],
    ) -> dict[ModuleID, dict[Literal["qf", "qf_target"], Any]]:
        """Extract parameters from Q-function states.

        Args:
            states: Mapping from module IDs to DQN state dicts

        Returns:
            Dictionary mapping module IDs to dictionaries with "qf" and "qf_target" parameters
        """
        parameters: dict[ModuleID, dict[Literal["qf", "qf_target"], Any]] = dict.fromkeys(
            states.keys(), cast("dict", None)
        )
        for module_id, state in states.items():
            parameters[module_id] = {
                "qf": state["qf"].params if isinstance(state["qf"], TrainState) else state["qf"],
                "qf_target": (
                    state["qf_target"].params if isinstance(state["qf_target"], TrainState) else state["qf_target"]
                ),
            }
        return parameters

    def compute_loss_for_module(  # type: ignore[override]
        self,
        *,
        qf_state_params: Optional[Mapping[str, Any]],
        module_id: ModuleID,
        config: DQNConfig,  # noqa: ARG002
        batch: SampleBatch | dict[str, Any],
        fwd_out: dict[str, TensorType],
    ) -> tuple[TensorType, dict[str, chex.Numeric]]:
        """Compute DQN loss for a single module.

        This method is called during training to compute the TD-error loss
        for the given module and batch.

        Args:
            qf_state_params: Q-function parameters (for gradient computation)
            module_id: ID of the module to compute loss for
            config: DQN configuration (unused, kept for signature compatibility)
            batch: Training batch with observations, actions, rewards, dones
            fwd_out: Forward pass outputs containing Q-value predictions

        Returns:
            Tuple of (total_loss, metrics_dict) where metrics_dict contains:
                - qf_loss: Q-function loss value
                - td_error_mean: Mean absolute TD error
                - qf_mean: Mean Q-value
                - qf_max: Maximum Q-value
                - qf_min: Minimum Q-value
        """
        # Get parameters if not provided
        if qf_state_params is None:
            qf_state_params = self._states[module_id]["qf"].params  # type: ignore[union-attr]

        # Get gamma and double_q from config
        module_config = self.config.get_config_for_module(module_id)
        gamma = getattr(module_config, "gamma", 0.99)
        double_q = getattr(module_config, "double_q", True)

        # Compute loss using JIT-compiled function
        total_loss, (td_error_mean, q_mean, q_max, q_min) = self._compute_loss_for_modules[module_id](
            qf_state_params,
            batch=batch,  # type: ignore[arg-type]
            fwd_out=fwd_out,  # type: ignore[arg-type]
            gamma=float(gamma),
            double_q=bool(double_q),
        )

        # Return total loss and metrics
        return total_loss, {
            POLICY_LOSS_KEY: total_loss,  # Use POLICY_LOSS_KEY for consistency with Ray
            QF_LOSS_KEY: total_loss,
            TD_ERROR_MEAN_KEY: td_error_mean,
            QF_MEAN_KEY: q_mean,
            QF_MAX_KEY: q_max,
            QF_MIN_KEY: q_min,
        }

    def _jax_compute_losses(
        self,
        parameters: dict[ModuleID, dict[Literal["qf", "qf_target"], Mapping[str, Any]]],
        fwd_out: dict[str, Any],
        batch: dict[str, Any],
    ):
        """Compute losses for all modules (JIT-compatible).

        Args:
            parameters: Q-function parameters for all modules
            fwd_out: Forward pass outputs for all modules
            batch: Training batch for all modules

        Returns:
            Tuple of (loss_per_module, aux_data) where aux_data contains metrics
        """
        loss_per_module = {}
        aux_data = {}

        for module_id, module_state in parameters.items():
            module_batch = batch[module_id]
            module_fwd_out = fwd_out[module_id]

            loss, aux = self.compute_loss_for_module(
                module_id=module_id,
                config=self.config,  # type: ignore[arg-type]
                batch=dict(module_batch),
                fwd_out=module_fwd_out,
                qf_state_params=module_state["qf"],
            )
            aux_data[module_id] = aux
            loss_per_module[module_id] = loss

        return loss_per_module, aux_data

    def _forward_train_call(
        self,
        batch,
        parameters: dict[ModuleID, dict[Literal["qf", "qf_target"], Mapping[str, Any]]],
        **kwargs,
    ):
        """Execute forward pass for training (JIT-compatible).

        Args:
            batch: Training batch
            parameters: Q-function parameters
            **kwargs: Additional arguments

        Returns:
            Forward pass outputs for all modules
        """
        fwd_out = {
            mid: cast("JaxDQNModule", self.module._rl_modules[mid])._forward_train(
                batch[mid], parameters=parameters[mid]["qf"], **kwargs
            )
            for mid in batch.keys()
            if mid in self.module
        }
        return fwd_out

    def _jax_forward_pass(
        self,
        parameters: dict[ModuleID, dict[Literal["qf", "qf_target"], Mapping[str, Any]]],
        batch: dict[str, Any],
    ) -> tuple[chex.Numeric, tuple[Any, dict[ModuleID, chex.Numeric], dict[str, Any]]]:
        """JIT-compilable forward pass with loss computation.

        Note:
            Do not use directly - use wrapped version: `self._forward_with_grads`

        Args:
            parameters: Q-function parameters for all modules
            batch: Training batch

        Returns:
            Tuple of (total_loss, (fwd_out, loss_per_module, aux_data))
        """
        fwd_out = self._forward_train_call(batch, parameters=parameters)
        loss_per_module, compute_loss_aux = self._jax_compute_losses(parameters, fwd_out, batch)

        # Gradient computation requires a scalar loss
        return jax.tree.reduce(jnp.sum, loss_per_module), (fwd_out, loss_per_module, compute_loss_aux)

    def _update_jax(
        self,
        states: Mapping[ModuleID, JaxDQNStateDict],
        batch: dict[str, Any],
        *,
        accumulate_gradients_every: int,  # noqa: ARG002
    ) -> tuple[Mapping[ModuleID, JaxDQNStateDict], tuple[Any, dict[ModuleID, chex.Numeric], dict[str, Any]]]:
        """JIT-compiled parameter update.

        Args:
            states: Current Q-function states
            batch: Training batch
            accumulate_gradients_every: Gradient accumulation steps (unused in current implementation)

        Returns:
            Tuple of (new_states, (fwd_out, loss_per_module, aux_data))
        """
        parameters = self._get_state_parameters(states)

        # Compute gradients
        (_all_losses_combined, (fwd_out, loss_per_module, compute_loss_aux)), (gradients,) = self._forward_with_grads(
            parameters, batch
        )

        # Apply gradients
        new_states = self.apply_gradients(gradients, states=states)  # type: ignore[call-arg]

        return new_states, (fwd_out, loss_per_module, compute_loss_aux)

    def _update(self, batch: dict[str, Any] | SampleBatch, **kwargs) -> tuple[Any, Any, Any]:  # noqa: ARG002
        """Update parameters for one training step.

        This is the main update method called during training. It:
        1. Converts batch to dictionary format
        2. Calls JIT-compiled update function
        3. Updates internal states
        4. Updates module states

        Args:
            batch: Training batch
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (fwd_out, loss_per_module, aux_data)
        """
        self.metrics.activate_tensor_mode()

        # Convert batch to dict format for JIT
        batch_dict = {mid: dict(v) for mid, v in batch.items()}

        # Perform JIT-compiled update
        new_states, (fwd_out, loss_per_module, compute_loss_aux) = self._update_jax(
            states=self._states,
            batch=batch_dict,
            accumulate_gradients_every=self.config.learner_config_dict.get("accumulate_gradients_every", 1),
        )

        # Update internal states
        self._states = new_states  # type: ignore[assignment]
        self.module.set_state(self._states)  # type: ignore[arg-type]

        return fwd_out, loss_per_module, compute_loss_aux

    def after_gradient_based_update(self, *, timesteps: dict[str, Any]) -> None:
        """Update target Q-networks after gradient-based update.

        This method is called after each gradient update to potentially
        update the target networks based on the configured update frequency.

        Args:
            timesteps: Dictionary with timestep information including
                NUM_ENV_STEPS_SAMPLED_LIFETIME
        """
        # Call parent to handle other updates
        super().after_gradient_based_update(timesteps=timesteps)

        timestep = timesteps.get(NUM_ENV_STEPS_SAMPLED_LIFETIME, 0)

        # Update target networks for each module
        for module_id, module in self.module._rl_modules.items():
            config = self.config.get_config_for_module(module_id)
            last_update_ts_key = (module_id, LAST_TARGET_UPDATE_TS)

            # Get config attributes with defaults
            target_update_freq = getattr(config, "target_network_update_freq", 1000)
            tau = getattr(config, "tau", 1.0)

            # Check if it's time to update target network
            if timestep - self.metrics.peek(last_update_ts_key, default=0) >= target_update_freq and isinstance(
                module.unwrapped(), TargetNetworkAPI
            ):
                # Update target networks
                for main_net, target_net in module.unwrapped().get_target_network_pairs():
                    update_target_network(
                        main_net=main_net,
                        target_net=target_net,
                        tau=tau,
                    )

                # Update metrics
                self.metrics.log_value((module_id, NUM_TARGET_UPDATES), 1, reduce="sum")
                self.metrics.log_value(last_update_ts_key, timestep, window=1)
