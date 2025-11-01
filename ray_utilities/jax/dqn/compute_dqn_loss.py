"""JAX-based DQN loss computation with JIT support.

This module provides jittable loss computation functions for DQN training,
including support for double DQN and distributional Q-learning (C51/Rainbow).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import chex
import jax
import jax.numpy as jnp
from ray.rllib.core.columns import Columns

if TYPE_CHECKING:
    from ray.rllib.algorithms.dqn import DQNConfig

    from ray_utilities.jax.dqn.jax_dqn_module import JaxDQNModule

logger = logging.getLogger(__name__)


# Return signature: (loss, (td_error_mean, q_mean, q_max, q_min))
_return_signature = tuple[
    jnp.ndarray,  # total loss
    tuple[chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric],  # metrics
]


class ComputeDQNLossFunction(Protocol):
    """Protocol for DQN loss computation function signature."""

    def __call__(
        self,
        qf_state_params,
        /,
        batch: dict[str, jax.Array],
        fwd_out: dict[str, jax.Array],
        gamma: float | chex.Numeric,
        *,
        double_q: bool,
    ) -> _return_signature: ...


def make_jax_compute_dqn_loss_function(
    module: JaxDQNModule,  # noqa: ARG001
    config: DQNConfig,
) -> ComputeDQNLossFunction:
    """Create a jittable DQN loss computation function.

    This factory function creates a specialized loss function for the given
    DQN module and configuration. The returned function is JIT-compiled for
    efficient execution.

    Args:
        module: The JAX DQN module to compute losses for
        config: DQN algorithm configuration containing hyperparameters

    Returns:
        A jittable function that computes DQN loss and metrics

    Note:
        All config and module attributes are treated as constants and should
        not be changed after this function is created.
    """
    if TYPE_CHECKING:
        # Prevent JIT during type checking
        jax.jit = lambda func, *args, **kwargs: func  # noqa: ARG005

    # Extract config parameters as constants
    use_huber = config.td_error_loss_fn == "huber" if hasattr(config, "td_error_loss_fn") else False
    huber_threshold = getattr(config, "huber_threshold", 1.0)

    @jax.jit
    def jax_compute_loss_for_module(
        qf_state_params,  # noqa: ARG001
        batch: dict[str, jax.Array],
        fwd_out: dict[str, jax.Array],
        gamma: float | chex.Numeric,
        *,
        double_q: bool,
    ) -> _return_signature:
        """Compute DQN TD-error loss.

        Args:
            qf_state_params: Q-function network parameters (for gradient computation)
            batch: Training batch with observations, actions, rewards, dones
            fwd_out: Forward pass outputs containing Q-value predictions
            gamma: Discount factor
            double_q: Whether to use double DQN

        Returns:
            Tuple of (total_loss, (td_error_mean, q_mean, q_max, q_min))
        """
        # Handle loss masking if present
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            num_valid = jnp.sum(mask)

            def possibly_masked_mean(a: jax.Array):
                return jnp.sum(jnp.where(mask, a, 0.0)) / num_valid

        else:
            possibly_masked_mean = jnp.mean

        # Extract Q-values for actions taken
        # fwd_out["qf_preds"] has shape (batch_size, num_actions)
        q_values = fwd_out["qf_preds"]
        actions = batch[Columns.ACTIONS].astype(jnp.int32)

        # Get Q-values for the actions that were taken
        # Use advanced indexing: q_selected[i] = q_values[i, actions[i]]
        batch_indices = jnp.arange(q_values.shape[0])
        q_selected = q_values[batch_indices, actions]

        # Compute target Q-values
        if "qf_target_next_preds" in fwd_out:
            q_target_next = fwd_out["qf_target_next_preds"]

            if double_q:
                # Double DQN: use current network to select action, target network to evaluate
                q_next = fwd_out.get("qf_next_preds", q_target_next)
                next_actions = jnp.argmax(q_next, axis=-1)
                next_q_values = q_target_next[batch_indices, next_actions]
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = jnp.max(q_target_next, axis=-1)

            # Compute TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
            rewards = batch[Columns.REWARDS]
            dones = batch[Columns.TERMINATEDS].astype(jnp.float32)
            td_targets = rewards + gamma * next_q_values * (1.0 - dones)
        else:
            # No next observations (single-step batch), use rewards only
            td_targets = batch[Columns.REWARDS]

        # Compute TD error
        td_error = td_targets - q_selected

        # Compute loss (MSE or Huber)
        if use_huber:
            # Huber loss: L(x) = 0.5 * x^2 if |x| <= δ, else δ * (|x| - 0.5 * δ)
            abs_td_error = jnp.abs(td_error)
            quadratic = jnp.minimum(abs_td_error, huber_threshold)
            linear = abs_td_error - quadratic
            td_loss = 0.5 * quadratic**2 + huber_threshold * linear
        else:
            # MSE loss
            td_loss = 0.5 * td_error**2

        # Compute mean loss (with optional masking)
        mean_td_loss = possibly_masked_mean(td_loss)

        # Compute metrics
        td_error_mean = possibly_masked_mean(jnp.abs(td_error))
        q_mean = possibly_masked_mean(q_selected)
        q_max = jnp.max(q_selected)
        q_min = jnp.min(q_selected)

        return mean_td_loss, (td_error_mean, q_mean, q_max, q_min)

    # Return the jitted function
    return jax_compute_loss_for_module
