"""JAX-based DQN implementation for Ray RLlib.

This module provides JAX implementations of DQN algorithm components including
the neural network module, catalog for model creation, learner for training,
and jittable loss computation.

Components:
    - JaxDQNModule: JAX-based RLModule for DQN with support for dueling architectures
      and distributional Q-learning
    - JaxDQNStateDict: TypedDict for DQN module state (qf, qf_target, module_key)
    - JaxDQNCatalog: Catalog for creating JAX-based DQN models (encoder, AF/VF heads)
    - JaxDQNLearner: JAX-based learner for DQN training with JIT-compiled updates
    - make_jax_compute_dqn_loss_function: Factory for creating jittable DQN loss functions
"""

from __future__ import annotations

from ray_utilities.jax.dqn.compute_dqn_loss import make_jax_compute_dqn_loss_function
from ray_utilities.jax.dqn.jax_dqn_catalog import JaxDQNCatalog
from ray_utilities.jax.dqn.jax_dqn_learner import JaxDQNLearner
from ray_utilities.jax.dqn.jax_dqn_module import JaxDQNModule, JaxDQNStateDict

__all__ = [
    "JaxDQNCatalog",
    "JaxDQNLearner",
    "JaxDQNModule",
    "JaxDQNStateDict",
    "make_jax_compute_dqn_loss_function",
]
