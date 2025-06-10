from __future__ import annotations

from typing import TYPE_CHECKING

from flax.training.train_state import TrainState as FlaxTrainState

if TYPE_CHECKING:
    import jax.numpy as jnp


class ExtendedTrainState(FlaxTrainState):
    grad_accum: jnp.ndarray
