from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import jax.numpy as jnp
from chex import dataclass
from flax.training.train_state import TrainState as FlaxTrainState

if TYPE_CHECKING:
    import chex
    import jax


class ExtendedTrainState(FlaxTrainState):
    grad_accum: jnp.ndarray


@dataclass(kw_only=True, frozen=True, eq=False)
class Indices:
    """Frozen and hashable variant of the indices dict used by SYMPOL to allow using them as static_args"""

    features_by_estimator: jax.Array = field(hash=False, compare=True)
    # do not use repr because long
    path_identifier_list: jax.Array = field(hash=False, compare=True, repr=False)
    internal_node_index_list: jax.Array = field(hash=False, compare=True, repr=False)

    def __hash__(self) -> int:
        if self._hash is None:  # type: ignore
            object.__setattr__(
                self,
                "_hash",
                hash(
                    (
                        tuple(a.item() for entry in self.features_by_estimator for a in entry),
                        tuple(a.item() for entry in self.path_identifier_list for a in entry),
                        tuple(a.item() for entry in self.internal_node_index_list for a in entry),
                    )
                ),
            )
        return self._hash

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", None)
        if TYPE_CHECKING:
            self._hash: int = None  # pyright: ignore
        self.__hash__()

    # backwards compatibility
    def __getitem__(self, key: str) -> chex.Array:
        return getattr(self, key)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Indices):
            return NotImplemented
        out = (
            jnp.array_equal(self.features_by_estimator, other.features_by_estimator).item()
            and jnp.array_equal(self.path_identifier_list, other.path_identifier_list).item()
            and jnp.array_equal(self.internal_node_index_list, other.internal_node_index_list).item()
        )
        return out
