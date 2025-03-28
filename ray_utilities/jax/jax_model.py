from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import flax.linen as nn
import jax
from ray.rllib.core.models.base import Model

if TYPE_CHECKING:
    import chex
    from flax.core.scope import (
        CollectionFilter,
        DenyList,
        Variable,
        union_filters,
    )
    from flax.training.train_state import TrainState
    from flax.typing import (
        FrozenVariableDict,
        PRNGKey,
        RNGSequences,
        VariableDict,
    )
    from ray.rllib.utils.typing import TensorType

logger = logging.getLogger(__name__)


class BaseModel(Model):

    def get_num_parameters(self) -> tuple[int, int]:
        # Unknown
        logger.warning("Warning num_parameters called which might be wrong")
        try:
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(self))
            return (
                param_count,  # trainable
                param_count - param_count,  # non trainable? 0?
            )
        except Exception:
            logger.exception("Error getting number of parameters")
            return 42, 42

    def _set_to_dummy_weights(self, value_sequence=...) -> None:
        # Unknown
        logger.warning("Requested setting to dummy weights, but not implemented")
        return super()._set_to_dummy_weights(value_sequence)


class FlaxRLModel(BaseModel, nn.Module):
    def _forward(self, input_dict: dict, **kwargs) -> dict | chex.Array:
        breakpoint()
        out = super().__call__(input_dict["obs"], **kwargs)
        return out


class JaxRLModel(BaseModel):
    @abstractmethod
    def init_state(self, rng: chex.PRNGKey, sample: TensorType | chex.Array) -> TrainState:
        pass

    @abstractmethod
    def apply(
        self,
        *args,
        **kwargs,
    ) -> Any | tuple[Any, FrozenVariableDict | dict[str, Any]]:
        pass

    def _forward(self, input_dict: dict, **kwargs) -> dict | TensorType:
        variables, indices = "XXX", "XXX"  # TODO: implement
        return self.apply(inputs=input_dict["obs"], **kwargs)

    def __call__(self, *args, **kwargs):
        breakpoint()
        # This is a dummy method to do checked forward passes.
        return self._forward(*args, **kwargs)


if TYPE_CHECKING:
    ...
