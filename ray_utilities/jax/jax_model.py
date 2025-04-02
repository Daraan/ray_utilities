from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Mapping, Protocol, runtime_checkable

import jax
from ray.rllib.core.models.base import Model
from typing_extensions import TypeVar

if TYPE_CHECKING:
    # TODO: move to submodule
    from sympol import Indices
    import chex
    import flax.linen as nn
    from flax.training.train_state import TrainState
    from ray.rllib.utils.typing import TensorType
    from flax.typing import FrozenVariableDict

    from config_types.params_types import GeneralParams

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="GeneralParams", default="GeneralParams")
ModelType = TypeVar("ModelType", bound="nn.Module", default="nn.Module")


class BaseModel(Model):
    def __call__(self, *args, **kwargs) -> TensorType:
        return self._forward(*args, **kwargs)

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


class FlaxRLModel(Generic[ModelType, ConfigType], BaseModel):
    def __call__(self, *args, **kwargs) -> chex.Array:
        return self._forward(*args, **kwargs)

    @abstractmethod
    def _setup_model(self, *args, **kwargs) -> ModelType:
        """Set up the underlying flax model."""
        ...

    def __init__(self, config: ConfigType, **kwargs):
        self.config: ConfigType = config
        super().__init__(config=config)  # pyright: ignore[reportArgumentType]  # ModelConfig
        self.model: ModelType = self._setup_model(**kwargs)

    def _forward(self, input_dict: dict, **kwargs) -> TensorType:
        out = self.model.apply(input_dict["state"].params, input_dict["obs"], **kwargs)
        return out

    @abstractmethod
    def init_state(self, *args, **kwargs) -> TrainState: ...


@runtime_checkable
class PureJaxModelProtocol(Protocol):
    # TODO: maybe generalize args
    def apply(
        self,
        params: FrozenVariableDict,
        inputs: chex.Array,
        indices: FrozenVariableDict | dict | Mapping,
        **kwargs: Any,
    ) -> jax.Array | chex.Array:
        """Applies the model to the input data."""
        ...

    def init(self, random_key: chex.Array, *args, **kwargs) -> dict[str, chex.Array]:
        """Initializes the model with random keys and arguments."""
        ...

    def init_indices(self, random_key: chex.Array, *args, **kwargs) -> dict[str, chex.Array] | Indices: ...


class JaxRLModel(BaseModel):
    if TYPE_CHECKING:

        def __init__(self, *, config, **kwargs):
            self.model: PureJaxModelProtocol
            super().__init__(config=config, **kwargs)

    @abstractmethod
    def init_state(self, rng: chex.PRNGKey, sample: TensorType | chex.Array) -> TrainState:
        pass

    def _forward(self, input_dict: dict, **kwargs) -> dict | TensorType:
        return self.model.apply(
            input_dict["state"].params,
            inputs=input_dict["obs"],
            indices=input_dict["state"].indices,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        # This is a dummy method to do checked forward passes.
        return self._forward(*args, **kwargs)
