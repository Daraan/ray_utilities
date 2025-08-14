from __future__ import annotations

import abc
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Mapping, Protocol, overload, runtime_checkable

import flax.linen as nn
import jax
from flax.typing import FrozenVariableDict
from ray.rllib.core.models.base import Model
from typing_extensions import TypeVar

if TYPE_CHECKING:
    # TODO: move to submodule
    import chex
    import jax.numpy as jnp
    from flax.core.scope import CollectionFilter
    from flax.training.train_state import TrainState
    from flax.typing import FrozenVariableDict, PRNGKey, RNGSequences, VariableDict
    from ray.rllib.utils.typing import TensorType

    from config_types.params_types import GeneralParams
    from ray_utilities.typing.model_return import Batch
    from sympol import Indices

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="GeneralParams", default="GeneralParams")
ModelType = TypeVar("ModelType", bound="nn.Module", default="nn.Module | FlaxTypedModule")


class BaseModel(Model):
    def __call__(self, input_dict: dict[str, Any], *args, **kwargs) -> TensorType:
        return self._forward(input_dict, *args, **kwargs)  # type: ignore # wrong in rllib

    @abc.abstractmethod
    def _forward(self, input_dict: dict, *, parameters, **kwargs) -> jax.Array: ...  # pyright: ignore[reportIncompatibleMethodOverride]

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
    def __call__(self, input_dict: Batch, *, parameters, **kwargs) -> jax.Array:
        return self._forward(input_dict, parameters=parameters, **kwargs)

    @abstractmethod
    def _setup_model(self, *args, **kwargs) -> ModelType:
        """Set up the underlying flax model."""
        ...

    def __init__(self, config: ConfigType, **kwargs):
        self.config: ConfigType = config
        super().__init__(config=config)  # pyright: ignore[reportArgumentType]  # ModelConfig
        self.model: ModelType = self._setup_model(**kwargs)

    def _forward(self, input_dict: Batch, *, parameters, **kwargs) -> jax.Array:
        # NOTE: Ray's return type-hint is a dict, however this is often not true and rather an array.
        out = self.model.apply(parameters, input_dict["obs"], **kwargs)
        if kwargs.get("mutable"):
            try:
                out, _aux = out
            except ValueError:
                pass
        # Returns a single output if mutable=False (default), otherwise a tuple
        return out  # type: ignore

    @abstractmethod
    def init_state(self, *args, **kwargs) -> TrainState: ...


@runtime_checkable
class PureJaxModelProtocol(Protocol):
    # TODO: maybe generalize args
    def apply(
        self,
        params: FrozenVariableDict | Mapping,
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
    def init_state(self, rng: chex.PRNGKey, sample: TensorType | chex.Array) -> TrainState: ...

    def _forward(
        self,
        input_dict: Batch[jnp.ndarray],
        *,
        parameters: FrozenVariableDict | Mapping,
        indices: FrozenVariableDict | dict | Mapping,
        **kwargs,  # noqa: ARG002
    ) -> jax.Array:
        # kwargs might contain t: #timesteps when exploring
        # NOTE: current pyright error is likely a bug
        return self.model.apply(params=parameters, inputs=input_dict["obs"], indices=indices)  # pyright: ignore[reportReturnType]

    def __call__(
        self,
        input_dict: Batch[jax.Array],
        *,
        parameters: FrozenVariableDict | Mapping,
        indices: FrozenVariableDict | dict | Mapping,
        **kwargs,
    ) -> jax.Array:
        # This is a dummy method to do checked forward passes.
        return self._forward(input_dict, parameters=parameters, indices=indices, **kwargs)


if TYPE_CHECKING:

    class FlaxTypedModule(nn.Module):
        # Module with typed apply method.

        if TYPE_CHECKING:

            @overload
            def apply(
                self,
                variables: VariableDict,
                *args,
                rngs: PRNGKey | RNGSequences | None = None,
                method: Callable[..., Any] | str | None = None,
                mutable: Literal[False] = False,
                capture_intermediates: bool | Callable[["nn.Module", str], bool] = False,
                **kwargs,
            ) -> jax.Array:
                """Applies the model to the input data."""
                ...

            @overload
            def apply(
                self,
                variables: VariableDict,
                *args,
                rngs: PRNGKey | RNGSequences | None = None,
                method: Callable[..., Any] | str | None = None,
                mutable: CollectionFilter,
                capture_intermediates: bool | Callable[["nn.Module", str], bool] = False,
                **kwargs,
            ) -> tuple[jax.Array, FrozenVariableDict | dict[str, Any]]:
                """Applies the model to the input data."""
                ...

            def apply(
                self,
                variables: VariableDict,
                *args,
                rngs: PRNGKey | RNGSequences | None = None,
                method: Callable[..., Any] | str | None = None,
                mutable: CollectionFilter = False,
                capture_intermediates: bool | Callable[["nn.Module", str], bool] = False,
                **kwargs,
            ) -> jax.Array | tuple[jax.Array, FrozenVariableDict | dict[str, Any]]:
                """Applies the model to the input data."""
                ...
else:
    FlaxTypedModule = nn.Module
