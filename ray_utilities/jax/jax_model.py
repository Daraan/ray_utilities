"""JAX-based model implementations for Ray RLlib.

This module provides JAX-specific model classes that integrate with the Ray RLlib
framework. It includes:

- Base model classes that extend RLlib's Model interface
- Flax-based neural network model implementations
- Protocol definitions for pure JAX models
- Type-safe interfaces for JAX array operations

The models support both Flax (JAX's neural network library) and pure JAX
implementations, allowing flexibility in how neural networks are defined
and used within the RLlib ecosystem.

Type Variables:
    ConfigType: Configuration type bound to GeneralParams
    ModelType: Model type bound to Flax Module
"""

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

    from ray_utilities.jax.utils import Indices
    from ray_utilities.typing.jax import GeneralParams
    from ray_utilities.typing.model_return import Batch

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="GeneralParams", default="GeneralParams")
ModelType = TypeVar("ModelType", bound="nn.Module", default="nn.Module | FlaxTypedModule")


class BaseModel(Model):
    """Base JAX model class extending Ray RLlib's Model interface.

    This class provides a JAX-specific implementation of the RLlib Model interface,
    serving as a base for all JAX-based neural network models used in reinforcement
    learning algorithms.

    The class implements the standard Model interface but adapts it for JAX operations,
    including proper handling of JAX arrays and functional programming patterns.

    Abstract Methods:
        _forward: Must be implemented by subclasses to define the forward pass
            using JAX operations.

    Warning:
        Some methods like :meth:`get_num_parameters` may not be fully implemented
        and will emit warnings when called.
    """

    def __call__(self, input_dict: dict[str, Any], *args, **kwargs) -> TensorType:
        """Forward pass through the model.

        Args:
            input_dict: Dictionary containing input tensors/arrays.
            *args: Additional positional arguments passed to _forward.
            **kwargs: Additional keyword arguments passed to _forward.

        Returns:
            Output tensor/array from the forward pass.
        """
        return self._forward(input_dict, *args, **kwargs)  # type: ignore # wrong in rllib

    @abc.abstractmethod
    def _forward(self, input_dict: dict, *, parameters, **kwargs) -> jax.Array:
        """Abstract method for implementing the forward pass.

        Args:
            input_dict: Dictionary containing input data.
            parameters: Model parameters (weights and biases).
            **kwargs: Additional keyword arguments.

        Returns:
            JAX array with the model output.
        """
        ...  # pyright: ignore[reportIncompatibleMethodOverride]

    def get_num_parameters(self) -> tuple[int, int]:
        """Get the number of trainable and non-trainable parameters.

        Warning:
            This method may not be fully implemented and could return
            incorrect values. It attempts to count parameters by traversing
            the JAX tree structure.

        Returns:
            Tuple of (trainable_params, non_trainable_params). Currently
            assumes all parameters are trainable, so non-trainable count is 0.

        Raises:
            Exception: If parameter counting fails, returns placeholder values (42, 42).
        """
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
