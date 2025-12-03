from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from abc import abstractmethod
from jax import lax
from ray.rllib.core.models.base import ActorCriticEncoder
from ray.rllib.core.rl_module import RLModule
from typing_extensions import NotRequired, TypedDict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import chex
    import jax.numpy as jnp
    from flax.training.train_state import TrainState
    from ray.rllib.utils.typing import StateDict

    from ray_utilities.jax.utils import ExtendedTrainState
    from ray_utilities.typing.model_return import ActorCriticEncoderOutput


class JaxActorCriticEncoder(ActorCriticEncoder):
    def __call__(self, inputs: dict, **kwargs) -> ActorCriticEncoderOutput[jnp.ndarray]:
        return self._forward(inputs, **kwargs)  # pyright: ignore[reportReturnType]  # interface untyped dict


# pyright: enableExperimentalFeatures=true
class JaxStateDict(TypedDict):
    module_key: int | chex.PRNGKey


class JaxActorCriticStateDict(JaxStateDict):
    actor: ExtendedTrainState | TrainState
    critic: TrainState
    module_key: int | chex.PRNGKey


class JaxModuleState(TypedDict):
    jax_state: JaxStateDict | StateDict | JaxActorCriticStateDict
    model_config: NotRequired[StateDict]


class JaxModule(RLModule):
    """
    Attributes:
        states: A dictionary of state variables.

    Methods:
        - get_state
        - set_state
        - _forward_exploration calls self._forward with lax.stop_gradient(batch)
        - _forward_inference calls self._forward with lax.stop_gradient(batch)
    """

    def __init__(self, *args, **kwargs):
        # set before super; RLModule.__init__ will call setup
        self.states: JaxStateDict | StateDict | JaxActorCriticStateDict = {}
        super().__init__(*args, **kwargs)

    def get_state(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *args,  # noqa: ARG002
        inference_only: bool = False,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> JaxModuleState | StateDict:
        # TODO: could return less than the full state if not inference_only; i.e. do not return the critic
        state = super().get_state(*args, **kwargs)
        state["jax_state"] = self.states
        if hasattr(self, "model_config"):
            state["module_config"] = self.model_config
        return state

    def set_state(self, state: JaxModuleState | StateDict) -> None:
        # Note: Not entirely following Rllib interface
        if not state:
            logger.warning("State is empty, not setting state.")
        self.states = state["jax_state"].copy()
        if "model_config" in state:
            self.model_config = state["model_config"]
        # NOTE: possibly need to update models with new model config!

    @abstractmethod
    def update_jax_state(self, **kwargs):
        """Update the complete subkeys of self.states"""
        # Bruteforce method as some kind of fallback
        updated = False
        for key, value in kwargs.items():
            if key in self.states:
                self.states[key] = value
                updated = True
        if kwargs and not updated:
            raise KeyError("Provided arguments match no keys: ", kwargs, self.states.keys())

    def _forward_exploration(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        return self._forward(lax.stop_gradient(batch), **kwargs)

    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        return self._forward(lax.stop_gradient(batch), **kwargs)


if TYPE_CHECKING:
    JaxModule()
