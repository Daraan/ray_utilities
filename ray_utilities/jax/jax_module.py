from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jax import lax
from ray.rllib.core.models.base import ActorCriticEncoder
from ray.rllib.core.rl_module import RLModule
from typing_extensions import TypedDict

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
    ) -> JaxStateDict | StateDict | JaxActorCriticStateDict:
        # TODO: could return less than the full state if not inference_only; i.e. do not return the critic
        return self.states

    def set_state(self, state: JaxStateDict | StateDict | JaxActorCriticStateDict) -> None:
        # Note: Not entirely following Rllib interface
        if not state:
            logger.warning("State is empty, not setting state.")
        self.states = state

    def _forward_exploration(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        return self._forward(lax.stop_gradient(batch), **kwargs)

    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        return self._forward(lax.stop_gradient(batch), **kwargs)


if TYPE_CHECKING:
    JaxModule()
