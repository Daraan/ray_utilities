from __future__ import annotations


import logging
from typing import TYPE_CHECKING, Any, TypeVar

from ray.rllib.core.rl_module import RLModule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ray.rllib.utils.typing import StateDict

    _StateDict = TypeVar("_StateDict", bound=dict[str, Any], covariant=True)


class JaxModule(RLModule):
    """
    Attributes:
        states: A dictionary of state variables.

    Methods:
        - get_state
        - set_state
    """

    def __init__(self, *args, **kwargs):
        self.states: StateDict | Any = {}  # set before super; RLModule.__init__ will call setup
        super().__init__(*args, **kwargs)

    def get_state(self, *args, inference_only: bool = False, **kwargs) -> StateDict:  # noqa: ARG002
        # TODO: could return less than the full state if not inference_only; i.e. do not return the critic
        return self.states

    def set_state(self, state: StateDict) -> None:
        if not state:
            logger.warning("State is empty, not setting state.")
        self.states = state


if TYPE_CHECKING:
    JaxModule()
