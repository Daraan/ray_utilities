from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from typing_extensions import TypedDict, NotRequired

if TYPE_CHECKING:
    from ray.rllib.utils.typing import TensorType

logger = logging.getLogger(__name__)


class EncoderOut(TypedDict):
    actor: TensorType
    critic: NotRequired[TensorType]


class ActorCriticEncoderOutput(TypedDict):
    encoder_out: EncoderOut


# Test implemented values

_bad_keys = []
if ENCODER_OUT not in ActorCriticEncoderOutput.__required_keys__:
    _bad_keys.append(ENCODER_OUT)
if ACTOR not in EncoderOut.__required_keys__:
    _bad_keys.append(ACTOR)
if CRITIC not in EncoderOut.__optional_keys__:
    _bad_keys.append(CRITIC)
if _bad_keys:
    logger.warning("Keys %s have changed in RLlib; this module needs an update", _bad_keys)
