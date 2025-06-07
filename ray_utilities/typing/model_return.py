from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generic, Mapping

from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from typing_extensions import NotRequired, TypeAliasType, TypedDict, TypeVar

if TYPE_CHECKING:
    import chex
    from ray.rllib.utils.typing import TensorType

logger = logging.getLogger(__name__)

_TensorT = TypeVar("_TensorT", bound="TensorType | chex.Array", default="TensorType")


# needed for extra_items
# pyright: enableExperimentalFeatures=true
class BatchDict(TypedDict, Generic[_TensorT], extra_items="_TensorT"):
    """Keys correspond to ray.rllib.core.columns.Columns"""

    action_dist_inputs: _TensorT
    action_logp: _TensorT


Batch = TypeAliasType("Batch", BatchDict[_TensorT] | Mapping[str, "_TensorT"], type_params=(_TensorT,))


class EncoderOut(TypedDict, Generic[_TensorT]):
    actor: Batch[_TensorT]
    critic: NotRequired[Batch[_TensorT]]


class ActorCriticEncoderOutput(TypedDict, Generic[_TensorT]):
    encoder_out: EncoderOut[_TensorT]


# Test implemented values

_bad_keys = []
if ENCODER_OUT not in ActorCriticEncoderOutput.__required_keys__:
    _bad_keys.append(ENCODER_OUT)
if ACTOR not in EncoderOut.__required_keys__:
    _bad_keys.append(ACTOR)
if CRITIC not in EncoderOut.__optional_keys__:
    _bad_keys.append(CRITIC)
if _bad_keys:
    logger.error("Keys %s have changed in RLlib; this module %s needs an update", _bad_keys, __name__)
