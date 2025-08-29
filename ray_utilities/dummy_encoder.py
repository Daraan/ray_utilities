"""Dummy encoder implementations for Ray RLlib models with minimal overhead.

This module provides lightweight encoder implementations that bypass complex
neural network processing while maintaining compatibility with Ray RLlib's
model framework. These encoders are useful for testing, debugging, or scenarios
where minimal computational overhead is desired.

The dummy encoders provide pass-through functionality, returning inputs as
outputs without any transformation or parameter overhead, making them ideal
for rapid prototyping and performance baseline establishment.

Key Components:
    - :class:`DummyActorCriticEncoder`: Pass-through actor-critic encoder
    - :class:`DummyActorCriticEncoderConfig`: Configuration for dummy encoder

These implementations are particularly useful when testing RL algorithms
without the computational overhead of neural network forward passes, or when
establishing performance baselines for optimization purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, ActorCriticEncoder
from ray.rllib.core.models.configs import ActorCriticEncoderConfig

if TYPE_CHECKING:
    from ray_utilities.typing.model_return import ActorCriticEncoderOutput, Batch


class DummyActorCriticEncoder(ActorCriticEncoder):
    """Lightweight pass-through encoder for Ray RLlib actor-critic models.

    This encoder provides a minimal implementation that returns input data
    unchanged, effectively bypassing neural network processing while maintaining
    compatibility with RLlib's model framework. It's designed for scenarios
    where computational overhead needs to be minimized or for testing purposes.

    The encoder has zero parameters and performs no transformations, making it
    ideal for establishing performance baselines, debugging model architectures,
    or rapid prototyping where neural network computation is not needed.

    Features:
        - Zero parameter count for minimal memory usage
        - Pass-through functionality with no computational overhead
        - Full compatibility with RLlib's ActorCriticEncoder interface
        - Support for both actor and critic outputs
        - Framework-agnostic implementation

    Example:
        >>> config = DummyActorCriticEncoderConfig()
        >>> encoder = config.build()
        >>> output = encoder({"obs": observation_tensor})
        >>> # output["encoder_out"]["actor"] == observation_tensor

    Note:
        This encoder is particularly useful when testing algorithm logic
        without neural network overhead or when establishing performance
        baselines for optimization comparisons.

    See Also:
        :class:`DummyActorCriticEncoderConfig`: Configuration class for this encoder
        :class:`ray.rllib.core.models.base.ActorCriticEncoder`: Base encoder class
    """

    def __init__(self, config: ActorCriticEncoderConfig) -> None:
        # Do not call ActorCriticEncoder with overhead parts
        # This currently only sets self.config
        super(ActorCriticEncoder, self).__init__(config)
        self.config: ActorCriticEncoderConfig

    def _forward(self, inputs: Batch, **kwargs) -> ActorCriticEncoderOutput:  # noqa: ARG002  # pyright: ignore[reportIncompatibleMethodOverride]
        return {
            ENCODER_OUT: {  # type: ignore[return-type]
                ACTOR: inputs,
                # Add critic from value network
                **({} if self.config.inference_only else {CRITIC: inputs}),
            },
        }

    def get_num_parameters(self):
        return 0, 0

    def _set_to_dummy_weights(self, value_sequence=...) -> None:  # noqa: ARG002
        return

    # NOTE: When using frameworks this should be framework.Module shadowed
    def __call__(self, inputs: dict, **kwargs) -> ActorCriticEncoderOutput:
        return self._forward(inputs, **kwargs)


class DummyActorCriticEncoderConfig(ActorCriticEncoderConfig):
    def build(self, framework: str = "does-not-matter") -> DummyActorCriticEncoder:  # noqa: ARG002
        # Potentially init TorchModel/TfModel here
        return DummyActorCriticEncoder(self)
