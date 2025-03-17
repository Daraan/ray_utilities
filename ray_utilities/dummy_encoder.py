from __future__ import annotations

from ray.rllib.core.models.base import ActorCriticEncoder

from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT

from ray.rllib.core.models.configs import ActorCriticEncoderConfig



class DummyActorCriticEncoder(ActorCriticEncoder):
    """A dummy encoder that just outputs the input as is."""

    def __init__(self, config: ActorCriticEncoderConfig) -> None:
        # Do not call ActorCriticEncoder with overhead parts
        # This currently only sets self.config
        super(ActorCriticEncoder, self).__init__(config)
        self.config: ActorCriticEncoderConfig

    def _forward(self, inputs: dict, **kwargs) -> dict:  # noqa: ARG002
        return {
            ENCODER_OUT: {
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
    def __call__(self, inputs: dict, **kwargs) -> dict:
        return self._forward(inputs, **kwargs)


class DummyActorCriticEncoderConfig(ActorCriticEncoderConfig):
    def build(self, framework: str = "does-not-matter") -> DummyActorCriticEncoder:  # noqa: ARG002
        # Potentially init TorchModel/TfModel here
        return DummyActorCriticEncoder(self)
