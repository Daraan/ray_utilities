from __future__ import annotations

from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

import logging

from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT

from ray_utilities.jax.jax_module import JaxActorCriticEncoder, JaxModule, JaxActorCriticStateDict

if TYPE_CHECKING:
    from ray.rllib.utils.typing import TensorType

    from ray_utilities.jax.jax_model import FlaxRLModel, JaxRLModel

_logger = logging.getLogger(__name__)


class JaxPPOModule(DefaultPPORLModule, JaxModule):
    config: object
    """deprecated do not use"""

    def __init__(self, *args, **kwargs):
        # Possibly use catalog_class here
        super().__init__(*args, **kwargs)
        self.encoder: JaxActorCriticEncoder

    def setup(self) -> None:
        super().setup()
        self.pi: JaxRLModel | FlaxRLModel
        self.vf: FlaxRLModel
        import jax  # import jax late, to avoid some os.fork warnings  # noqa: PLC0415

        actor = self.pi
        critic = self.vf

        if (
            not self.model_config
            or (is_dataclass(self.model_config) and (seed := getattr(self.model_config, "seed", -1) == -1))
            or (seed := self.model_config.get("seed", -1)) == -1  # pyright: ignore[reportAttributeAccessIssue]
        ):
            _logger.warning(
                "No seed provided in the model_config_dict of %s, using default seed 123.", self.__class__.__name__
            )
            seed = 123
        module_key = jax.random.PRNGKey(seed)
        module_key, actor_key, critic_key = jax.random.split(module_key, 3)

        assert self.observation_space is not None
        sample = self.observation_space.sample()
        actor_state = actor.init_state(actor_key, sample)
        critic_state = critic.init_state(critic_key, sample)

        self.states: JaxActorCriticStateDict  # pyright: ignore[reportIncompatibleVariableOverride]
        self.set_state(
            JaxActorCriticStateDict(
                {
                    "actor": actor_state,
                    "critic": critic_state,
                    "module_key": module_key,
                }
            )
        )

    def _forward(self, batch: dict[str, Any], *, parameters: Mapping[str, Any], **kwargs) -> dict[str, Any]:
        """Default forward pass (used for inference and exploration)."""
        output = {}
        # Encoder forward pass.
        encoder_outs = self.encoder(batch)
        # Stateful encoder?
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]  # pyright: ignore[reportGeneralTypeIssues]  # key is present
        # Pi head
        # NOTE: Current error likely related to pyright error
        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR], parameters=parameters, **kwargs)
        return output

    def _forward_train(self, batch: dict[str, Any], *, parameters: Mapping[str, Any], **kwargs) -> dict[str, Any]:
        """Train forward pass (keep embeddings for possible shared value func. call)."""
        output = {}
        encoder_outs = self.encoder(batch)
        output[Columns.EMBEDDINGS] = encoder_outs[ENCODER_OUT][CRITIC]  # pyright: ignore[reportTypedDictNotRequiredAccess]  # key present during train
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]  # pyright: ignore[reportGeneralTypeIssues]  # key is present
        # NOTE: Current error likely related to pyright error
        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR], parameters=parameters, **kwargs)
        return output

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: Optional[Any] = None,
        *,
        parameters: Optional[
            Mapping[str, Any]
        ] = None,  # XXX For GaeIn the Connector pipeline setting this to None; however it may not be omitted for gradient computation  # noqa: E501
    ) -> TensorType:
        """Computes the value estimates given `batch`.

        Note:
            To allow gradient computation, pass `parameters` via keyword argument,
            otherwise set it to None

        Args:
            batch: The batch to compute value function estimates for.
            embeddings: Optional embeddings already computed from the `batch` (by
                another forward pass through the model's encoder (or other subcomponent
                that computes an embedding). For example, the caller of thie method
                should provide `embeddings` - if available - to avoid duplicate passes
                through a shared encoder.
            parameters: Parameters of the model.

        Returns:
            A tensor of shape (B,) or (B, T) (in case the input `batch` has a
            time dimension.

            Attention:
                That the last value dimension should already be squeezed out (not 1!).
        """
        if embeddings is None:
            # Separate vf-encoder.
            if hasattr(self.encoder, "critic_encoder"):
                batch_ = batch
                if self.is_stateful():
                    # The recurrent encoders expect a `(state_in, h)`  key in the
                    # input dict while the key returned is `(state_in, critic, h)`.
                    batch_ = batch.copy()
                    batch_[Columns.STATE_IN] = batch[Columns.STATE_IN][CRITIC]
                embeddings = self.encoder.critic_encoder(batch_)[ENCODER_OUT]  # pyright: ignore[reportOptionalCall]
            # Shared encoder.
            else:
                embeddings = self.encoder(batch)[ENCODER_OUT][CRITIC]  # pyright: ignore[reportTypedDictNotRequiredAccess]  # key present during train
            assert embeddings is not None

        if False and parameters is None:
            _logger.debug(
                "No parameters passed to compute_values, using current parameters; "
                "this is ONLY fine when called in general_advantage_estimation.",
                stack_info=False,
                stacklevel=2,
            )
        # Value head. Should not be a list even in continuous case.
        vf_out = self.vf(
            embeddings,
            parameters=parameters if parameters is not None else self.states[CRITIC].params,
        )
        vf_out = vf_out.squeeze(axis=-1)  # pyright: ignore[reportArgumentType]
        # NEW: # TODO: rllib does not add this to batch here, why; do during learner update?
        # During rollout we do not need a JAX array here; casting would also allow to get rid of SampleBatch monkeypatch
        # during inference/rollout this used stop_gradient
        # NOTE: Cannot use this when inside jit
        # possibly use. https://github.com/jax-ml/jax/discussions/9241
        # if not self.inference_only:  # probably only need this in legacy
        #    batch[Columns.VF_PREDS] = np.asarray(vf_out)
        return vf_out


# Check ABC
if TYPE_CHECKING:
    JaxPPOModule()
