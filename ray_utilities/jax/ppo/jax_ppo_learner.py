from __future__ import annotations

# some of ray's TypeAlias are not detected as such
# pyright: reportInvalidTypeForm=warning
import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner as RayPPOLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.learner.learner import ENTROPY_KEY, POLICY_LOSS_KEY, VF_LOSS_KEY
from ray.rllib.core.learner.tf.tf_learner import TfLearner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.apis import SelfSupervisedLossAPI

from ray_utilities.connectors.debug_connector import add_debug_connectors
from ray_utilities.connectors.dummy_connector import DummyNumpyToTensor
from ray_utilities.jax.jax_learner import JaxLearner
from ray_utilities.jax.ppo.compute_ppo_loss import make_jax_compute_ppo_loss_function

if TYPE_CHECKING:
    from collections.abc import Mapping

    import chex
    from flax.training.train_state import TrainState
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.rllib.utils.typing import ModuleID, ResultDict, TensorType

    from ray_utilities.jax.ppo.jax_ppo_module import JaxActorCriticStateDict, JaxPPOModule
    from ray_utilities.typing.jax import type_grad_and_value

logger = logging.getLogger(__name__)


class JaxPPOLearner(RayPPOLearner, JaxLearner):
    def build(self, **kwargs) -> None:
        super().build(**kwargs)
        self._states: Mapping[ModuleID, JaxActorCriticStateDict]
        self._rng_key = self.config.learner_config_dict["rng_key"]
        if self._learner_connector is not None and (self.config.add_default_connectors_to_learner_pipeline):
            # super().build adds (PPO) AddOneTsToEpisodesAndTruncate, GeneralAdvantageEstimation
            # Remove NumpyToTensorConnector, if present.
            self._learner_connector.remove(name_or_class="NumpyToTensor")

            # At the end of the pipeline (when the batch is already completed), add the
            # GAE connector, which performs a vf forward pass, then computes the GAE
            # computations, and puts the results of this (advantages, value targets)
            # directly back in the batch. This is then the batch used for
            # `forward_train` and `compute_losses`.
            if self.config.learner_config_dict.get("no_numpy_to_tensor_connector", True):
                idx = -1
                for idx, con in enumerate(self._learner_connector.connectors):  # noqa: B007
                    if con.__class__ is GeneralAdvantageEstimation:
                        break
                if idx >= 0:
                    con = cast("GeneralAdvantageEstimation", self._learner_connector.connectors[idx])
                    # Add connector that does NOT modify the batch, does not convert to numpy
                    con._numpy_to_tensor_connector = DummyNumpyToTensor(as_learner_connector=True)
                    # TODO: if using no converter need monkeypatch; maybe convert there
                    # con._numpy_to_tensor_connector = LimitedToNumpyConverter()
            # not needed anymore; state passed to self.vf(obs, state=state)
            # self._learner_connector.append(RemoveStateFromBatch())
        add_debug_connectors(self)
        self._compute_loss_for_modules = {
            module_id: make_jax_compute_ppo_loss_function(
                module,  # pyright: ignore[reportArgumentType]  # has to be JaxPPOModule
                self.config,
            )
            for module_id, module in self.module.items()
        }
        if TYPE_CHECKING:
            # _forward_with_gradds is used inside _update_jax
            self._forward_with_grads = type_grad_and_value(self._jax_forward_pass)
        else:
            self._forward_with_grads = jax.jit(jax.value_and_grad(self._jax_forward_pass, has_aux=True, argnums=(0,)))
        self._update_jax = jax.jit(self._update_jax)

    # jittable
    @staticmethod
    def _get_state_parameters(
        states: Mapping[ModuleID, JaxActorCriticStateDict],
    ) -> dict[ModuleID, dict[Literal["actor", "critic"], Any]]:
        parameters: dict[ModuleID, dict[Literal["actor", "critic"], Any]] = dict.fromkeys(
            states.keys(), cast("dict", None)
        )
        for module_id, state in states.items():
            parameters[module_id] = {
                "actor": state["actor"].params,
                "critic": state["critic"].params,
            }
        return parameters

    def compute_loss_for_module(  # pyright: ignore[reportIncompatibleMethodOverride]  # additional params
        self,
        *,
        critic_state_params: Optional[Mapping[str, Any]],
        module_id: ModuleID,
        config: "AlgorithmConfig | PPOConfig",  # noqa: ARG002
        batch: SampleBatch | dict[str, Any],
        fwd_out: dict[str, TensorType],
        curr_entropy_coeff: float | chex.Numeric | TensorType,
        curr_kl_coeff: Optional[float | chex.Numeric | TensorType],
    ) -> tuple[TensorType, dict[str, chex.Numeric]]:
        # jittable and grad wrt critic_state_params
        (
            total_loss,
            (
                mean_entropy,
                mean_vf_loss,
                mean_vf_unclipped_loss,
                variance_explained,
                policy_loss_key,
                mean_kl_loss,
            ),
        ) = self._compute_loss_for_modules[module_id](
            # batch is a SampleBatch which is not compatible
            critic_state_params if critic_state_params is not None else self._states[module_id]["critic"].params,
            batch=batch,
            fwd_out=fwd_out,
            curr_entropy_coeffs=curr_entropy_coeff,
            curr_kl_coeffs=curr_kl_coeff,
        )

        # Return the total loss.
        return total_loss, {
            POLICY_LOSS_KEY: policy_loss_key,
            VF_LOSS_KEY: mean_vf_loss,
            LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
            LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: variance_explained,
            ENTROPY_KEY: mean_entropy,
            LEARNER_RESULTS_KL_KEY: mean_kl_loss,
        }

    def compute_losses(self, *, fwd_out: ResultDict[str, Any], batch: ResultDict[str, Any]):
        """
        NOTE:
            Use _jax_compute_losses instead of this function to compute gradients
        """
        logger.warning("compute_losses called, which makes no use of jit - additional step in shedule")
        curr_entropy_coeffs, curr_kl_coeffs = self._generate_curr_coeffs()
        loss_per_module, aux_data = self._jax_compute_losses(
            parameters=self._get_state_parameters(self._states),
            fwd_out=fwd_out,
            batch=batch,
            curr_entropy_coeffs=curr_entropy_coeffs,
            curr_kl_coeffs=curr_kl_coeffs,
        )
        return loss_per_module

    def _jax_compute_losses(
        self,
        parameters: dict[ModuleID, dict[Literal["actor", "critic"], Mapping[str, Any]]],
        fwd_out: dict[str, Any],
        batch: dict[str, Any],
        curr_entropy_coeffs: dict[ModuleID, float | chex.Numeric | TensorType],
        curr_kl_coeffs: Optional[dict[ModuleID, float | chex.Numeric | TensorType]] = None,
    ):
        loss_per_module = {}
        aux_data = {}

        for module_id, module_state in parameters.items():
            module_batch = batch[module_id]
            module_fwd_out = fwd_out[module_id]
            module = self.module[module_id].unwrapped()
            if isinstance(module, SelfSupervisedLossAPI):
                logger.error("Self-supervised loss not implemented with jax suport")
                loss = module.compute_self_supervised_loss(
                    learner=self,
                    module_id=module_id,
                    config=self.config.get_config_for_module(module_id),
                    batch=module_batch,
                    fwd_out=module_fwd_out,
                )
            else:
                loss, aux = self.compute_loss_for_module(
                    module_id=module_id,
                    config=self.config.get_config_for_module(module_id),  # pyright: ignore[reportArgumentType]
                    batch=dict(module_batch),
                    fwd_out=module_fwd_out,
                    critic_state_params=module_state["critic"],
                    curr_entropy_coeff=curr_entropy_coeffs[module_id],
                    curr_kl_coeff=curr_kl_coeffs[module_id] if curr_kl_coeffs else None,
                )
                aux_data[module_id] = aux
            loss_per_module[module_id] = loss

        return loss_per_module, aux_data

    def _forward_train_call(
        self, batch, parameters: dict[ModuleID, dict[Literal["actor", "critic"], Mapping[str, Any]]], **kwargs
    ):
        """jittable"""
        fwd_out = {
            mid: cast("JaxPPOModule", self.module._rl_modules[mid])._forward_train(
                batch[mid], parameters=parameters[mid]["actor"], **kwargs
            )
            for mid in batch.keys()
            if mid in self.module
        }
        return fwd_out

    # NOTE: do not pass indices as states
    # @jax.jit
    # @partial(jax.value_and_grad, has_aux=True, argnums=(0,))
    def _jax_forward_pass(
        self,
        parameters: dict[ModuleID, dict[Literal["actor", "critic"], Mapping[str, Any]]],
        batch: dict[str, Any],
        curr_entropy_coeffs: dict[ModuleID, float | chex.Numeric | TensorType],
        curr_kl_coeffs: Optional[dict[ModuleID, float | chex.Numeric | TensorType]] = None,
    ) -> tuple[chex.Numeric, tuple[Any, dict[ModuleID, chex.Numeric], dict[str, Any]]]:
        """
        Note:
            Do not use directly use wrapped version: `self._forward_with_grads`
        """
        fwd_out = self._forward_train_call(batch, parameters=parameters)
        loss_per_module, compute_loss_aux = self._jax_compute_losses(
            parameters, fwd_out, batch, curr_entropy_coeffs, curr_kl_coeffs
        )
        # gradient needs a scalar loss:
        return jax.tree.reduce(jnp.sum, loss_per_module), (fwd_out, loss_per_module, compute_loss_aux)

    def _update_jax(
        self,
        states: Mapping[ModuleID, JaxActorCriticStateDict],
        batch: dict[str, Any],
        curr_entropy_coeffs: dict[ModuleID, float | chex.Numeric | TensorType],
        curr_kl_coeffs: Optional[dict[ModuleID, float | chex.Numeric | TensorType]] = None,
        *,
        accumulate_gradients_every: int,  # possibly make book and static
    ) -> tuple[Mapping[ModuleID, JaxActorCriticStateDict], tuple[Any, dict[ModuleID, chex.Numeric], dict[str, Any]]]:
        # TODO: Could make accumulate_gradients_every a bool -> two different jax-compilations, BUT only when not taking mean  # noqa: E501
        parameters = self._get_state_parameters(states)
        gradients: dict[ModuleID, dict[Literal["actor", "critic"], Any]]
        (_all_losses_combined, (fwd_out, loss_per_module_do_not_use, compute_loss_aux)), (gradients,) = (
            self._forward_with_grads(parameters, batch, curr_entropy_coeffs, curr_kl_coeffs)
        )
        if 0:
            # consider if implementation is necessary
            self.postprocess_gradients_for_module
            postprocessed_gradients: dict = self.postprocess_gradients(gradients)  # type: ignore
        else:
            postprocessed_gradients = gradients
        new_states = self.apply_gradients(
            postprocessed_gradients, states=states, accumulate_gradients_every=accumulate_gradients_every
        )
        return new_states, (fwd_out, loss_per_module_do_not_use, compute_loss_aux)

    def _generate_curr_coeffs(
        self,
    ) -> tuple[dict[ModuleID, TensorType | chex.Numeric], dict[ModuleID, TensorType | chex.Numeric]]:
        """Get the current entropy and kl coefficients for each module."""
        curr_entropy_coeffs: dict[ModuleID, TensorType | chex.Numeric] = {}
        curr_kl_coeffs: dict[ModuleID, TensorType | chex.Numeric] = {}
        for module_id in self.module.keys():
            curr_entropy_coeffs[module_id] = self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
            # Note: curr_kl is piped trough self._get_tensor_variable
            curr_kl_coeffs[module_id] = self.curr_kl_coeffs_per_module[module_id]
        return curr_entropy_coeffs, curr_kl_coeffs

    def _update(self, batch: dict[str, Any] | SampleBatch, **kwargs) -> tuple[Any, Any, Any]:  # noqa: ARG002
        """
        NOTE: The amount of processed data is minibatch_size * epochs

        Calls a.o.
        fwd_out = self.module.forward_train(batch)
        loss_per_module = self.compute_losses(fwd_out=fwd_out, batch=batch)
        gradients = self.compute_gradients(loss_per_module)
        postprocessed_gradients = self.postprocess_gradients(gradients)
        self.apply_gradients(postprocessed_gradients)
        """
        # possibly use jit and wrap them all
        if 0:
            TfLearner._untraced_update
            TorchLearner._uncompiled_update
        # get them from somewhere else?
        self.metrics.activate_tensor_mode()
        # fwd_out = self.module.forward_train(batch)
        # Cannot pass SampleBatch as input
        # TODO: fwd_out["default_policy"]["embeddings"] has many keys
        curr_entropy_coeffs, curr_kl_coeffs = self._generate_curr_coeffs()
        new_states, (fwd_out, loss_per_module, compute_loss_aux) = self._update_jax(
            states=self._states,
            batch={mid: dict(v) for mid, v in batch.items()},
            curr_entropy_coeffs=curr_entropy_coeffs,
            curr_kl_coeffs=curr_kl_coeffs,
            accumulate_gradients_every=self.config.learner_config_dict.get("accumulate_gradients_every", 1),
        )
        self._states = new_states  # pyright: ignore[reportIncompatibleVariableOverride]
        self.module.set_state(self._states)  # pyright: ignore[reportArgumentType]

        # Log important loss stats.
        for module_id in fwd_out.keys():
            self.metrics.log_dict(
                compute_loss_aux[module_id],
                key=module_id,
                window=1,  # <- single items (should not be mean/ema-reduced over time).
            )
        return fwd_out, loss_per_module, self.metrics.deactivate_tensor_mode()

    # jittable
    def apply_gradients(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        # Normally dict[Hashable | ParamRef, Param]
        gradients_dict: dict[ModuleID, dict[Literal["actor", "critic"], jax.Array]],
        *,
        states: Mapping[ModuleID, JaxActorCriticStateDict],
        accumulate_gradients_every: int,
    ) -> Mapping[ModuleID, JaxActorCriticStateDict]:
        """
        Apply the gradients to the passed states. Should be implemented in a way to
        be jittable, meaning it should not have side effects and should return the new states.

        Attention:
            This has a different signature to RLlib's Learner.apply_gradients to be jittable.
        """
        for module_id in self.module.keys():
            module_grads = gradients_dict[module_id]
            critic_grads = module_grads["critic"]
            states[module_id]["critic"] = states[module_id]["critic"].apply_gradients(grads=critic_grads)

            # actor
            actor_grads = module_grads["actor"]
            actor_grad_accum = jax.tree_util.tree_map(
                lambda x, y: x + y,
                actor_grads,
                states[module_id]["actor"].grad_accum,  # TODO: Fix interface TrainState with grad_accum
            )
            if "legacy" == "code":  # legacy code steps likely wrong apply_gradient is called at least once or twice.
                # jax.debug.print("Actor step start (legacy): {}", states[module_id]["actor"].step)
                actor_state = states[module_id]["actor"].apply_gradients(grads=actor_grads)

                def update_fn(actor_state=actor_state, actor_grad_accum=actor_grad_accum):
                    jax.debug.print("Actor applying accumulated gradients")
                    grads = jax.tree_util.tree_map(lambda x: x / accumulate_gradients_every, actor_grad_accum)
                    new_state = actor_state.apply_gradients(
                        grads=grads,
                        grad_accum=jax.tree_util.tree_map(jnp.zeros_like, grads),  # reset accumulated to zero.
                    )
                    return new_state

                # jax.debug.print(
                #     "Actor state step (legacy): {}, accumulate_gradients_every={}",
                #     actor_state.step,
                #     accumulate_gradients_every,
                # )
                actor_state = jax.lax.cond(
                    # NOTE: should do a + 1 to not update at step 0 - however the apply above sets it to 1 at this point
                    actor_state.step % accumulate_gradients_every
                    == 0,  # TODO: For gradient accumulation must be passed as argument
                    lambda _: update_fn(),
                    lambda _, actor_state=actor_state, actor_grad_accum=actor_grad_accum: actor_state.replace(
                        grad_accum=actor_grad_accum,
                        step=actor_state.step + 1,  # step increased by apply gradient already
                    ),
                    None,
                )
                # jax.debug.print("Actor step after (legacy): {}", actor_state.step)
            else:

                def apply_fn(accum_grads_and_state: tuple[Any, TrainState]):
                    accum_grads, state = accum_grads_and_state
                    updated_state = state.apply_gradients(grads=accum_grads)
                    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
                    return updated_state.replace(grad_accum=zero_grads)

                def accumulate_only_fn(accum_grads_and_state: tuple[Any, TrainState]):
                    accum_grads, state = accum_grads_and_state
                    return state.replace(grad_accum=accum_grads, step=state.step + 1)

                actor_state: TrainState = states[module_id]["actor"]

                # Use lax.cond to decide what to do
                actor_state = jax.lax.cond(
                    # +1 to not apply on step 0
                    (actor_state.step + 1) % accumulate_gradients_every == 0,
                    apply_fn,
                    accumulate_only_fn,
                    operand=(actor_grad_accum, actor_state),
                )
            states[module_id]["actor"] = actor_state
        return states

    def _update_module_kl_coeff(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        kl_loss: float,
    ) -> None:
        # Note uses:
        # TODO: check if this is called and should it be called
        logger.warning("_update_module_kl_coeff called which is only minimally implemented")
        # Does not use any torch functions; uses kl_target and curr_kl_coeffs_per_module
        PPOTorchLearner._update_module_kl_coeff(
            self,  # pyright: ignore[reportArgumentType]
            module_id=module_id,
            config=config,
            kl_loss=kl_loss,
        )


# Check ABC
if TYPE_CHECKING:
    __conf: Any = ...
    JaxPPOLearner(config=__conf)
