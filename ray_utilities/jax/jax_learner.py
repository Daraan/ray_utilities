from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence

import jax
import jax.numpy as jnp
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.learner.tf.tf_learner import TfLearner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.policy.sample_batch import MultiAgentBatch

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
    from ray.rllib.utils.typing import ModuleID, Optimizer, Param, ParamDict, TensorType

    from ray_utilities.jax.jax_module import JaxActorCriticStateDict, JaxModule, JaxStateDict
    from rllib_port.core.sympol_module import SympolPPOModule  # FIXME: Uses downstream
    from utils.utils import ActorTrainState

logger = logging.getLogger(__name__)


class JaxLearner(Learner):
    framework = "jax"

    def __init__(
        self,
        *,
        config: "PPOConfig",
        module_spec: Optional[RLModuleSpec | MultiRLModuleSpec] = None,
        module: Optional[RLModule] = None,
    ):
        # calls configure_optimziers_for_module
        super().__init__(config=config, module_spec=module_spec, module=module)
        self.config: PPOConfig
        # Should use learner config
        # TODO
        # possible use config["accumulate_grad_batches"]
        self._accumulate_gradients_every_initial: int = config.learner_config_dict["accumulate_gradients_every"]
        # XXX Possibly do not keep them and access via self.module!
        self._states: dict[ModuleID, JaxStateDict | Mapping[str, Any]] = {}

    # calls configure_optimziers_for_module
    # def configure_optimizers(self) -> None:
    #    return super().configure_optimizers()

    def configure_optimizers_for_module(
        self,
        module_id: ModuleID,
        config: Optional["AlgorithmConfig"] = None,  # noqa: ARG002
    ) -> None:
        # MAYBE NOT NEEDED
        module: JaxModule = self._module[module_id]  # type: ignore[assignment]
        # likely do not need these here
        actor_params, critic_params = self.get_parameters(module)  # Re-enable this line
        # Optimizer is set in init_state
        self._states[module_id] = module.get_state(inference_only=False)

        if False:
            # commented out to avoid linter errors
            self.register_optimizer(
                module_id=module_id,
                # optimizer=optimizer,
                params=actor_params,
                # lr_or_lr_schedule=config.lr,
            )
            # module.states["actor"].tx = optimizer
            self.register_optimizer(
                module_id=module_id,
                # optimizer=optimizer,
                params=critic_params,
                # lr_or_lr_schedule=config.lr,
            )

    def get_parameters(self, module: SympolPPOModule | Any) -> tuple[Sequence[Param], Sequence[Param]]:
        logger.warning("JaxLearner.get_parameters called which is not fully implemented", stacklevel=2)
        # module.states["actor"].params is a dict
        return list(module.states["actor"].params.values()), list(module.states["critic"].params.values())

    def get_param_ref(self, param: Param) -> Hashable:
        # Reference to param: self._params[param_ref] = param
        logger.warning("JaxLearner.get_param_ref called which is not fully implemented")
        return param

    def compute_gradients(self, *args, **kwargs) -> ParamDict:  # noqa: ARG002
        # TODO: Can this be its own function?
        logger.warning("compute_gradients called which is not used in the jax implementation")
        raise NotImplementedError("compute_gradients not implemented for jax")

    # jittable
    @abstractmethod
    def apply_gradients(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        # Normally dict[Hashable | ParamRef, Param]
        gradients_dict: dict[ModuleID, Any],
        *,
        states: Mapping[ModuleID, Any],
    ) -> Mapping[ModuleID, Any]:
        """
        Apply the gradients to the passed states. This function should be implemented in a way to
        be jittable, meaning it should not have side effects and should return the new states.

        Attention:
            This has a different signature to RLlib's Learner.apply_gradients to be jittable.
        """

    def _convert_batch_type(self, batch: MultiAgentBatch) -> MultiAgentBatch:
        # TODO: put on device
        logger.warning("_convert_batch_type called which is not fully implemented")
        length = max(len(b) for b in batch.values())
        batch = MultiAgentBatch(batch, env_steps=length)
        return batch

    @staticmethod
    def _get_clip_function():
        logger.warning("_get_clip_function called which is not fully implemented")
        if 0:
            # returns
            from ray.rllib.utils.tf_utils import clip_gradients
            from ray.rllib.utils.torch_utils import clip_gradients
        # possibly use optax.clip; but needs to be in transformation pipeline
        from ray_utilities.jax.math import clip_gradient

        # Has wrong interface
        return clip_gradient

    @staticmethod
    def _get_global_norm_function() -> Any:
        logger.warning("_get_global_norm_function called which is not fully implemented")
        return super(JaxLearner)._get_global_norm_function()

    def _get_tensor_variable(self, value: Any, dtype: Any = None, trainable: bool = False) -> TensorType:  # noqa
        # TODO: is kl_coeffs a variable that is learned?
        logger.warning("_get_tensor_variable called which is not fully implemented", stacklevel=2)
        if 0:
            TorchLearner._get_tensor_variable(value, dtype, trainable)
            TfLearner._get_tensor_variable(value, dtype, trainable)
        v = jnp.array(value, dtype=dtype)
        if not trainable:
            v = jax.lax.stop_gradient(v)
        return v

    @staticmethod
    def _get_optimizer_lr(optimizer: Optimizer) -> float:
        logger.warning("_get_optimizer_lr called which is not fully implemented")
        return super(JaxLearner)._get_optimizer_lr(optimizer)

    @staticmethod
    def _set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
        logger.warning("_set_optimizer_lr called which is not fully implemented")
        # Needs to change opt_state
        # TODO: reduce lr not implemented
        super(JaxLearner)._set_optimizer_lr(optimizer, lr)

    def _get_optimizer_state(self, *args, **kwargs):
        logger.warning("_get_optimizer_state called which is not fully implemented")
        return super()._get_optimizer_state(*args, **kwargs)


# pyright: reportAbstractUsage=information
if TYPE_CHECKING:
    __conf: Any = ...
    JaxLearner(config=__conf)  # still abstract
