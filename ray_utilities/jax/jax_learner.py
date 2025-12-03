"""JAX-based learner implementation for Ray RLlib.

This module provides a JAX-specific implementation of the RLlib Learner interface,
enabling the use of JAX for neural network training within the Ray RLlib framework.
The implementation includes support for:

- JAX neural network modules and state management
- Integration with Ray RLlib's learner architecture
- PPO algorithm support with JAX backends
- Gradient accumulation and optimization workflows

Note:
    This is an experimental implementation with some methods marked as incomplete
    or having warnings about their implementation status.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Optional, Sequence

import jax
import jax.numpy as jnp
import optax
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.util import log_once

from ray_utilities.jax.math import clip_gradients
from ray_utilities.nice_logger import ImportantLogger

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
    from ray.rllib.utils.typing import ModuleID, Optimizer, Param, ParamDict, StateDict, TensorType

    from ray_utilities.jax.jax_module import JaxActorCriticStateDict, JaxModule, JaxStateDict
    from ray_utilities.jax.ppo.jax_ppo_module import JaxPPOModule

logger = logging.getLogger(__name__)


class JaxOptStateWrapper:
    """Need a hashable object to be a valid Optimizer in RLlib Learner."""

    def __init__(self, module: JaxModule):
        self.module = module


class JaxLearner(Learner):
    """JAX-based implementation of the Ray RLlib Learner interface.

    This class provides a JAX backend for neural network training within the
    Ray RLlib framework. It extends the base :class:`ray.rllib.core.learner.learner.Learner`
    class to support JAX-specific operations and state management.

    The learner handles:
        - JAX neural network modules and their states
        - Optimizer configuration and gradient accumulation
        - Integration with RLlib's training pipeline
        - PPO algorithm support with JAX backends

    Attributes:
        config: The PPO configuration object.
        _accumulate_gradients_every_initial: Number of batches to accumulate
            gradients over before applying updates.
        _states: Dictionary mapping module IDs to their JAX states.

    Warning:
        This is an experimental implementation. Some methods may emit warnings
        about incomplete functionality or may not be fully implemented.

    Example:
        >>> learner = JaxLearner(config=ppo_config, module_spec=module_spec)
        >>> # Use with RLlib training pipeline
    """

    framework = "torch"
    """Set to "jax" to indicate the backend framework. Possibly needs to be changed for rllib compatibility."""

    def __init__(
        self,
        *,
        config: "PPOConfig",
        module_spec: Optional[RLModuleSpec | MultiRLModuleSpec] = None,
        module: Optional[RLModule] = None,
    ):
        """Initialize the JAX learner with configuration and module specifications.

        Args:
            config: PPO algorithm configuration object containing learning parameters
                and optimizer settings.
            module_spec: Optional specification for creating the RLModule. Either this
                or ``module`` should be provided.
            module: Optional pre-instantiated RLModule. Either this or ``module_spec``
                should be provided.

        Note:
            This constructor calls the parent's ``configure_optimizers_for_module``
            method and sets up gradient accumulation based on the learner configuration.
        """
        # calls configure_optimziers_for_module
        super().__init__(config=config, module_spec=module_spec, module=module)
        self.config: PPOConfig
        # Should use learner config
        # TODO
        # possible use config["accumulate_grad_batches"]
        self._accumulate_gradients_every_initial: int = config.learner_config_dict["accumulate_gradients_every"]

    # calls configure_optimziers_for_module
    # def configure_optimizers(self) -> None:
    #    return super().configure_optimizers()

    def get_jax_states(self) -> dict[ModuleID, JaxStateDict | JaxActorCriticStateDict]:
        """Get the current JAX states for all modules in the learner.

        Returns:
            A dictionary mapping module IDs to their corresponding JAX state dictionaries.
        """
        if self._module is None:
            return {}
        states = {}
        for module_id, module in self._module.items():
            states[module_id] = module.get_state(inference_only=False)["jax_state"]
        return states

    def configure_optimizers_for_module(
        self,
        module_id: ModuleID,
        config: Optional["AlgorithmConfig"] = None,  # noqa: ARG002
    ) -> None:
        # MAYBE NOT NEEDED
        super().configure_optimizers_for_module(module_id, config)
        logger.warning("JaxLearner.configure_optimizers_for_module called which is not fully implemented", stacklevel=2)
        module: JaxModule = self._module[module_id]  # type: ignore[assignment]
        # likely do not need these here
        actor_params, critic_params = self.get_parameters(module)  # Re-enable this line
        # Optimizer is set in init_state
        # TODO: PPO version
        params = [actor_params, critic_params]
        states = self.get_jax_states()[module_id]
        opt_states: Sequence[optax.OptState] = tuple(
            state.opt_state for state in states.values() if hasattr(state, "opt_state")
        )
        assert len(params) == len(opt_states)
        # logger.info(f"Module {module_id} has optimizers with states: {opt_states}")
        # should be a dict like {'actor': optax.OptState, 'critic': optax.OptState} where opt state is a tuple

        # commented out to avoid linter errors
        self.register_optimizer(
            module_id=module_id,
            optimizer=JaxOptStateWrapper(module),  # needs to be a hashable
            params=params,  # needs to be list. NOTE: The parameters change with every iteration!
            lr_or_lr_schedule=config.lr if config else None,
        )

    def get_parameters(self, module: JaxPPOModule | Any) -> tuple[Sequence[Param], Sequence[Param]]:
        # TODO: This is PPO; should move it to a subclass and make this abstract in base
        logger.warning("JaxLearner.get_parameters called which is not fully implemented", stacklevel=2)
        # module.states["actor"].params is a dict
        return list(module.states["actor"].params.values()), list(module.states["critic"].params.values())

    def get_param_ref(self, param: Param) -> Hashable:
        # Reference to param: self._params[param_ref] = param
        # param is a list but we need to return a Hashable
        param_ref = id(param)
        logger.warning("JaxLearner.get_param_ref called which is not fully implemented")
        return param_ref

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
        return MultiAgentBatch(batch, env_steps=length)

    @staticmethod
    def _get_clip_function():
        return clip_gradients

    @staticmethod
    def _get_global_norm_function() -> Any:
        # should behave like
        if 0:
            from ray.rllib.utils.torch_utils import compute_global_norm
        return optax.global_norm

    def _get_tensor_variable(self, value: Any, dtype: Any = None, trainable: bool = False) -> TensorType:  # noqa: FBT001, FBT002
        # TODO: is kl_coeffs a variable that is learned?
        logger.warning("_get_tensor_variable called which is not fully implemented", stacklevel=2)
        if 0:
            # References for implementation
            from ray.rllib.core.learner.tf.tf_learner import (  # pyright: ignore[reportMissingImports] # removed somewhere around 2.48
                TfLearner,
            )

            TorchLearner._get_tensor_variable(value, dtype, trainable)
            TfLearner._get_tensor_variable(value, dtype, trainable)
        v = jnp.array(value, dtype=dtype)
        if not trainable:
            v = jax.lax.stop_gradient(v)
        return v

    def _get_optimizer_lr(self, optimizer: Optimizer) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        # Attempt to get LR from optimizer state if it follows Flax/Optax patterns with inject_hyperparams
        opt_states = self._get_optimizer_state()
        lrs = self._find_lrs(opt_states, key_or_pattern="learning_rate", strict=True)
        if len(lrs) > 1 and log_once("multiple_lrs_found"):
            ImportantLogger.important_info(
                logger,
                f"_get_optimizer_lr found multiple learning rates {lrs}, returning the first one. "
                "To log individual lrs implement a custom logging mechanism.",
            )
        return lrs[0]

    @staticmethod
    def _find_lrs(
        opt_state: optax.OptState | list[optax.OptState],
        key_or_pattern="learning_rate",
        update: Optional[float] = None,
        strict: bool = True,
    ) -> list[float]:
        found_lrs = []
        hyperparams = optax.tree_utils.tree_get_all_with_path(opt_state, "hyperparams")
        for _path, hp in hyperparams:
            if "*" not in key_or_pattern:
                if key_or_pattern in hp:
                    found_lrs.append(float(hp[key_or_pattern]))
                    if update is not None:
                        hp[key_or_pattern] = jnp.array(update) if not isinstance(update, jnp.ndarray) else update
                elif strict:
                    raise KeyError(f"Could not find learning rate with key {key_or_pattern} in hyperparams {hp}")
            else:
                keys = hp.keys()
                matched = False
                for k in keys:
                    if fnmatch(k, key_or_pattern):
                        found_lrs.append(float(hp[k]))
                        matched = True
                        if update is not None:
                            hp[k] = jnp.array(update) if not isinstance(update, jnp.ndarray) else update
                if strict and not matched:
                    raise KeyError(
                        f"Could not find learning rate with key pattern {key_or_pattern} in hyperparams {hp}"
                    )
        return found_lrs

    def _set_optimizer_lr(self, optimizer: Any, lr: float) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        # JAX optimizers (Optax) are functional and states are immutable.
        # Setting LR in place is not typically supported unless using a mutable wrapper.
        # ImportantLogger.important_warning(logger, "_set_optimizer_lr called which is not implemented and a no-op")
        # TODO: allow multiple optimizers
        # states = self.get_jax_states()
        opt_states = self._get_optimizer_state()
        self._find_lrs(opt_states, key_or_pattern="learning_rate", update=lr, strict=True)

    def _get_optimizer_state(self, module: Optional[JaxModule] = None, module_id: Optional[str] = None) -> StateDict:
        """Returns the state of all optimizers currently registered in this Learner."""
        optimizer_states = {}
        if not module:
            states = self.get_jax_states()
        else:
            module_id = module_id if module_id else DEFAULT_MODULE_ID
            states = {DEFAULT_MODULE_ID: module.states}
        for mid, state_dict in states.items():
            if module_id is not None and module_id != mid:
                continue
            optimizer_states[mid] = {}
            for key, train_state in state_dict.items():
                if hasattr(train_state, "opt_state"):
                    optimizer_states[mid][key] = train_state.opt_state  # pyright: ignore[reportAttributeAccessIssue]
        return optimizer_states

    def _set_optimizer_state(self, state: StateDict) -> None:
        """Sets the state of all optimizers currently registered in this Learner."""
        states = self.get_jax_states()
        for module_id, module_opt_states in state.items():
            if module_id not in states:
                continue
            for key, opt_state in module_opt_states.items():
                if key in states[module_id]:
                    train_state = states[module_id][key]
                    if hasattr(train_state, "replace"):
                        states[module_id][key] = train_state.replace(opt_state=opt_state)


# pyright: reportAbstractUsage=information
if TYPE_CHECKING:
    __conf: Any = ...
    JaxLearner(config=__conf)  # still abstract
