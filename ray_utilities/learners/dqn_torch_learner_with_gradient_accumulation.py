"""DQN Torch Learner with gradient accumulation support.

This module provides a DQN learner that accumulates gradients over multiple
batches before performing a parameter update. This is useful for training
with effectively larger batch sizes than what fits in memory.

Note:
    Experimental - edge cases not fully tested yet.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Collection

from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ParamDict

if TYPE_CHECKING:
    import torch
    from ray.rllib.utils.typing import (
        ModuleID,
        ParamDict,
        ParamRef,
        TensorType,
    )

_logger = logging.getLogger(__name__)


class DQNTorchLearnerWithGradientAccumulation(DQNTorchLearner):
    """DQN Torch Learner with gradient accumulation.

    This learner extends the standard DQNTorchLearner to accumulate gradients
    over multiple minibatches before applying them. This allows training with
    effectively larger batch sizes than what fits in GPU memory.

    The gradient accumulation is controlled by the `accumulate_gradients_every`
    parameter in the learner_config_dict.

    Features:
        - Accumulate gradients over N batches
        - Automatic gradient scaling by accumulation factor
        - State tracking for checkpointing and restoration
        - Compatible with DQN's replay buffer and target network updates

    Note:
        Unlike PPO which samples synchronously, DQN samples from a replay buffer,
        so gradient accumulation doesn't change the experience distribution but
        allows larger effective batch sizes for more stable Q-value estimates.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the DQN learner with gradient accumulation tracking."""
        super().__init__(*args, **kwargs)
        self._step_count = 0
        self._gradient_updates = 0
        self._last_gradient_update_step: int | None = None
        self._params: dict[ParamRef, torch.Tensor]  # pyright: ignore[reportIncompatibleVariableOverride]

    @override(DQNTorchLearner)
    def compute_gradients(self, loss_per_module: dict[ModuleID, TensorType], **kwargs) -> ParamDict:  # noqa: ARG002
        """Compute gradients with accumulation support.

        Args:
            loss_per_module: Dictionary mapping module IDs to loss tensors
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of gradients if this is an update step, empty dict otherwise
        """
        self._step_count += 1
        accumulate_gradients_every = self.config.learner_config_dict["accumulate_gradients_every"]
        update_gradients_this_step = self._step_count % accumulate_gradients_every == 0

        # Zero gradients at the start of each accumulation cycle
        if (self._step_count - 1) % accumulate_gradients_every == 0:
            for optim in self._optimizer_parameters:
                # `set_to_none=True` is a faster way to zero out the gradients.
                optim.zero_grad(set_to_none=True)

        # Compute total loss (with gradient scaling if applicable)
        if self._grad_scalers is not None:
            total_loss = sum(self._grad_scalers[mid].scale(loss) for mid, loss in loss_per_module.items())
        else:
            total_loss = sum(loss_per_module.values())  # pyright: ignore

        # If we don't have any loss computations, `sum` returns 0
        if isinstance(total_loss, int):
            assert total_loss == 0
            return {}

        # Backward pass - gradients accumulate in the tensors
        total_loss.backward()

        # Only return gradients when we've accumulated enough batches
        if update_gradients_this_step:
            self._gradient_updates += 1
            self._last_gradient_update_step = self._step_count
            _logger.debug("Updating gradients for step %s", self._step_count)

            # Scale gradients by the accumulation factor
            if accumulate_gradients_every != 1:
                grads = {
                    pid: p.grad / accumulate_gradients_every if p.grad is not None else p.grad
                    for pid, p in self._params.items()
                }
            else:  # No accumulation
                grads = {pid: p.grad for pid, p in self._params.items()}
            return grads  # pyright: ignore[reportReturnType]  # contains None

        _logger.debug(
            "Skipping gradient update for step %s, accumulating gradients",
            self._step_count,
        )
        return {}

    @override(DQNTorchLearner)
    def apply_gradients(self, gradients_dict: ParamDict) -> None:
        """Apply gradients only when accumulation cycle is complete.

        Args:
            gradients_dict: Dictionary of gradients to apply
        """
        if self._step_count % self.config.learner_config_dict["accumulate_gradients_every"] == 0:
            # Calls optimizer.step(), scaler.step() and update if applicable
            super().apply_gradients(gradients_dict)

    @override(DQNTorchLearner)
    def after_gradient_based_update(self, *, timesteps: dict[str, Any]) -> None:
        """Perform post-update operations.

        For DQN, this handles target network updates and learning rate scheduling.
        These are performed after each gradient update (not after each compute_gradients call).

        Args:
            timesteps: Dictionary containing timestep information
        """
        # Always call parent's after_gradient_based_update
        # This handles target network updates which should happen based on
        # training steps, not gradient accumulation steps
        super().after_gradient_based_update(timesteps=timesteps)

    def get_state(
        self,
        components: str | Collection[str] | None = None,
        *,
        not_components: str | Collection[str] | None = None,
        **kwargs,
    ):
        """Get the learner state including gradient accumulation tracking.

        Args:
            components: Components to include in state
            not_components: Components to exclude from state
            **kwargs: Additional arguments

        Returns:
            Dictionary containing learner state
        """
        state_dict = super().get_state(components, not_components=not_components, **kwargs)
        state_dict.update(
            {
                "step_count": self._step_count,
                "gradient_updates": self._gradient_updates,
                "last_gradient_update_step": self._last_gradient_update_step,
            }
        )
        return state_dict

    def set_state(self, state) -> None:
        """Restore learner state including gradient accumulation tracking.

        Args:
            state: Dictionary containing learner state
        """
        super().set_state(state)
        self._step_count = state.get("step_count", 0)
        self._gradient_updates = state.get("gradient_updates", 0)
        self._last_gradient_update_step = state.get("last_gradient_update_step", None)
