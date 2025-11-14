"""NOTE: Experimental, edge cases not tested yet."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Collection

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
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


class PPOTorchLearnerWithGradientAccumulation(PPOTorchLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_count = 0
        self._gradient_updates = 0
        self._last_gradient_update_step: int | None = None
        self._params: dict[ParamRef, torch.Tensor]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._accumulated_time_steps: int = 0
        # TODO: Test checkpoint loading

    # TODO:
    # [ ] What about last batches if not divisible by accumulate_gradients_every? Reset, keep until next batch?
    # [ ] Add scaling: mean (sum divided by accumulate_gradients_every) or just sum
    @override(PPOTorchLearner)
    def compute_gradients(self, loss_per_module: dict[ModuleID, TensorType], **kwargs) -> ParamDict:  # noqa: ARG002
        self._step_count += 1
        accumulate_gradients_every = self.config.learner_config_dict["accumulate_gradients_every"]
        update_gradients_this_step = self._step_count % accumulate_gradients_every == 0

        # Only zero gradients at the start of a new accumulation cycle
        if (self._step_count - 1) % accumulate_gradients_every == 0:
            for optim in self._optimizer_parameters:
                optim.zero_grad(set_to_none=True)

        if self._grad_scalers is not None:
            total_loss = sum(self._grad_scalers[mid].scale(loss) for mid, loss in loss_per_module.items())
        else:
            total_loss = sum(loss_per_module.values())  # pyright: ignore

        # If we don't have any loss computations, `sum` returns 0. Type will not be a pure int in this case
        if isinstance(total_loss, int):
            assert total_loss == 0
            return {}

        total_loss.backward()  # pyright: ignore[reportAttributeAccessIssue]

        if update_gradients_this_step:
            self._gradient_updates += 1
            self._last_gradient_update_step = self._step_count
            _logger.debug("Updating gradients for step %s", self._step_count)

            # Only scale and copy gradients when actually applying them
            if accumulate_gradients_every != 1:
                # In-place division to avoid creating new tensors
                for p in self._params.values():
                    if p.grad is not None:
                        p.grad.div_(accumulate_gradients_every)

            # Return existing gradients
            return {pid: p.grad for pid, p in self._params.items()}

        _logger.debug(
            "Skipping gradient update for step %s, accumulating gradients",
            self._step_count,
        )
        return {}

    def apply_gradients(self, gradients_dict: ParamDict) -> None:
        if self._step_count % self.config.learner_config_dict["accumulate_gradients_every"] == 0:
            # Calls optimizer.step(), scaler.step() and update if applicable
            super().apply_gradients(gradients_dict)

    @override(PPOTorchLearner)
    def after_gradient_based_update(self, *, timesteps: dict[str, Any]) -> None:
        if self._step_count % self.config.learner_config_dict["accumulate_gradients_every"] == 0:
            super().after_gradient_based_update(
                timesteps={**timesteps, NUM_ENV_STEPS_SAMPLED_LIFETIME: self._accumulated_time_steps}
            )
            self._accumulated_time_steps = 0
        # otherwise could call it with 0 timesteps

    def update(self, *args, timesteps: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Cache the episode data reference to avoid repeated lookups
        training_data = kwargs["training_data"]
        episode_steps = sum(map(len, training_data.episodes))
        timesteps[NUM_ENV_STEPS_SAMPLED_LIFETIME] = episode_steps
        self._accumulated_time_steps += episode_steps
        return super().update(*args, timesteps=timesteps, **kwargs)

    def get_state(
        self,
        components: str | Collection[str] | None = None,
        *,
        not_components: str | Collection[str] | None = None,
        **kwargs,
    ):
        state_dict = super().get_state(components, not_components=not_components, **kwargs)
        state_dict.update(
            {
                "step_count": self._step_count,
                "gradient_updates": self._gradient_updates,
                "last_gradient_update_step": self._last_gradient_update_step,
                "accumulated_time_steps": self._accumulated_time_steps,
            }
        )
        return state_dict

    def set_state(self, state) -> None:
        super().set_state(state)
        self._step_count = state.get("step_count", 0)
        self._gradient_updates = state.get("gradient_updates", 0)
        self._last_gradient_update_step = state.get("last_gradient_update_step", None)
        self._accumulated_time_steps = state.get("accumulated_time_steps", 0)
