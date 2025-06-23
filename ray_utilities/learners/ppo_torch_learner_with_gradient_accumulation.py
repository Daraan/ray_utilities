"""NOTE: Experimental, not tested yet."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
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


class PPOTorchLearnerWithGradientAccumulation(PPOTorchLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_count = 0
        self._params: dict[ParamRef, torch.Tensor]  # pyright: ignore[reportIncompatibleVariableOverride]
        # TODO: Test checkpoint loading

    # TODO:
    # [ ] What about last batches if not divisible by accumulate_gradients_every? Reset, keep until next batch?
    # [ ] Add scaling: mean (sum divided by accumulate_gradients_every) or just sum
    @override(PPOTorchLearner)
    def compute_gradients(self, loss_per_module: dict[ModuleID, TensorType], **kwargs) -> ParamDict:  # noqa: ARG002
        self._step_count += 1
        if (
            update_gradients_this_step := self._step_count
            % (accumulate_gradients_every := self.config.learner_config_dict["accumulate_gradients_every"])
            == 0
        ):
            _logger.debug("Updating gradients for step %s", self._step_count)
            for optim in self._optimizer_parameters:
                # `set_to_none=True` is a faster way to zero out the gradients.
                optim.zero_grad(set_to_none=True)
        else:
            _logger.debug(
                "Skipping gradient update for step %s, accumulating gradients",
                self._step_count,
            )

        if self._grad_scalers is not None:
            total_loss = sum(self._grad_scalers[mid].scale(loss) for mid, loss in loss_per_module.items())
        else:
            total_loss = sum(loss_per_module.values())  # pyright: ignore

        # If we don't have any loss computations, `sum` returns 0.
        if isinstance(total_loss, int):
            assert total_loss == 0
            return {}

        total_loss.backward()
        if update_gradients_this_step:
            # PPO loss is a mean, should divide by the number of accumulated batches.
            # TODO: When changing accumulate_gradients_every dynamically to 1 and accumulation did not happen yet,
            #      then not dividing by the amount of accumulated batches is not correct.
            if accumulate_gradients_every != 1:
                grads = {
                    pid: p.grad / accumulate_gradients_every if p.grad is not None else p.grad
                    for pid, p in self._params.items()
                }
            else:
                grads = {pid: p.grad for pid, p in self._params.items()}
        else:
            return {}

        return grads  # pyright: ignore[reportReturnType]  # contains None

    # @override(TorchLearner)
    def apply_gradients(self, gradients_dict: ParamDict) -> None:
        if self._step_count % self.config.learner_config_dict["accumulate_gradients_every"] == 0:
            # Calls optimizer.step(), scaler.step() and update if applicable
            super().apply_gradients(gradients_dict)

    def after_gradient_based_update(self, *, timesteps: dict[str, Any]) -> None:
        # TODO: Discuss: Scheduler is updated based on non-exact sampled timesteps.
        # if self._step_count % self.config.learner_config_dict["accumulate_gradients_every"] == 0:
        # Updates KL coefficients, and LR Schedulers; if doing this every step batch_size
        # accumulation is not exact to multipling batch size - but should be minor influence if any.
        super().after_gradient_based_update(timesteps=timesteps)
