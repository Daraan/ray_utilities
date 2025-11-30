"""NOTE: Experimental, edge cases not tested yet."""

from __future__ import annotations

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray_utilities.learners.torch_learner_with_gradient_accumulation_base import (
    TorchLearnerWithGradientAccumulationBase,
)


class PPOTorchLearnerWithGradientAccumulation(TorchLearnerWithGradientAccumulationBase, PPOTorchLearner):
    """
    A PPO learner for PyTorch with gradient accumulation support.

    This class combines :class:`TorchLearnerWithGradientAccumulationBase` and
    :class:`PPOTorchLearner` to extend the standard PPOTorchLearner to accumulate gradients
    before applying them. This allows training with effectively larger batch
    sizes than what fit in (GPU) memory.
    """
