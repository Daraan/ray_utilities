"""DQN Torch Learner with gradient accumulation support.

This module provides a DQN learner that accumulates gradients over multiple
batches before performing a parameter update. This is useful for training
with effectively larger batch sizes than what fits in memory.

Note:
    Experimental - edge cases not fully tested yet.
"""

from __future__ import annotations

from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner

from ray_utilities.learners.torch_learner_with_gradient_accumulation_base import TorchLearnerWithGradientAccumulationBase  # noqa: E501


class DQNTorchLearnerWithGradientAccumulation(TorchLearnerWithGradientAccumulationBase, DQNTorchLearner):
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
