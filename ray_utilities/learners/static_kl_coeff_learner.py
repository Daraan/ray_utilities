from __future__ import annotations

# As config.training allows None values for several parameters
# pyright: reportOperatorIssue=none
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from ray.rllib.algorithms.ppo.ppo import LEARNER_RESULTS_CURR_KL_COEFF_KEY, PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from ray_utilities.learners.torch_learner_with_gradient_accumulation_base import (
    TorchLearnerWithGradientAccumulationBase,
)

if TYPE_CHECKING:
    from ray.rllib.utils.typing import ModuleID

logger = logging.getLogger(__name__)


class StaticKLCoeffPPOTorchLearner(PPOTorchLearner):
    def _update_module_kl_coeff(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,  # noqa: ARG002
        kl_loss: float,
    ) -> None:
        if np.isnan(kl_loss):
            logger.warning(
                f"KL divergence for Module {module_id} is non-finite, this "  # noqa: G004
                "will likely destabilize your model and the training "
                "process. Action(s) in a specific state have near-zero "
                "probability. This can happen naturally in deterministic "
                "environments where the optimal policy has zero mass for a "
                "specific action. To fix this issue, consider setting "
                "`kl_coeff` to 0.0 or increasing `entropy_coeff` in your "
                "config."
            )

        # No update of the KL coefficient.
        curr_var: torch.Tensor = self.curr_kl_coeffs_per_module[module_id]  # pyright: ignore[reportAssignmentType]

        assert torch.isclose(curr_var, torch.tensor(config.kl_coeff)), (
            f"StaticKLCoeffPPOTorchLearner should have constant kl_coeff! Got {curr_var} vs. {config.kl_coeff}"
        )
        # Log the updated KL-coeff value.

        self.metrics.log_value(
            (module_id, LEARNER_RESULTS_CURR_KL_COEFF_KEY),
            curr_var.item(),
            window=1,
        )


class StaticKLCoeffPPOTorchLearnerWithGradAccum(
    TorchLearnerWithGradientAccumulationBase, StaticKLCoeffPPOTorchLearner
): ...
