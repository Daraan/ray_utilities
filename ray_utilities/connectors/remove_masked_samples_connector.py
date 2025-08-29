"""Ray RLlib connector for removing masked samples from learner batches.

This module provides the :class:`RemoveMaskedSamplesConnector` which removes
masked samples that are added by Ray RLlib's AddOneTsToEpisodesAndTruncate
connector but are not needed for loss calculation.

Key Components:
    - :class:`RemoveMaskedSamplesConnector`: Connector for masked sample removal
    - Integration with Ray RLlib's learner connector pipeline
    - Metrics tracking for environment steps passed to learners

This connector is particularly useful when combined with exact sampling callbacks
to ensure batches contain precisely the intended number of samples.
"""
# ruff: noqa: ARG002

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.utils.metrics import ALL_MODULES  # pyright: ignore[reportPrivateImportUsage]
from ray.rllib.utils.postprocessing.episodes import remove_last_ts_from_episodes_and_restore_truncateds

from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME

if TYPE_CHECKING:
    import numpy as np
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.typing import EpisodeType, ModuleID

_logger = logging.getLogger(__name__)


class RemoveMaskedSamplesConnector(ConnectorV2):
    """Connector for removing masked samples from learner batches in Ray RLlib.

    This connector removes samples that are added by the AddOneTsToEpisodesAndTruncate
    connector but are masked out and not needed for loss calculation. It's designed
    to be placed at the end of a learner connector pipeline, after GAE computation.

    When combined with exact sampling callbacks, this connector ensures that batches
    contain precisely the intended number of samples by removing masked timesteps
    that would otherwise inflate batch sizes unnecessarily.

    The connector processes the loss mask to identify and remove masked samples from
    all relevant batch keys, while maintaining proper metrics tracking for the
    actual number of environment steps passed to learners.

    Features:
        - Removes masked samples using loss mask information
        - Updates episode data to maintain consistency
        - Tracks environment step metrics for monitoring
        - Provides warnings when loss masks are missing

    Warning:
        Since custom learner_connector arguments to AlgorithmConfig only prepend
        connectors, this connector needs to be added differently, such as by using
        the RemoveMaskedSamplesLearner wrapper.

    Example:
        >>> # Used typically through RemoveMaskedSamplesLearner
        >>> config.training(learner_class=RemoveMaskedSamplesLearner)

    Note:
        This connector modifies episodes using mean values for consistency and should
        not be used to track the exact number of episodes passed, as the batch is
        the primary data structure used after processing.

    See Also:
        :class:`ray.rllib.connectors.connector_v2.ConnectorV2`: Base connector class
        :class:`ray_utilities.learners.remove_masked_samples_learner.RemoveMaskedSamplesLearner`: Learner wrapper
    """

    _logged_warning = False

    @staticmethod
    def _log_and_increase_module_steps(
        metrics: Optional[MetricsLogger],
        module_id: ModuleID,
        module_batch: dict[str, Any],
        num_steps: int,
    ) -> int:
        module_steps = len(module_batch[Columns.OBS])
        if metrics:
            metrics.log_value(
                (module_id, NUM_ENV_STEPS_PASSED_TO_LEARNER), module_steps, reduce="sum", clear_on_reduce=True
            )
            metrics.log_value((module_id, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), module_steps, reduce="sum")
        return num_steps + module_steps

    def __call__(
        self,
        *,
        batch: dict[ModuleID, dict[str, Any]],
        episodes: list[EpisodeType],
        metrics: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> Any:
        # Fix batch length by removing samples that are masked out.
        num_steps = 0
        for module_id, module_batch in batch.items():
            loss_mask: np.ndarray | None = module_batch.get(Columns.LOSS_MASK, None)
            if loss_mask is None:
                if not self._logged_warning:
                    _logger.warning("No loss_mask found in batch, skipping removal of masked samples.")
                    self._logged_warning = True
                num_steps = self._log_and_increase_module_steps(metrics, module_id, module_batch, num_steps)
                continue
            for key in module_batch:
                if key == Columns.LOSS_MASK:
                    continue
                module_batch[key] = module_batch[key][loss_mask]
            module_batch[Columns.LOSS_MASK] = loss_mask[loss_mask]
            num_steps = self._log_and_increase_module_steps(metrics, module_id, module_batch, num_steps)
        # Remove from episodes as well - for correct learner_connector_sum_episodes_length_out logging
        # Note: This uses a mean value; do not use to keep track of episodes passed!
        # original truncated information is unknown; but likely not needed afterwards as only batch is used
        remove_last_ts_from_episodes_and_restore_truncateds(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False),  # pyright: ignore[reportArgumentType] # ray function should have Iterable not list
            orig_truncateds=[False]
            * len(episodes),  # TODO: check in later training if potentially batch.truncated can be used here
        )
        if metrics:
            metrics.log_value(
                (ALL_MODULES, NUM_ENV_STEPS_PASSED_TO_LEARNER), num_steps, reduce="sum", clear_on_reduce=True
            )
            metrics.log_value((ALL_MODULES, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), num_steps, reduce="sum")
        return batch
