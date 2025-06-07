# ruff: noqa: ARG002

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.postprocessing.episodes import remove_last_ts_from_episodes_and_restore_truncateds

from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME

if TYPE_CHECKING:
    import numpy as np
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.typing import EpisodeType, ModuleID

_logger = logging.getLogger(__name__)


class RemoveMaskedSamplesConnector(ConnectorV2):
    """
    A connector to be put at the end of a learner connector pipeline (after GAE) to remove samples
    added by the `ray.rllib.connectors.learner.add_one_ts_to_episodes_and_truncate.
    AddOneTsToEpisodesAndTruncate` connector piece that are masked and useless for the loss calculation.

    When combined with the `exact_sampling_callback` this assures that the batch really has the
    exact number of samples.

    Attention:
        As a custom learner_connector argument to the AlgorithmConfig will only prepend
        connectors this connector needs to be added in a different way, e.g. by using the
        `RemoveMaskedSamplesLearner`.
    """

    _logged_warning = False

    @staticmethod
    def _log_and_increase_module_steps(
        metrics: MetricsLogger,
        module_id: ModuleID,
        module_batch: dict[str, Any],
        num_steps: int,
    ) -> int:
        module_steps = len(module_batch[Columns.OBS])
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
        metrics: MetricsLogger,
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
        metrics.log_value((ALL_MODULES, NUM_ENV_STEPS_PASSED_TO_LEARNER), num_steps, reduce="sum", clear_on_reduce=True)
        metrics.log_value((ALL_MODULES, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), num_steps, reduce="sum")
        return batch
