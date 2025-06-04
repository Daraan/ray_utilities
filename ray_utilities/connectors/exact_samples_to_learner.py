from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray.rllib.connectors.connector_v2 import ConnectorV2
from typing_extensions import Self

if TYPE_CHECKING:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.spaces.space_utils import BatchedNdArray
    from ray.rllib.utils.typing import AgentID, EpisodeType, ModuleID, StateDict

__all__ = ["ExactSamplesConnector"]

logger = logging.getLogger(__name__)


class ExactSamplesConnector(ConnectorV2):
    @classmethod
    def creator(
        cls,
        input_observation_space,
        input_action_space,
    ) -> Self:
        """
        Argument that adds a single custom learner connector to trim the episodes to
        the exact number of samples.

        To be used with AlgorithmConfig.training(learner_connector=ExactSamplesConnector.creator).
        """
        return cls(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
        )

    def __call__(
        self,
        *,
        rl_module: RLModule | MultiRLModule,
        batch: dict[str, Any],
        episodes: list[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        metrics: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> Any:
        rl_module.config
        breakpoint()
        logger.debug("ExactSamplesConnector called with batch")
        total_samples = sum(len(sae) for sae in episodes)
        config = metrics.peek("config")
        # FIXME: Problem have no access to a config here, rl_module.config is deprecated or not sufficient
        exact_timesteps = ...
        if total_samples > exact_timesteps:
            diff = total_samples - exact_timesteps
            for i, sample in enumerate(episodes):
                if not sample.is_done and len(sample) >= diff:
                    episodes[i] = sample[:-diff]
                    break
            else:
                # this is wrong when the last sample is done but very short.
                episodes[-1] = episodes[-1][:-diff]
            total_samples = sum(len(sae) for sae in episodes)

        assert total_samples == exact_timesteps, (
            f"Total samples {total_samples} does not match exact timesteps {exact_timesteps}."
        )
        return batch


def learner_connector_with_exact_samples(
    input_observation_space,
    input_action_space,
) -> ConnectorV2:
    """
    Argument that adds a single custom learner connector to trim the episodes to
    the exact number of samples.

    Attention:
        The AddOneTsToEpisodesAndTruncate connector that is added afterwards duplicates,
        but masks, the last observation. The effect of said Connector needs to be handled,
        separately.
    """
    return ExactSamplesConnector(input_observation_space=input_observation_space, input_action_space=input_action_space)
