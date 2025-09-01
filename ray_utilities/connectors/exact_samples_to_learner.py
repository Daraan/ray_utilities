"""Exact samples connector for precise episode length control in Ray RLlib.

.. warning::
    This module is highly experimental and **not ready for production use**.
    The implementation may never work correctly due to fundamental limitations
    in accessing configuration within the connector framework.

This module provides experimental connectors for controlling the exact number
of samples processed by learners. The connector trims episodes to match specified
sample counts, which can be useful for precise batch size control in training.

.. danger::
    This implementation is marked as deprecated and likely non-functional due to
    configuration access limitations within the connector framework. Use at your own risk.

Key Components:
    - :class:`ExactSamplesConnector`: Episode trimming connector (experimental/deprecated)
    - :func:`learner_connector_with_exact_samples`: Factory function for connector setup

Note:
    These utilities require careful handling of downstream connectors that may
    duplicate or mask observations, such as AddOneTsToEpisodesAndTruncate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray.rllib.connectors.connector_v2 import ConnectorV2
from typing_extensions import Self, deprecated

if TYPE_CHECKING:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.typing import EpisodeType

__all__ = ["ExactSamplesConnector"]

logger = logging.getLogger(__name__)


@deprecated("Likely does not work as no config is available here")
class ExactSamplesConnector(ConnectorV2):
    """Experimental connector for trimming episodes to exact sample counts.

    .. warning::
        This connector is **not ready for production use** and is highly experimental.
        It may never work correctly and should not be used in any real applications.

    This connector attempts to ensure learners receive exactly the specified number
    of samples by trimming episodes as needed. However, it's marked as deprecated
    due to configuration access limitations within the connector framework.

    .. danger::
        This connector is likely non-functional because configuration data is not
        available within the connector call context, making it impossible to
        determine the target sample count.

    The connector would trim episodes by removing timesteps from the end of episodes
    until the total sample count matches the desired exact timesteps.

    Args:
        input_observation_space: Observation space specification (unused).
        input_action_space: Action space specification (unused).

    See Also:
        :func:`learner_connector_with_exact_samples`: Factory function for setup
        :class:`ray.rllib.connectors.connector_v2.ConnectorV2`: Base connector class
    """

    @classmethod
    def creator(
        cls,
        input_observation_space,
        input_action_space,
    ) -> Self:
        """Create an ExactSamplesConnector instance for learner connector setup.

        This class method provides a convenient way to create connector instances
        for use with AlgorithmConfig.training(learner_connector=ExactSamplesConnector.creator).

        Args:
            input_observation_space: Observation space specification.
            input_action_space: Action space specification.

        Returns:
            A new ExactSamplesConnector instance configured with the provided spaces.

        Example:
            >>> config.training(learner_connector=ExactSamplesConnector.creator)
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
        shared_data: Optional[dict] = None,  # noqa: ARG002
        metrics: Optional[MetricsLogger] = None,
        **kwargs,  # noqa: ARG002
    ) -> Any:
        # FIXME: Problem have no access to a config here, rl_module.config is deprecated or not sufficient
        _config_deprecated = rl_module.config
        _config_not_available = metrics.peek("config")  # config not yet logged.
        logger.debug("ExactSamplesConnector called with batch")
        total_samples = sum(len(sae) for sae in episodes)
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
    """Create a learner connector that trims episodes to exact sample counts.

    This factory function creates an ExactSamplesConnector for use in learner
    connector pipelines. The connector attempts to ensure precise sample counts
    by trimming episodes as needed.

    Warning:
        The AddOneTsToEpisodesAndTruncate connector that may be added afterwards
        duplicates (but masks) the last observation. This effect needs to be
        handled separately and may interfere with exact sample counting.

    Args:
        input_observation_space: Observation space for the connector.
        input_action_space: Action space for the connector.

    Returns:
        A configured ExactSamplesConnector instance.

    Note:
        This connector is deprecated and likely non-functional due to configuration
        access limitations.

    See Also:
        :class:`ExactSamplesConnector`: The underlying connector implementation
    """
    return ExactSamplesConnector(input_observation_space=input_observation_space, input_action_space=input_action_space)
