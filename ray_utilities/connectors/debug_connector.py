# ruff: noqa: ARG002

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray.rllib.connectors.connector_v2 import ConnectorV2

if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.typing import EpisodeType


class DebugConnector(ConnectorV2):
    """Debug connector for Ray RLlib pipeline inspection and troubleshooting.

    This connector provides debugging capabilities for Ray RLlib's data processing
    pipeline by adding logging, inspection points, and optional breakpoints at
    specific stages of the training workflow. It can be inserted anywhere in the
    connector pipeline to monitor data flow and help with troubleshooting.

    The connector logs detailed information about the data passing through it,
    including batches, episodes, exploration settings, and metrics. It supports
    both debug-level logging for continuous monitoring and warning-level logging
    with detailed dumps for specific debugging sessions.

    Features:
        - Configurable logging levels and message detail
        - Optional breakpoint insertion for interactive debugging
        - Non-intrusive data flow monitoring
        - Named connector instances for pipeline identification

    Args:
        name: Identifier for this connector instance, used in log messages
            to distinguish between multiple debug connectors in the pipeline.
        *args: Additional positional arguments passed to parent :class:`ConnectorV2`.
        **kwargs: Additional keyword arguments passed to parent :class:`ConnectorV2`.

    Example:
        >>> # Add to environment-to-module connector pipeline
        >>> config.env_to_module_connector = lambda obs_space, act_space: [
        ...     DebugConnector(name="env_to_module_debug"),
        ...     # ... other connectors
        ... ]

    Note:
        The debug logging and breakpoint functionality can be controlled by
        modifying the conditional checks in the ``__call__`` method. Set the
        condition to ``True`` to enable detailed logging dumps.

    See Also:
        :class:`ray.rllib.connectors.connector_v2.ConnectorV2`: Base connector class
        :class:`ray.rllib.core.rl_module.rl_module.RLModule`: RL module interface
    """

    def __init__(self, *args, name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger(__name__)
        self._name = name
        self._logger.setLevel(logging.DEBUG)

    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: dict[str, Any],
        episodes: list[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        metrics: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> Any:
        if False:
            self._logger.warning(
                "DebugConnector (%s): \nbatch=%s, \nepisodes=%s, \nexplore=%s, \nshared_data=%s, \nmetrics=%s\n-------",
                self._name,
                batch,
                episodes,
                explore,
                shared_data,
                metrics,
            )
        self._logger.debug("DebugConnector called with batch: %s", batch)
        breakpoint()  # noqa: T100
        return batch


def add_debug_connectors(learner: Learner) -> None:
    """
    Adds DebugConnector to the learner connector pipeline if the config
    has "_debug_connectors" set to True.

    Will clean existing debug connectors (start, end only) if they exist.
    """
    if learner.config.learner_config_dict.get("_debug_connectors", False):
        if not learner._learner_connector:
            # create a learner connector
            learner._learner_connector = learner.config.build_learner_connector(
                input_observation_space=None,
                input_action_space=None,
                device=learner._device,
            )
            learner._learner_connector.append(DebugConnector(name="Learner debug End"))
            return
        # make sure to clean existing debug connectors (start, end only)
        if learner._learner_connector:
            if isinstance(learner._learner_connector.connectors[0], DebugConnector):
                learner._learner_connector.connectors.pop(0)
            if isinstance(learner._learner_connector.connectors[-1], DebugConnector):
                learner._learner_connector.connectors.pop(-1)
        learner._learner_connector.prepend(DebugConnector(name="Learner debug Start"))
        learner._learner_connector.append(DebugConnector(name="Learner debug End"))


if TYPE_CHECKING:
    DebugConnector(name="abc")
