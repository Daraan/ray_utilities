"""
DebugConnector (ModuleToEnvStart):
{batch={'action_dist_inputs': Array([[-0.04725214, -0.00324919]], dtype=float32)}},
{episodes=[SAEps(len=0 done=False R=0 id_=de4fd9cc76ef40aa9259d434c433dabd)]},
{explore=True},
{shared_data={'memorized_map_structure': ['de4fd9cc76ef40aa9259d434c433dabd']}},
{metrics=<ray.rllib.utils.metrics.metrics_logger.MetricsLogger object at 0x7fe0ff2b9990>}

--- DebugConnector (ModuleToEnvEnd):
{batch={'action_dist_inputs': [Array([-0.04725214, -0.00324919], dtype=float32)],
'actions': array([1], dtype=int32),
'RNG': [Array(3089211859, dtype=uint32)],  # Added in GetAction.
# FIXME: Should do this in module or a special connector
# e.g. update a key on module and pass another key forward
'action_logp': [Array(-0.67138773, dtype=float32)],
'actions_for_env': array([1], dtype=int32)}},
{episodes=[SAEps(len=0 done=False R=0 id_=de4fd9cc76ef40aa9259d434c433dabd)]},
{explore=True}, {shared_data={'memorized_map_structure': ['de4fd9cc76ef40aa9259d434c433dabd']}},
{metrics=<ray.rllib.utils.metrics.metrics_logger.MetricsLogger object at 0x7fe0ff2b9990>}

DebugConnector (EnvToModuleStart): {batch={}}, {episodes=[SAEps(len=1 done=False R=1.0 id_=de4fd9cc76ef40aa9259d434c433dabd)]}, {explore=True}, {shared_data={'memorized_map_structure': ['de4fd9cc76ef40aa9259d434c433dabd']}}, {metrics=<ray.rllib.utils.metrics.metrics_logger.MetricsLogger object at 0x7fe0ff2b9990>}

--- LearnerConnectorStart ---
DebugConnector (LearnerConnectorStart): {batch={}},
{episodes=[SAEps(len=18 done=True R=18.0 id_=de4fd9cc76ef40aa9259d434c433dabd), SAEps(len=10 done=True R=10.0 id_=1f3b3a234df844a7afd8b1efed6e9f19), SAEps(len=8 done=False R=8.0 id_=ef9e18950d8d4bca90e7cad820989117)]},
{explore=False}, {shared_data={}}, {metrics=<ray.rllib.utils.metrics.metrics_logger.MetricsLogger object at 0x7fe10bb26aa0>}

# Cnvert Episodes -> batch

DebugConnector (LearnerConnectorEnd): {batch={'default_policy':
    {'obs': array([[ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ],
    [ 0.02727336,  0.18847767,  0.03625453, -0.26141977],
    ...
    [-0.01050504,  0.58184975, -0.04927642, -0.8679848 ]], dtype=float32),
    'actions': array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0], dtype=int32),
    'rewards': array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1.]),
    'terminateds': array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,  True,
        False, False, False, False, False, False, False, False, False,
        True, False, False, False, False, False, False, False, False]),
    'truncateds': array([False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False]),
    'action_dist_inputs': array([[-0.04725214, -0.00324919],
        [ 0.08704305,  0.12770459],
        ...
        [ 0.03847424,  0.03154632]], dtype=float32),
    'RNG': array([3089211859,  676310478, 4197501953, 2886768726, 1408500994,
        ...
        2570299311], dtype=uint32),
'action_logp': array([-0.67138773, -0.71368456, ...
-0.6896892 ], dtype=float32),
    'weights_seq_no': array([0, ... 0])}}},
    {episodes=[SAEps(len=18 done=True R=18.0 id_=de4fd9cc76ef40aa9259d434c433dabd_0), SAEps(len=10 done=True R=10.0 id_=1f3b3a234df844a7afd8b1efed6e9f19_1), SAEps(len=8 done=False R=8.0 id_=ef9e18950d8d4bca90e7cad820989117_2)]}, {explore=False}, {shared_data={'memorized_map_structure': ['de4fd9cc76ef40aa9259d434c433dabd_0', '1f3b3a234df844a7afd8b1efed6e9f19_1', 'ef9e18950d8d4bca90e7cad820989117_2']}}, {metrics=<ray.rllib.utils.metrics.metrics_logger.MetricsLogger object at 0x7fe10bb26aa0>}
"""

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
    """
    Connector to be added in the pipeline for debugging purposes
    by adding a `breakpoint()`.
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
