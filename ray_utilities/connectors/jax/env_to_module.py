"""
env_to_module_connector: ((:class:`EnvType`) -> (:class:`ConnectorV2` | List[:class:`ConnectorV2`])) | None = NotProvided

The default env-to-module connector pipeline is::

    [
        [0 or more user defined :class:`ConnectorV2` pieces],
        :class:`AddObservationsFromEpisodesToBatch`,
        :class:`AddTimeDimToBatchAndZeroPad`,
        :class:`AddStatesFromEpisodesToBatch`,
        :class:`AgentToModuleMapping`,  # only in multi-agent setups!
        :class:`BatchIndividualItems`,
        :class:`NumpyToTensor`,
    ]


The default Learner connector pipeline is::

    [
        [0 or more user defined :class:`ConnectorV2` pieces],
        :class:`AddObservationsFromEpisodesToBatch`,
        :class:`AddColumnsFromEpisodesToTrainBatch`,
        :class:`AddTimeDimToBatchAndZeroPad`,
        :class:`AddStatesFromEpisodesToBatch`,
        :class:`AgentToModuleMapping`,  # only in multi-agent setups!
        :class:`BatchIndividualItems`,
        :class:`NumpyToTensor`,
    ]

"""

# ruff: noqa: ARG001

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from ray.rllib.connectors.env_to_module import (
    AddObservationsFromEpisodesToBatch,
    AddStatesFromEpisodesToBatch,
    AgentToModuleMapping,
    BatchIndividualItems,
    # EnvToModulePipeline,
    # NumpyToTensor,
)

from ray_utilities.connectors.debug_connector import DebugConnector

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.connectors.connector_v2 import ConnectorV2

    from ray_utilities.typing import EnvType

logger = logging.getLogger(__name__)


def _default_env_to_module_without_numpy(
    env: EnvType,
    spaces=None,
    device=None,
    *,
    algo: AlgorithmConfig,
    debug=False,
) -> list[ConnectorV2]:
    """Default pipleine without NumpyToTensor conversion.

    [
        [0 or more user defined ConnectorV2 pieces],
        AddObservationsFromEpisodesToBatch,
        AddTimeDimToBatchAndZeroPad,
        AddStatesFromEpisodesToBatch,
        AgentToModuleMapping,  # XX removed only in multi-agent setups!
        BatchIndividualItems,  # no multi agent support XX removed
        NumpyToTensor, # XX removed
    ]
    """
    pipeline = []
    if debug:
        pipeline.append(DebugConnector(name="EnvToModuleStart"))
    # Append OBS handling.
    pipeline.append(AddObservationsFromEpisodesToBatch())  # <-- extracts episodes obs to batch
    # Append time-rank handler.
    try:
        from ray.rllib.connectors.env_to_module import AddTimeDimToBatchAndZeroPad  # noqa: PLC0415
    except ImportError:
        logger.error(
            "AddTimeDimToBatchAndZeroPad not found on current ray version. This might lead to a broken pipeline"
        )
    else:
        pipeline.append(AddTimeDimToBatchAndZeroPad())
    # Append STATE_IN/STATE_OUT handler.
    pipeline.append(AddStatesFromEpisodesToBatch())
    # If multi-agent -> Map from AgentID-based data to ModuleID based data.
    if algo.is_multi_agent:
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec  # noqa: PLC0415

        pipeline.append(
            AgentToModuleMapping(
                rl_module_specs=(
                    algo.rl_module_spec.rl_module_specs
                    if isinstance(algo.rl_module_spec, MultiRLModuleSpec)
                    else set(algo.policies)  # pyright: ignore[reportArgumentType] # old api
                ),
                agent_to_module_mapping_fn=algo.policy_mapping_fn,
            )
        )
    # Batch all data.
    pipeline.append(BatchIndividualItems(multi_agent=algo.is_multi_agent))
    # Convert to Tensors.
    # pipeline.append(NumpyToTensor(device=device))
    if debug:
        pipeline.append(DebugConnector(name="EnvToModuleEnd"))
    return pipeline


def make_env_to_module_without_numpy(algo, *, debug=False) -> partial[list[ConnectorV2]]:
    """Make env_to_module without NumpyToTensor conversion."""
    return partial(_default_env_to_module_without_numpy, algo=algo, debug=debug)
