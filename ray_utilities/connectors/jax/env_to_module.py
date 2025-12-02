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
from typing import TYPE_CHECKING, Callable, Optional

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
    from ray.rllib.core.rl_module import RLModuleSpec
    from ray.rllib.policy.policy import PolicySpec  # old API
    from ray.rllib.utils.typing import AgentID, PolicyID, EpisodeType

    from ray_utilities.typing import EnvType

logger = logging.getLogger(__name__)


def _default_env_to_module_without_numpy(
    env: EnvType,
    spaces=None,
    device=None,
    *,
    is_multi_agent: bool = False,
    rl_module_spec: Optional[RLModuleSpec] = None,
    multi_agent_policies: Optional[dict[str, PolicySpec]] = None,
    policy_mapping_fn: Optional[Callable[[AgentID, "EpisodeType"], PolicyID]] = None,
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
    if is_multi_agent:
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec  # noqa: PLC0415

        pipeline.append(
            AgentToModuleMapping(
                rl_module_specs=(
                    rl_module_spec.rl_module_specs
                    if isinstance(rl_module_spec, MultiRLModuleSpec)
                    else set(multi_agent_policies)  # pyright: ignore[reportArgumentType] # old api
                ),
                agent_to_module_mapping_fn=policy_mapping_fn,
            )
        )
    # Batch all data.
    pipeline.append(BatchIndividualItems(multi_agent=is_multi_agent))
    # Convert to Tensors.
    # pipeline.append(NumpyToTensor(device=device))
    if debug:
        pipeline.append(DebugConnector(name="EnvToModuleEnd"))
    return pipeline


class EnvToModuleWithoutNumpyConnector:
    """Factory class to create a connector for env_to_module without NumpyToTensor conversion."""

    def __init__(
        self,
        algo: AlgorithmConfig,
        *,
        debug=False,
    ):
        self.is_multi_agent = algo.is_multi_agent
        self.enable_env_runner_and_connector_v2 = algo.enable_env_runner_and_connector_v2
        if self.is_multi_agent:
            assert algo.rl_module_spec
            self.rl_module_spec = algo.rl_module_spec
        else:
            self.rl_module_spec = None
        self.policies = algo.policies
        self.policy_mapping_fn = algo.policy_mapping_fn
        self.debug = debug

    def __call__(self, env, spaces=None, device=None, /) -> list["ConnectorV2"]:  # noqa: ARG002
        if not self.is_multi_agent:
            return _default_env_to_module_without_numpy(env, spaces, device, is_multi_agent=False, debug=self.debug)
        if self.enable_env_runner_and_connector_v2:
            assert self.rl_module_spec is not None
        else:
            assert self.policy_mapping_fn is not None
        return _default_env_to_module_without_numpy(
            env,
            spaces,
            device,
            is_multi_agent=True,
            rl_module_spec=self.rl_module_spec,  # pyright: ignore[reportArgumentType]
            multi_agent_policies=self.policies,
            policy_mapping_fn=self.policy_mapping_fn,  # pyright: ignore[reportArgumentType]
            debug=self.debug,
        )

    def __eq__(self, other):
        if not isinstance(other, EnvToModuleWithoutNumpyConnector):
            return False
        return bool(
            self.is_multi_agent == other.is_multi_agent
            and self.enable_env_runner_and_connector_v2 == other.enable_env_runner_and_connector_v2
            and self.rl_module_spec == other.rl_module_spec
            and self.policies == other.policies
            and self.policy_mapping_fn == other.policy_mapping_fn
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.is_multi_agent,
                self.enable_env_runner_and_connector_v2,
                self.rl_module_spec,
                self.policies,
                self.policy_mapping_fn,
            )
        )
