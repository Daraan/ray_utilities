"""module_to_env_connector: ((EnvType, RLModule) -> (ConnectorV2 | List[ConnectorV2])) | None = NotProvided,"""

from __future__ import annotations

# Prepend: Anything that has to do with plain data processing (not
# particularly with the actions).
from typing import TYPE_CHECKING

from ray.rllib.connectors.module_to_env import (
    ListifyDataForVectorEnv,
    ModuleToAgentUnmapping,
    NormalizeAndClipActions,
    RemoveSingleTsTimeRankFromBatch,
    UnBatchToIndividualItems,
)

from ray_utilities.connectors.debug_connector import DebugConnector
from ray_utilities.connectors.jax.get_actions import GetActionsJaxDistr

if TYPE_CHECKING:
    import chex
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.connectors.connector_v2 import ConnectorV2
    from ray.rllib.core.rl_module.rl_module import RLModule

    from ray_utilities.typing import EnvType


__all__ = [
    "make_jax_module_to_env_connector",
]


def _jax_module_to_env_connector(
    env: EnvType,  # noqa: ARG001
    rl_module: RLModule | None = None,  # noqa: ARG001
    *,
    key: chex.PRNGKey,
    is_multi_agent: bool,
    debug=False,
    normalize_actions: bool,
    clip_actions: bool,
    **kwargs,  # noqa: ARG001
) -> list["ConnectorV2"]:
    # NOTE: rl_module might not be used

    # Remove extra time-rank, if applicable.
    pipeline = []
    pipeline.insert(0, RemoveSingleTsTimeRankFromBatch())

    # If multi-agent -> Map from ModuleID-based data to AgentID based data.
    if is_multi_agent:
        pipeline.insert(0, ModuleToAgentUnmapping())

    # Unbatch all data.
    pipeline.insert(0, UnBatchToIndividualItems())

    # Convert to numpy.
    # TODO: Maybe a Jax to numpy
    # pipeline.insert(0, TensorToNumpy())

    # Sample actions from ACTION_DIST_INPUTS (if ACTIONS not present).
    pipeline.insert(0, GetActionsJaxDistr(key=key))
    if debug and False:
        pipeline.insert(0, DebugConnector(name="ModuleToEnvStart"))

    # Append: Anything that has to do with action sampling.
    # Unsquash/clip actions based on config and action space.
    pipeline.append(
        NormalizeAndClipActions(
            normalize_actions=normalize_actions,
            clip_actions=clip_actions,
        )
    )
    # Listify data from ConnectorV2-data format to normal lists that we can
    # index into by env vector index. These lists contain individual items
    # for single-agent and multi-agent dicts for multi-agent.
    pipeline.append(ListifyDataForVectorEnv())
    if debug:
        pipeline.append(DebugConnector(name="ModuleToEnvEnd"))
    return pipeline


class MakeJaxModuleToEnvConnector:
    """
    Same as the default pipeline, but with JAX compatible.

    This especially removes the TensorToNumpy connector.
    Optionally can add a DebugConnector at the start and end of the pipeline.

    Attention:
        This binds algo.is_multi_agent, algo.clip_actions, algo.normalize_actions, and
        algo.policies at creation time. If these change during training or by definitions after
        calling this function, the changes will not be reflected in the created connector!

        You should use this function *after* calling config.environment(...)
    """

    def __init__(self, algo: AlgorithmConfig, *, key: "chex.PRNGKey", debug=False):
        self.key = key
        self.debug = debug
        self.is_multi_agent = algo.is_multi_agent
        self.clip_actions = algo.clip_actions
        self.normalize_actions = algo.normalize_actions

    def __call__(self, env, rl_module) -> "ConnectorV2" | list["ConnectorV2"]:
        return _jax_module_to_env_connector(
            env,
            rl_module,
            is_multi_agent=self.is_multi_agent,
            clip_actions=bool(self.clip_actions),
            normalize_actions=bool(self.normalize_actions),
            key=self.key,
            debug=self.debug,
        )

    def __eq__(self, other):
        if not isinstance(other, MakeJaxModuleToEnvConnector):
            return False
        return bool(
            self.is_multi_agent == other.is_multi_agent
            and self.clip_actions == other.clip_actions
            and self.normalize_actions == other.normalize_actions
            and all(self.key == other.key)
        )

    def __hash__(self) -> int:
        return hash((type(self), self.is_multi_agent, self.clip_actions, self.normalize_actions, tuple(self.key)))
