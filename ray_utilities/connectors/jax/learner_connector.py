# Option 1 make subclass of Algorithm
# https://docs.ray.io/en/latest/rllib/rllib-offline.html#connector-level

"""
Default Learner connector pipeline for JAX RLlib integration.

This module defines the default connector pipeline for learner processing in Ray RLlib with JAX.
The pipeline is used to preprocess and batch data for RLlib algorithms.

Default learner connector pipeline::

    [
        # 0 or more user defined ConnectorV2 pieces
        AddObservationsFromEpisodesToBatch,
        AddColumnsFromEpisodesToTrainBatch,
        AddTimeDimToBatchAndZeroPad,
        AddStatesFromEpisodesToBatch,
        AgentToModuleMapping,  # only in multi-agent setups!
        BatchIndividualItems,
        NumpyToTensor,
    ]
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

from ray.rllib.connectors.learner import (
    AddColumnsFromEpisodesToTrainBatch,
    AddObservationsFromEpisodesToBatch,
    AddOneTsToEpisodesAndTruncate,
    AddStatesFromEpisodesToBatch,
    AddTimeDimToBatchAndZeroPad,
    AgentToModuleMapping,
    BatchIndividualItems,
    GeneralAdvantageEstimation,
)
from typing_extensions import deprecated

from ray_utilities.connectors.debug_connector import DebugConnector
from ray_utilities.connectors.dummy_connector import DummyConnector

if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.connectors.connector_v2 import ConnectorV2


# Append OBS handling.
# deprecated; this is mostly not needed. Default pipeline can be used and adjusted afterwards.
def _learner_connector_without_numpy(
    obs_space,  # noqa: ARG001
    action_space,  # noqa: ARG001
    *,
    algo: PPOConfig,
    debug: bool = False,
) -> ConnectorV2 | list[ConnectorV2]:
    pipeline = []
    if debug:
        pipeline.append(DebugConnector(name="LearnerConnectorStart"))
    pipeline.append(AddObservationsFromEpisodesToBatch(as_learner_connector=True))
    # Append all other columns handling.
    pipeline.append(AddColumnsFromEpisodesToTrainBatch())
    # Append time-rank handler.
    pipeline.append(AddTimeDimToBatchAndZeroPad(as_learner_connector=True))
    # Append STATE_IN/STATE_OUT handler.
    pipeline.append(AddStatesFromEpisodesToBatch(as_learner_connector=True))
    # If multi-agent -> Map from AgentID-based data to ModuleID based data.
    if algo.is_multi_agent:
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec  # noqa: PLC0415

        pipeline.append(
            AgentToModuleMapping(
                rl_module_specs=(
                    algo.rl_module_spec.rl_module_specs
                    if isinstance(algo.rl_module_spec, MultiRLModuleSpec)
                    else set(algo.policies)  # pyright: ignore[reportArgumentType]
                ),
                agent_to_module_mapping_fn=algo.policy_mapping_fn,
            )
        )
    # Batch all data.
    pipeline.append(BatchIndividualItems(multi_agent=algo.is_multi_agent))
    # Convert to Tensors.
    # pipeline.append(NumpyToTensor(as_learner_connector=True, device=device))
    if debug:
        pipeline.append(DebugConnector(name="LearnerConnectorEnd"))
    # NOTE Ts and GAE estimator are missing
    pipeline.insert(0, AddOneTsToEpisodesAndTruncate())
    # At the end of the pipeline (when the batch is already completed), add the
    # GAE connector, which performs a vf forward pass, then computes the GAE
    # computations, and puts the results of this (advantages, value targets)
    # directly back in the batch. This is then the batch used for
    # `forward_train` and `compute_losses`.
    gae_estimator = GeneralAdvantageEstimation(gamma=algo.gamma, lambda_=algo.lambda_)
    if algo.learner_config_dict.get("no_numpy_to_tensor_connector", True):
        gae_estimator._numpy_to_tensor_connector = DummyConnector()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    pipeline.append(gae_estimator)
    return pipeline


@deprecated(
    "This connector is not sufficient. Pipeline, especially GeneralAdvantageEstimation, "
    "need further motification, best done in the class itself. "
)
def make_learner_connector_without_numpy(
    algo, *, debug=False
) -> Callable[[gym.Space, gym.Space], ConnectorV2 | list[ConnectorV2]]:
    """Make env_to_module without NumpyToTensor conversion."""
    return partial(_learner_connector_without_numpy, algo=algo, debug=debug)
