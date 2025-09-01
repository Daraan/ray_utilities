"""Discrete evaluation utilities for Ray RLlib algorithms.

This module provides functionality for running discrete evaluation episodes
alongside standard evaluation in Ray RLlib experiments. The discrete evaluation
feature allows for additional evaluation metrics that are logged separately
from the main evaluation results.

The main function :func:`discrete_evaluate_on_local_env_runner` is a modified
version of Ray's standard evaluation that logs results to a separate
``evaluation/discrete`` key in the metrics hierarchy.

Functions:
    :func:`discrete_evaluate_on_local_env_runner`: Run discrete evaluation episodes

See Also:
    :class:`ray_utilities.callbacks.algorithm.discrete_eval_callback.DiscreteEvalCallback`:
        Callback that integrates discrete evaluation into training
    :data:`ray_utilities.constants.DISC_EVAL_METRIC_RETURN_MEAN`:
        Metric key for discrete evaluation results
    :mod:`ray.rllib.evaluation.metrics`:
        Ray RLlib evaluation utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ray.rllib.evaluation.metrics import summarize_episodes
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
)

if TYPE_CHECKING:
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
    from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
    from ray.rllib.evaluation.metrics import RolloutMetrics


def discrete_evaluate_on_local_env_runner(
    self: Algorithm, env_runner: SingleAgentEnvRunner, metrics_logger: MetricsLogger
):
    """Run discrete evaluation episodes on a local environment runner.

    This is a modified version of Ray's standard evaluation function that logs
    results to the ``evaluation/discrete`` path instead of the main evaluation
    path. This allows for additional evaluation metrics without interfering
    with the standard evaluation process.

    Args:
        self: The Ray RLlib :class:`~ray.rllib.algorithms.Algorithm` instance.
        env_runner: The :class:`~ray.rllib.env.single_agent_env_runner.SingleAgentEnvRunner` to use for evaluation.
        metrics_logger: The
            :class:`~ray.rllib.utils.metrics.metrics_logger.MetricsLogger` for
            recording evaluation metrics.

    Returns:
        A tuple containing:

        - **env_runner_results**: Evaluation results dictionary (or ``None`` for new API)
        - **env_steps**: Number of environment steps taken during evaluation
        - **agent_steps**: Number of agent steps taken during evaluation
        - **all_batches**: List of sample batches from evaluation episodes

    Raises:
        ValueError: If the local environment runner doesn't have an environment
            or if parallel evaluation is enabled (not supported for discrete evaluation).

    Example:
        This function is typically used within a callback rather than called
        directly.

        .. code-block:: python

            # In a custom callback
            from ray_utilities.discrete_evaluation import (
                discrete_evaluate_on_local_env_runner,
            )


            class MyCallback(DefaultCallbacks):
                def on_algorithm_iteration_end(self, *, algorithm, result, **kwargs):
                    if should_run_discrete_eval():
                        env_runner = algorithm.env_runner
                        metrics_logger = algorithm.metrics_logger
                        discrete_evaluate_on_local_env_runner(algorithm, env_runner, metrics_logger)

    Note:
        - This function respects the algorithm's evaluation configuration settings
          like ``evaluation_duration`` and ``evaluation_duration_unit``
        - Results are logged to ``evaluation/discrete/env_runner_results`` path
        - Compatible with both Ray's old and new API stacks
        - For the new API stack (Ray >= 2.40.0), results are logged directly
          via the metrics logger rather than returned

    See Also:
        :class:`ray_utilities.callbacks.algorithm.discrete_eval_callback.DiscreteEvalCallback`:
            Callback that uses this function for automated discrete evaluation
        :data:`ray_utilities.constants.RAY_NEW_API_STACK_ENABLED`:
            Version flag for API compatibility
    """
    if hasattr(env_runner, "input_reader") and env_runner.input_reader is None:  # type: ignore[attr-defined]
        raise ValueError(
            "Can't evaluate on a local worker if this local worker does not have "
            "an environment!\nTry one of the following:"
            "\n1) Set `evaluation_interval` > 0 to force creating a separate "
            "evaluation EnvRunnerGroup.\n2) Set `create_env_on_driver=True` to "
            "force the local (non-eval) EnvRunner to have an environment to "
            "evaluate on."
        )
    assert self.config
    if self.config.evaluation_parallel_to_training:
        raise ValueError(
            "Cannot run on local evaluation worker parallel to training! Try "
            "setting `evaluation_parallel_to_training=False`."
        )

    # How many episodes/timesteps do we need to run?
    unit = self.config.evaluation_duration_unit
    duration: int = self.config.evaluation_duration  # type: ignore
    eval_cfg = self.evaluation_config

    env_steps = agent_steps = 0

    all_batches: list[SampleBatch | MultiAgentBatch] = []
    if self.config.enable_env_runner_and_connector_v2:
        episodes = env_runner.sample(
            num_timesteps=duration if unit == "timesteps" else None,  # pyright: ignore[reportArgumentType]
            num_episodes=duration if unit == "episodes" else None,  # pyright: ignore[reportArgumentType]
        )
        agent_steps += sum(e.agent_steps() for e in episodes)
        env_steps += sum(e.env_steps() for e in episodes)
    elif unit == "episodes":
        # OLD API
        for _ in range(duration):
            batch: SampleBatch | MultiAgentBatch = env_runner.sample()  # type: ignore[assignment]
            agent_steps += batch.agent_steps()
            env_steps += batch.env_steps()
            if self.reward_estimators:
                all_batches.append(batch)
    else:
        batch: SampleBatch | MultiAgentBatch = env_runner.sample()  # type: ignore[assignment]
        agent_steps += batch.agent_steps()
        env_steps += batch.env_steps()
        if self.reward_estimators:
            all_batches.append(batch)

    env_runner_results = env_runner.get_metrics()

    if not self.config.enable_env_runner_and_connector_v2:
        # OLD API
        env_runner_results = cast("list[RolloutMetrics]", env_runner_results)  # pyright: ignore[reportInvalidTypeForm] # bad support in ray
        env_runner_results = summarize_episodes(
            env_runner_results,
            env_runner_results,
            keep_custom_metrics=eval_cfg.keep_per_episode_custom_metrics,  # pyright: ignore
        )
    else:
        # NEW API; do not return result dict but use metrics logger
        metrics_logger.log_dict(
            env_runner_results,
            key=(EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS),
        )
        env_runner_results = None

    return env_runner_results, env_steps, agent_steps, all_batches
