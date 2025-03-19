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
    """
    Copy of rays evaluate that logs to a evaluation/discrete key

    See Also:
        DiscreteEvalCallback
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
