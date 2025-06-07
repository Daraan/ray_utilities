# ruff: noqa: ARG001,ARG002
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing_extensions import deprecated

from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME

if TYPE_CHECKING:
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
    from ray.rllib.policy.sample_batch import SampleBatch  # noqa: F401
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.typing import EpisodeType

__all__ = ["exact_sampling_callback"]

logger: logging.Logger = logging.getLogger(__name__)


def _log_steps_to_learner(metrics: MetricsLogger, num_steps: int) -> None:
    """Log the number of steps that are actually passed to the learner."""
    metrics.log_value(NUM_ENV_STEPS_PASSED_TO_LEARNER, num_steps, reduce="sum", clear_on_reduce=True)
    metrics.log_value(NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME, num_steps, reduce="sum")


def exact_sampling_callback(
    *,
    env_runner: SingleAgentEnvRunner,
    metrics_logger: MetricsLogger,
    samples: list[EpisodeType],  # could also be SampleBatch
    worker: Optional["EnvRunner"] = None,
    **kwargs,
) -> None:
    if env_runner.config.in_evaluation:
        # We do not care about trimming in evaluation; further during evaluation (with evaluation_config)
        # the rollout_fragment_length is 1 *episode* if evaluation_duration_unit == "episodes"
        # otherwise it evaluation_duration / self.evaluation_num_env_runners
        return
    total_samples = _total_samples_before = sum(len(sae) for sae in samples)
    exact_timesteps = env_runner.config.get_rollout_fragment_length(env_runner.worker_index) * env_runner.num_envs
    if total_samples > exact_timesteps:
        diff = total_samples - exact_timesteps
        for i, sample in enumerate(samples):
            if not sample.is_done and len(sample) >= diff:
                samples[i] = sample[:-diff]
                break
        else:
            # this is wrong when the last sample is done but very short.
            samples[-1] = samples[-1][:-diff]
        total_samples = sum(len(sae) for sae in samples)
    assert total_samples == exact_timesteps, (
        f"Total samples {total_samples} does not match exact timesteps {exact_timesteps}."
    )
    # _correct_increase_sampled_metrics(metrics_logger, total_samples)
    _log_steps_to_learner(metrics_logger, exact_timesteps)


@deprecated("old api, no need for stateful class")
class ExactSamplingCallback(DefaultCallbacks):
    """Reduces the samples of the env_runners to an exact number of samples"""

    on_sample_end = staticmethod(exact_sampling_callback)  # pyright: ignore[reportAssignmentType]
