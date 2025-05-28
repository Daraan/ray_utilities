# ruff: noqa: ARG001,ARG002
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing_extensions import deprecated

if TYPE_CHECKING:
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
    from ray.rllib.policy.sample_batch import SampleBatch  # noqa: F401
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.typing import EpisodeType

__all__ = ["exact_sampling_callback"]

logger: logging.Logger = logging.getLogger(__name__)


def exact_sampling_callback(
    *,
    env_runner: SingleAgentEnvRunner,
    metrics_logger: Optional[MetricsLogger] = None,
    samples: list[EpisodeType],  # could also be SampleBatch
    worker: Optional["EnvRunner"] = None,
    **kwargs,
) -> None:
    total_samples = sum(len(sae) for sae in samples)
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


@deprecated("old api, no need for statefull class")
class ExactSamplingCallback(DefaultCallbacks):
    """Reduces the samples of the env_runners to an exact number of samples"""

    on_sample_end = staticmethod(exact_sampling_callback)  # pyright: ignore[reportAssignmentType]
