# ruff: noqa: ARG001,ARG002
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing_extensions import deprecated

from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME

if TYPE_CHECKING:
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


def _remove_or_trim_samples(samples: list[EpisodeType], total_samples: int, exact_timesteps: int) -> None:
    """
    Removes or trims samples to match the exact number of timesteps.

    Attention:
        Remove samples in-place
    """
    _total_samples_before = total_samples
    diff = total_samples - exact_timesteps
    lengths = [len(sae) for sae in samples]
    exact_matches = [idx if length == diff else False for idx, length in enumerate(lengths)]
    # If there is a sample with exact length, remove it. Appears to be the most likely case.
    matching_done_sample_idx = None
    if exact_matches:
        # Look for a not done episode first.
        for idx in exact_matches:
            if idx is not False and not samples[idx].is_done:
                logger.debug(
                    "Removing a sample with exact length %d: %s. Sample was done: %s",
                    diff,
                    samples[idx],
                    samples[idx].is_done,
                )
                samples.pop(idx)
                return
        matching_done_sample_idx = next((i for i, s in enumerate(samples) if s.is_done), None)
    assert not any(exact_matches) or exact_matches[0] == 0

    # Now find samples that are not done and have enough timesteps to trim, but at least one more
    # Not done episodes are likely(?) at the back, reverse iterate
    for i, sample in enumerate(reversed(samples), start=1):
        if not sample.is_done and len(sample) >= diff + 1:
            samples[-i] = sample[:-diff]
            return

    # Should avoid trimming done episodes (might raise an error in metrics later)
    # trim multiple not done-episodes
    # TODO: is all episodes are short, trim whole episodes instead of all a little
    trimmed = 0
    min_trim = diff
    for i, sample in enumerate(reversed(samples), start=1):
        if not sample.is_done and len(sample) > 1:
            max_trim = min(min_trim, len(sample) - 1)  # at least one timestep should remain
            assert max_trim > 0
            # if it has length < 2; and diff >=2 could also remove episode.
            samples[-i] = sample[:-max_trim]
            logger.debug("Trimmed a not done episode: %s by %d. Need to trim %d/%d", sample, max_trim, trimmed, diff)
            trimmed += max_trim
            if trimmed >= diff:
                return
            min_trim -= max_trim  # reduce
    # if there are (now) some samples with length 1, we can maybe remove them:
    # TODO: Test if its okay to pop samples
    len_samples = len(samples)  # at start
    for i, sample in enumerate(reversed(samples), start=1):
        if not sample.is_done and len(sample) == 1:
            assert len(samples.pop(len_samples - i)) == 1
            logger.debug("Removed a sample with length 1: %s", sample)
            trimmed += 1
            if trimmed >= diff:
                return
    # Out of options need to trim /remove a done episode if it has enough timesteps.
    # We had one with exact length but as we might have trimmed already it might not match again:
    if matching_done_sample_idx is not None and len(samples[matching_done_sample_idx]) == diff:
        logger.info("Had to remove a done sample with exact length %d: %s", diff, samples[matching_done_sample_idx])
        samples.pop(matching_done_sample_idx)
        return
    for i, sample in enumerate(samples):
        if len(sample) >= diff + 1:
            logger.info("Had to trim one done episode: %s by %d", sample, diff)
            samples[i] = sample[diff:]  # keep end of episode slice away at start
            assert len(samples[i]) > 0
            # NOTE: Settings is terminated=False will raise an error on assertion that it
            # is done (episode is tracked in done episodes)
            # sample.is_terminated = False
            return
    # need to trim multiple episodes
    trimmed = 0
    for i, sample in enumerate(samples):
        if len(sample) > 1:
            max_trim = max(1, min(diff - trimmed, len(sample) - 1))  # at least one timestep should remain, never be 0
            logger.warning("Had to trim a done episode (one of multiple): %s by %d.", sample, max_trim)
            samples[i] = sample[max_trim:]
            assert len(samples[i]) > 0
            # sample.is_terminated = False
            trimmed += max_trim
            if trimmed >= diff:
                return
            diff -= max_trim
    logger.warning(
        "Could not trim enough samples to match exact timesteps %s. Total samples before: %s, after: %s.",
        exact_timesteps,
        _total_samples_before,
        total_samples,
    )


def exact_sampling_callback(
    *,
    env_runner: SingleAgentEnvRunner,
    metrics_logger: MetricsLogger,
    samples: list[EpisodeType],  # could also be SampleBatch. Is not a copy.
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
        _remove_or_trim_samples(samples, total_samples=total_samples, exact_timesteps=exact_timesteps)
        if any(len(sae) <= 0 for sae in samples):
            logger.error(
                "Some samples are empty after exact sampling: %s. Total samples before: %s, after: %s.",
                samples,
                _total_samples_before,
                total_samples,
            )
        total_samples = sum(len(sae) for sae in samples)
        if any(len(sae) <= 0 for sae in samples):
            logger.error(
                "Some samples are empty after exact sampling (this should not happen): %s. "
                "Total samples before: %s, after: %s.",
                samples,
                _total_samples_before,
                total_samples,
            )
            samples = [sample for sample in samples if len(sample) > 0]
    if total_samples != exact_timesteps:
        logger.error(
            "Total samples %s does not match exact timesteps %s. Some calculations might be off. "
            "This callback failed to reduce the samples.",
            total_samples,
            exact_timesteps,
        )
    # _correct_increase_sampled_metrics(metrics_logger, total_samples)
    _log_steps_to_learner(metrics_logger, total_samples)


@deprecated("old api, no need for stateful class")
class ExactSamplingCallback(DefaultCallbacks):
    """Reduces the samples of the env_runners to an exact number of samples"""

    on_sample_end = staticmethod(exact_sampling_callback)  # pyright: ignore[reportAssignmentType]
