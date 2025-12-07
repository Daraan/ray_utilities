# ruff: noqa: ARG001,ARG002
from __future__ import annotations

import logging
from typing import TYPE_CHECKING


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
    metrics.log_value(NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME, num_steps, reduce="sum", clear_on_reduce=False)


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
    len_samples = len(samples)  # at start
    for i, sample in enumerate(reversed(samples), start=1):
        if not sample.is_done and len(sample) == 1:
            assert len(samples.pop(len_samples - i)) == 1
            logger.debug("Removed a sample with length 1: %s. Need to trim %d/%d", sample, trimmed, diff)
            trimmed += 1
            if trimmed >= diff:
                return

    # Calculate remaining amount to trim after all previous operations
    remaining_to_trim = diff - trimmed

    # Out of options need to trim/remove a done episode if it has enough timesteps. Check if we can remove any episode
    exact_idx = next((idx for idx, sample in enumerate(samples) if len(sample) == remaining_to_trim), None)
    if exact_idx is not None:
        logger.debug(
            "Removing a done sample with exact length %d: %s.",
            remaining_to_trim,
            samples[exact_idx],
        )
        samples.pop(exact_idx)
        return

    # Need to trim done episodes which are longer; try to find one that is long enough
    for i, sample in enumerate(samples):
        if len(sample) >= remaining_to_trim + 1:
            logger.info("Had to trim one done episode by %d: %s", remaining_to_trim, sample)
            samples[i] = sample[remaining_to_trim:]  # keep end of episode slice away at start
            assert len(samples[i]) > 0
            # NOTE: Settings is_terminated=False will raise an error on assertion that it
            # is done (episode is tracked in done episodes)
            # sample.is_terminated = False
            return

    # need to trim multiple episodes
    # Use remaining_to_trim for final trimming phase
    for i, sample in enumerate(samples):
        if len(sample) > 1 and remaining_to_trim > 0:
            max_trim = min(remaining_to_trim, len(sample) - 1)  # at least one timestep should remain
            if max_trim > 0:  # Only trim if we actually can
                logger.warning("Had to trim a done episode (one of multiple): %s by %d.", sample, max_trim)
                samples[i] = sample[max_trim:]
                assert len(samples[i]) > 0
                # sample.is_terminated = False
                remaining_to_trim -= max_trim
                if remaining_to_trim <= 0:
                    return

    if remaining_to_trim > 0:
        logger.warning(
            "Could not trim enough samples to match exact timesteps %s. Total samples before: %s, still need to trim: %s.",
            exact_timesteps,
            _total_samples_before,
            remaining_to_trim,
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
            samples[:] = [sample for sample in samples if len(sample) > 0]

    if total_samples != exact_timesteps:
        logger.error(
            "Total samples %s does not match exact timesteps %s. Some calculations might be off. "
            "This callback failed to reduce the samples.",
            total_samples,
            exact_timesteps,
        )
    # _correct_increase_sampled_metrics(metrics_logger, total_samples)
    _log_steps_to_learner(metrics_logger, total_samples)
