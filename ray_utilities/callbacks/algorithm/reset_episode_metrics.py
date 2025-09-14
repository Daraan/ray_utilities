from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

from ray_utilities.constants import EPISODE_METRICS_KEYS

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


def _reset_episode_metrics(runner: EnvRunner):
    for key in EPISODE_METRICS_KEYS:
        runner.metrics.set_value(key, [])


def reset_eval_episode_metrics_each_time(
    algorithm: "Algorithm",
    **kwargs,
):
    """
    RLlib used metrics_num_episodes_for_smoothing to metrics like episode_reward_mean over
    a window of episodes.
    But when there are more episodes than the smoothing window, the result of the first episodes is
    ignored.
    Likewise if there are less episodes, information from the last iteration bleeds in.

    This callback is used on the algorithm `on_evaluation_start` (only eval)
    """
    algorithm.eval_env_runner_group.foreach_env_runner(lambda env_runner: _reset_episode_metrics(env_runner))


def reset_episode_metrics_each_iteration(
    env_runner: Optional["EnvRunner"] = None,
    metrics_logger: Optional[MetricsLogger] = None,
    **kwargs,
) -> None:
    """
    RLlib used metrics_num_episodes_for_smoothing to metrics like episode_reward_mean over
    a window of episodes.
    But when there are more episodes than the smoothing window, the result of the first episodes is
    ignored.
    Likewise if there are less episodes, information from the last iteration bleeds in.

    This callback is used on the algorithm `on_sample_end` (all env runners)
    """
    if metrics_logger is None:
        assert env_runner
        metrics_logger = env_runner.metrics
    for key in EPISODE_METRICS_KEYS:  # keys are not present in the first iteration
        metrics_logger.delete(key, key_error=metrics_logger.peek("weight_seq_no", compile=True, default=0))  # pyright: ignore[reportArgumentType]
