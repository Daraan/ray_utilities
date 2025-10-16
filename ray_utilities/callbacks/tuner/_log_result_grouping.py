"""
- non_metric_results: Considered metrics and should rather be logged to the config of a trial.
- exclude_results: Results are excluded from logging entirely.
"""

from __future__ import annotations

from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS
from typing_extensions import LiteralString

from ray_utilities.constants import ENVIRONMENT_RESULTS, SEED, SEEDS, TRAINING

non_metric_results: set[str | LiteralString] = {
    "comment",
    "cli_args/comment",
    "run_id",
    "experiment_name",
    "experiment_group",
    "experiment_key",
    "trainable_name",
    f"{ENV_RUNNER_RESULTS}/{ENVIRONMENT_RESULTS}/{SEEDS}",
    f"{ENV_RUNNER_RESULTS}/{ENVIRONMENT_RESULTS}/{SEEDS}/seed_sequence",
    f"{ENV_RUNNER_RESULTS}/{ENVIRONMENT_RESULTS}/{SEED}",
    f"{ENV_RUNNER_RESULTS}/{ENVIRONMENT_RESULTS}/{SEED}/initial_seed",
    f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{ENVIRONMENT_RESULTS}/{SEEDS}",
    f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{ENVIRONMENT_RESULTS}/{SEEDS}/seed_sequence",
    f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{ENVIRONMENT_RESULTS}/{SEED}",
    f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{ENVIRONMENT_RESULTS}/{SEED}/initial_seed",
    # New log style
    f"{TRAINING}/{ENVIRONMENT_RESULTS}/{SEEDS}",
    f"{TRAINING}/{ENVIRONMENT_RESULTS}/{SEEDS}/seed_sequence",
    f"{TRAINING}/{ENVIRONMENT_RESULTS}/{SEED}",
    f"{TRAINING}/{ENVIRONMENT_RESULTS}/{SEED}/initial_seed",
    f"{EVALUATION_RESULTS}/{ENVIRONMENT_RESULTS}/{SEEDS}",
    f"{EVALUATION_RESULTS}/{ENVIRONMENT_RESULTS}/{SEEDS}/seed_sequence",
    f"{EVALUATION_RESULTS}/{ENVIRONMENT_RESULTS}/{SEED}",
    f"{EVALUATION_RESULTS}/{ENVIRONMENT_RESULTS}/{SEED}/initial_seed",
}
"""Result keys for LoggerCallbacks that are not metrics and should not be plotted and e.g. logged to the config."""

exclude_results: set[str] = {
    "cli_args/test",
    "cli_args/num_jobs",
    "node_ip",
}
