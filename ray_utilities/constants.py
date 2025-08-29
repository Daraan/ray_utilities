"""Constants and configuration values used throughout Ray Utilities.

Defines important constants, version checks, and configuration values used across
the Ray Utilities package. Includes version compatibility flags, metric keys for
Ray RLlib, Comet ML configuration, and video logging constants.

Key Constants:
    :data:`RAY_VERSION`: Current Ray version for compatibility checks
    :data:`RAY_NEW_API_STACK_ENABLED`: Whether Ray's new API stack is available
    :data:`EVAL_METRIC_RETURN_MEAN`: Standard evaluation return metric key
    :data:`DEFAULT_VIDEO_DICT_KEYS`: Video logging configuration keys
    :data:`COMET_OFFLINE_DIRECTORY`: Path for offline Comet ML experiments

Example:
    >>> from ray_utilities.constants import RAY_NEW_API_STACK_ENABLED
    >>> if RAY_NEW_API_STACK_ENABLED:
    ...     # Use new Ray API features
    ...     pass
"""

import os
from pathlib import Path
import time

import gymnasium as gym
from packaging.version import Version, parse as parse_version
import ray
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
)

_COMET_OFFFLINE_DIRECTORY_SUGGESTION = (Path("../") / "outputs" / ".cometml-runs").resolve()
_COMET_OFFFLINE_DIRECTORY_SUGGESTION_STR = str(_COMET_OFFFLINE_DIRECTORY_SUGGESTION)

if (
    os.environ.get("COMET_OFFLINE_DIRECTORY", _COMET_OFFFLINE_DIRECTORY_SUGGESTION_STR)
    != _COMET_OFFFLINE_DIRECTORY_SUGGESTION_STR
):
    import logging

    logging.getLogger(__name__).warning(
        "COMET_OFFLINE_DIRECTORY already set to: %s", os.environ.get("COMET_OFFLINE_DIRECTORY")
    )

os.environ["COMET_OFFLINE_DIRECTORY"] = COMET_OFFLINE_DIRECTORY = _COMET_OFFFLINE_DIRECTORY_SUGGESTION_STR
"""str: Directory path for storing offline Comet ML experiments.

This directory is where Comet ML stores experiment archives when running in offline mode.
The default location is ``../outputs/.cometml-runs`` relative to the current working directory.
Can be overridden by setting the ``COMET_OFFLINE_DIRECTORY`` environment variable.

See Also:
    :class:`ray_utilities.comet.CometArchiveTracker`: For managing offline experiments
"""

# Evaluation and Training Metric Keys
EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
"""str: Standard path for evaluation episode return mean metric.

This metric key path follows Ray RLlib's hierarchical result structure:
``evaluation/env_runner_results/episode_return_mean``
"""

DISC_EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/discrete/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
"""str: Path for discrete evaluation episode return mean metric.

Used when discrete evaluation is enabled alongside standard evaluation:
``evaluation/discrete/env_runner_results/episode_return_mean``
"""

TRAIN_METRIC_RETURN_MEAN = ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
"""str: Standard path for training episode return mean metric.

Training metric path: ``env_runner_results/episode_return_mean``
"""

# Video Recording Constants
EPISODE_VIDEO_PREFIX = "episode_videos_"
"""str: Prefix used for all episode video metric keys."""

EPISODE_BEST_VIDEO = "episode_videos_best"
"""str: Key for best episode video recordings."""

EPISODE_WORST_VIDEO = "episode_videos_worst"
"""str: Key for worst episode video recordings."""

EVALUATION_BEST_VIDEO_KEYS = (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, EPISODE_BEST_VIDEO)
EVALUATION_WORST_VIDEO_KEYS = (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, EPISODE_WORST_VIDEO)
DISCRETE_EVALUATION_BEST_VIDEO_KEYS = (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, EPISODE_BEST_VIDEO)
DISCRETE_EVALUATION_WORST_VIDEO_KEYS = (EVALUATION_RESULTS, "discrete", ENV_RUNNER_RESULTS, EPISODE_WORST_VIDEO)

EVALUATION_BEST_VIDEO = "/".join(EVALUATION_BEST_VIDEO_KEYS)
EVALUATION_WORST_VIDEO = "/".join(EVALUATION_WORST_VIDEO_KEYS)
DISCRETE_EVALUATION_BEST_VIDEO = "/".join(DISCRETE_EVALUATION_BEST_VIDEO_KEYS)
DISCRETE_EVALUATION_WORST_VIDEO = "/".join(DISCRETE_EVALUATION_WORST_VIDEO_KEYS)

DEFAULT_VIDEO_DICT_KEYS = (
    EVALUATION_BEST_VIDEO_KEYS,
    EVALUATION_WORST_VIDEO_KEYS,
    DISCRETE_EVALUATION_BEST_VIDEO_KEYS,
    DISCRETE_EVALUATION_WORST_VIDEO_KEYS,
)
"""tuple[tuple[str, ...], ...]: Collection of video key tuples for default video logging.

This contains the hierarchical key paths for all default video types that can be logged
during evaluation. Each tuple represents a path through the nested result dictionary.

Note:
    The video data might be a dictionary containing both ``"video"`` and ``"reward"`` keys
    rather than just the raw video array.

See Also:
    :data:`DEFAULT_VIDEO_DICT_KEYS_FLATTENED`: String versions of these keys
"""

DEFAULT_VIDEO_DICT_KEYS_FLATTENED = (
    EVALUATION_BEST_VIDEO,
    EVALUATION_WORST_VIDEO,
    DISCRETE_EVALUATION_BEST_VIDEO,
    DISCRETE_EVALUATION_WORST_VIDEO,
)
"""tuple[str, ...]: Flattened string keys for default video logging.

These are the slash-separated string versions of the video keys for use in flat
dictionaries or when the nested structure has been flattened.

Note:
    The video data might be a dictionary containing both ``"video"`` and ``"reward"`` keys
    rather than just the raw video array.

See Also:
    :data:`DEFAULT_VIDEO_DICT_KEYS`: Tuple versions of these keys for nested access
"""

assert all(EPISODE_VIDEO_PREFIX in key for key in DEFAULT_VIDEO_DICT_KEYS_FLATTENED)

EVALUATED_THIS_STEP = "evaluated_this_step"
"""str: Boolean metric key to indicate evaluation was performed this training step.

When logged with ``reduce_on_results=True``, this metric tracks whether evaluation
was actually run during a particular training iteration, which is useful for
conditional processing of evaluation results.
"""

# Version Compatibility Flags
RAY_VERSION = parse_version(ray.__version__)
"""Version: Parsed version of the currently installed Ray package.

Used throughout the codebase for version-specific compatibility checks and feature detection.

Example:
    >>> from ray_utilities.constants import RAY_VERSION
    >>> if RAY_VERSION >= Version("2.40.0"):
    ...     print("New API stack available")
"""

GYM_VERSION = parse_version(gym.__version__)
"""Version: Parsed version of the currently installed Gymnasium package."""

GYM_V1: bool = GYM_VERSION >= Version("1.0.0")
"""bool: True if Gymnasium version 1.0.0 or higher is installed.

Gymnasium 1.0.0 introduced significant API changes and improvements over earlier versions.
"""

GYM_V_0_26: bool = GYM_VERSION >= Version("0.26")
"""bool: True if Gymnasium version 0.26 or higher is installed.

Version 0.26 was the first official Gymnasium release after the transition from OpenAI Gym.
This flag is used to handle compatibility between legacy Gym and modern Gymnasium.
"""

RAY_UTILITIES_INITIALIZATION_TIMESTAMP = time.time()
"""float: Unix timestamp of when the Ray Utilities package was first imported.

Useful for tracking package initialization time and calculating elapsed time since import.
"""

# CLI and Reporting Configuration
CLI_REPORTER_PARAMETER_COLUMNS = ["algo", "module", "model_config"]
"""list[str]: Default parameter columns to display in CLI progress reports.

These parameter keys from the search space will be shown in Ray Tune's command-line
progress reporter for easier experiment monitoring.
"""

RAY_NEW_API_STACK_ENABLED = RAY_VERSION >= Version("2.40.0")
"""bool: True if Ray's new API stack is available (Ray >= 2.40.0).

The new API stack introduced significant improvements to RLlib's architecture,
including better modularity and performance. This flag enables conditional
code paths for new vs. legacy API usage.

See Also:
    `Ray RLlib New API Stack Migration Guide
    <https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html>`_
"""

# Sampling and Training Metrics
NUM_ENV_STEPS_PASSED_TO_LEARNER = "num_env_steps_passed_to_learner"
"""str: Metric key for environment steps passed to learner with exact sampling.

When using exact sampling mode, this tracks the actual number of environment steps
that were passed to the learner during the current training iteration, which may
differ from the total steps sampled due to exact sampling constraints.
"""

NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME = "num_env_steps_passed_to_learner_lifetime"
"""str: Lifetime metric for environment steps passed to learner with exact sampling.

Cumulative count of environment steps passed to the learner over the entire
training run when using exact sampling mode.
"""

CURRENT_STEP = "current_step"
"""str: The current training step metric key.

This top-level metric tracks the current step in the training process. With exact
sampling, it aligns with :data:`NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME`, otherwise
it aligns with ``NUM_ENV_STEPS_SAMPLED_LIFETIME``.

Used for consistent step tracking across different sampling modes and is the
recommended metric for tracking training progress.
"""

PERTURBED_HPARAMS = "__perturbed__"
"""str: Configuration key indicating which hyperparameters have been perturbed.

This key in a trainable's config dictionary contains information about which
hyperparameters have been modified from their original values. It's primarily
used by :meth:`~ray_utilities.training.default_class.DefaultTrainable.load_checkpoint`
and not during trainable initialization.

Note:
    This is used internally by the checkpoint/restore system and typically should
    not be set manually in experiment configurations.
"""
