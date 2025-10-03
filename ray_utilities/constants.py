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
# pyright: enableExperimentalFeatures=true

import logging
import re
from typing import Final

import gymnasium as gym
import ray
from packaging.version import Version
from packaging.version import parse as parse_version
from ray.rllib.core import DEFAULT_AGENT_ID, DEFAULT_MODULE_ID
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_LEN_MAX,
    EPISODE_LEN_MEAN,
    EPISODE_LEN_MIN,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    EVALUATION_RESULTS,
)
from typing_extensions import Sentinel

from ._runtime_constants import (
    COMET_OFFLINE_DIRECTORY,
    ENTRY_POINT,
    ENTRY_POINT_ID,
    RAY_UTILITIES_INITIALIZATION_TIMESTAMP,
    RUN_ID,
)

__all__ = []
__all__ += [
    "COMET_OFFLINE_DIRECTORY",
    "ENTRY_POINT",
    "ENTRY_POINT_ID",
    "RAY_UTILITIES_INITIALIZATION_TIMESTAMP",
    "RUN_ID",
]

_logger = logging.getLogger(__name__)

# region runtime constants

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

RAY_NEW_API_STACK_ENABLED = RAY_VERSION >= Version("2.40.0")
"""bool: True if Ray's new API stack is available (Ray >= 2.40.0).

The new API stack introduced significant improvements to RLlib's architecture,
including better modularity and performance. This flag enables conditional
code paths for new vs. legacy API usage.

See Also:
    `Ray RLlib New API Stack Migration Guide
    <https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html>`_
"""

# endregion

RAY_UTILITIES_NEW_LOG_FORMAT: Final[str] = "RAY_UTILITIES_NEW_LOG_FORMAT"
"""
Environment variable key to enable new log format if found in :py:obj:`os.environ`.

Note:
    The new logging format is used in downstream loggers, e.g. for WandB or Comet ML.
    other callbacks and logs are not affected and should not check for the new format.
"""

TRAINING = "training"
"""For the rllib layout use ENV_RUNNER_RESULTS instead of "training"."""

# Evaluation and Training Metric Keys
EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
"""str: Standard path for evaluation episode return mean metric.

This metric key path follows Ray RLlib's hierarchical result structure:
``evaluation/env_runner_results/episode_return_mean``
"""

NEW_LOG_EVAL_METRIC = EVALUATION_RESULTS + "/" + EPISODE_RETURN_MEAN
"""Metric key used in loggers when new log format is used."""

DISC_EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/discrete/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
"""str: Path for discrete evaluation episode return mean metric.

Used when discrete evaluation is enabled alongside standard evaluation:
``evaluation/discrete/env_runner_results/episode_return_mean``
"""

NEW_LOG_DISC_EVAL_METRIC = EVALUATION_RESULTS + "/discrete/" + EPISODE_RETURN_MEAN
"""Metric key used in loggers when new log format is used and discrete evaluation is enabled."""

TRAIN_METRIC_RETURN_MEAN = ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
"""str: Standard path for training episode return mean metric.

Training metric path: ``env_runner_results/episode_return_mean``
"""

NEW_LOG_TRAIN_METRIC = TRAINING + "/" + EPISODE_RETURN_MEAN
"""Metric key used in loggers when new log format is used and training is enabled."""

DEFAULT_EVAL_METRIC = Sentinel("DEFAULT_EVAL_METRIC")
"""typing.Sentinel value

Default evaluation metric sentinel value to be replaced with actual metric key
depending on wether the new log format is used or not (variable RAY_UTILITIES_NEW_LOG_FORMAT is set),
also possibly depending on wether discrete evaluation is enabled or not.

The correct metric key should be retrieved by:

.. code-block:: python

    if metric is DEFAULT_EVAL_METRIC:
        metric = NEW_LOG_EVAL_METRIC if new_log_format_used() else EVAL_METRIC_RETURN_MEAN
"""

ENVIRONMENT_RESULTS = "environments"
"""
Metric key for environment-specific results in Trainable results.

See Also:
    :class:`SeededEnvsCallback`
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

# region: new layout for log metrics

# NEW variants: omit ENV_RUNNER_RESULTS for evaluation, substitute ENV_RUNNER_RESULTS with "training" for training
EVALUATION_BEST_VIDEO_KEYS_NEW = (EVALUATION_RESULTS, EPISODE_BEST_VIDEO)
EVALUATION_WORST_VIDEO_KEYS_NEW = (EVALUATION_RESULTS, EPISODE_WORST_VIDEO)
DISCRETE_EVALUATION_BEST_VIDEO_KEYS_NEW = (EVALUATION_RESULTS, "discrete", EPISODE_BEST_VIDEO)
DISCRETE_EVALUATION_WORST_VIDEO_KEYS_NEW = (EVALUATION_RESULTS, "discrete", EPISODE_WORST_VIDEO)

EVALUATION_BEST_VIDEO_NEW = "/".join(EVALUATION_BEST_VIDEO_KEYS_NEW)
EVALUATION_WORST_VIDEO_NEW = "/".join(EVALUATION_WORST_VIDEO_KEYS_NEW)
DISCRETE_EVALUATION_BEST_VIDEO_NEW = "/".join(DISCRETE_EVALUATION_BEST_VIDEO_KEYS_NEW)
DISCRETE_EVALUATION_WORST_VIDEO_NEW = "/".join(DISCRETE_EVALUATION_WORST_VIDEO_KEYS_NEW)

# NEW variants for video dict keys (evaluation omits ENV_RUNNER_RESULTS, training uses "training")
DEFAULT_VIDEO_DICT_KEYS_NEW = (
    EVALUATION_BEST_VIDEO_KEYS_NEW,
    EVALUATION_WORST_VIDEO_KEYS_NEW,
    DISCRETE_EVALUATION_BEST_VIDEO_KEYS_NEW,
    DISCRETE_EVALUATION_WORST_VIDEO_KEYS_NEW,
)
"""tuple[tuple[str, ...], ...]: Collection of new-style video key tuples for logging.

Evaluation keys omit ENV_RUNNER_RESULTS, training keys use "training" instead.
"""

DEFAULT_VIDEO_DICT_KEYS_FLATTENED_NEW = (
    EVALUATION_BEST_VIDEO_NEW,
    EVALUATION_WORST_VIDEO_NEW,
    DISCRETE_EVALUATION_BEST_VIDEO_NEW,
    DISCRETE_EVALUATION_WORST_VIDEO_NEW,
)
"""tuple[str, ...]: Flattened string keys for new-style video logging.

Evaluation keys omit ENV_RUNNER_RESULTS, training keys use "training" instead.
"""

# endregion

DEFAULT_VIDEO_DICT_KEYS = (
    *DEFAULT_VIDEO_DICT_KEYS_NEW,
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
    *DEFAULT_VIDEO_DICT_KEYS_FLATTENED_NEW,
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


# CLI and Reporting Configuration
CLI_REPORTER_PARAMETER_COLUMNS = ["algo", "module", "model_config"]
"""list[str]: Default parameter columns to display in CLI progress reports.

These parameter keys from the search space will be shown in Ray Tune's command-line
progress reporter for easier experiment monitoring.
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

EPISODE_METRICS_KEYS = (
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_LEN_MAX,
    EPISODE_LEN_MEAN,
    EPISODE_LEN_MIN,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    ("module_episode_return_mean", DEFAULT_MODULE_ID),
    ("agent_episode_return_mean", DEFAULT_AGENT_ID),
)
"""Keys that are by default logged with a window by RLlib. When using"""

FORK_FROM = "fork_from"
"""
Key in trial configs to indicate the trial id and optionally step a trial was forked from.

The value should have the format ``<trial_id>?_step=<step>`` where ``?_step=<step>`` is optional,
it can be parsed with :const:`RE_PARSE_FORK_FROM` (deprecated)
or be a dict defined as :class:`ForkFromData`.

Note:
    A trial should never contain the ``?`` character.
"""

RE_PARSE_FORK_FROM = re.compile(r"^(?P<fork_id>[^?]*?)(?:\?_step=(?P<fork_step>\d+))?$")
"""
Regex pattern to parse the value of :const:`FORK_FROM` into trial id and step.

The ``fork_id`` will be everything before the first ``?``, the ``?_step=<fork_step>`` part is optional.
"""
