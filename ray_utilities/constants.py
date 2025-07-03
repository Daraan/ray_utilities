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

EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
DISC_EVAL_METRIC_RETURN_MEAN = EVALUATION_RESULTS + "/discrete/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
# Keys
TRAIN_METRIC_RETURN_MEAN = ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN

EPISODE_VIDEO_PREFIX = "episode_videos_"
EPISODE_BEST_VIDEO = "episode_videos_best"
EPISODE_WORST_VIDEO = "episode_videos_worst"

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
"""
Collection of tuple[str, ...] keys for the default video keys to log

Note:
    The video might still be a dict with "video" and "reward" keys.
"""

DEFAULT_VIDEO_DICT_KEYS_FLATTENED = (
    EVALUATION_BEST_VIDEO,
    EVALUATION_WORST_VIDEO,
    DISCRETE_EVALUATION_BEST_VIDEO,
    DISCRETE_EVALUATION_WORST_VIDEO,
)
"""
String keys for the default video keys to log in flattened form

Note:
    The video might still be a dict with "video" and "reward" keys.
"""

assert all(EPISODE_VIDEO_PREFIX in key for key in DEFAULT_VIDEO_DICT_KEYS_FLATTENED)

EVALUATED_THIS_STEP = "evaluated_this_step"
"""
Metric to log as bool with reduce_on_results=True to indicate that evaluation was done this step.
"""


RAY_VERSION = parse_version(ray.__version__)
GYM_VERSION = parse_version(gym.__version__)
GYM_V1: bool = GYM_VERSION >= Version("1.0.0")
"""Gymnasium version 1.0.0 and above"""
GYM_V_0_26: bool = GYM_VERSION >= Version("0.26")
"""First gymnasium version and above"""
RAY_UTILITIES_INITIALIZATION_TIMESTAMP = time.time()

CLI_REPORTER_PARAMETER_COLUMNS = ["algo", "module", "model_config"]
"""Keys from param_space"""

RAY_NEW_API_STACK_ENABLED = RAY_VERSION >= Version("2.40.0")
"""
See Also:
    https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html
"""

NUM_ENV_STEPS_PASSED_TO_LEARNER = "num_env_steps_passed_to_learner"
"""When using exact sampling the key for the logger to log the number of environment steps actually
passed to the learner."""

NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME = "num_env_steps_passed_to_learner_lifetime"
"""When using exact sampling the key for the logger to log the number of environment steps actually
passed to the learner over the lifetime of the algorithm."""
