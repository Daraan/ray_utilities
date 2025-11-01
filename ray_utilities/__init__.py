"""Ray Utilities: Advanced utilities for Ray Tune and RLlib experiments.

Provides a comprehensive set of utilities, classes, and functions to streamline
Ray Tune hyperparameter optimization and Ray RLlib reinforcement learning experiments.

Main Components:
    - :class:`DefaultTrainable`: Base trainable class with checkpoint/restore functionality
    - :func:`run_tune`: Enhanced Ray Tune experiment runner with advanced logging
    - :func:`nice_logger`: Colored logging setup for better debugging
    - :func:`seed_everything`: Comprehensive seeding for reproducible experiments
    - :data:`AlgorithmReturnData`: Type definitions for algorithm return values

Example:
    >>> import ray_utilities as ru
    >>> logger = ru.nice_logger(__name__)
    >>> ru.seed_everything(env=None, seed=42)
    >>> trainable = ru.create_default_trainable(config_class=PPOConfig)
    >>> ru.run_tune(trainable, param_space=config, num_samples=10)
"""

# ruff: noqa: PLC0415  # imports at top level of file; safe import time if not needed.

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

# fmt: off
try:
    # Import comet early for its monkey patch
    import comet_ml
except ImportError:
    pass
else:
    del comet_ml
# fmt: on

from ray.runtime_env import RuntimeEnv as _RuntimeEnv

from ray_utilities.constants import (
    _RUN_ID,
    COMET_OFFLINE_DIRECTORY,
    ENTRY_POINT,
    ENTRY_POINT_ID,
    RAY_UTILITIES_INITIALIZATION_TIMESTAMP,
    get_run_id,
)
from ray_utilities.misc import get_trainable_name, is_pbar, shutdown_monitor, trial_name_creator
from ray_utilities.nice_logger import nice_logger
from ray_utilities.random import seed_everything
from ray_utilities.runfiles.run_tune import run_tune
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.functional import create_default_trainable, default_trainable
from ray_utilities.training.helpers import episode_iterator
from ray_utilities.typing.algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData

__all__ = [
    "ENTRY_POINT_ID",
    "AlgorithmReturnData",
    "DefaultTrainable",
    "StrictAlgorithmReturnData",
    "create_default_trainable",
    "default_trainable",
    "episode_iterator",
    "get_run_id",
    "get_trainable_name",
    "is_pbar",
    "nice_logger",
    "run_tune",
    "seed_everything",
    "trial_name_creator",
]


logger = nice_logger(__name__, level=os.environ.get("RAY_UTILITIES_LOG_LEVEL", "DEBUG"))
logger.info("Ray utilities imported. Run ID: %s", _RUN_ID)
logger.debug("Ray utilities logger debug level set")

# suppress a deprecation warning from ray, by creating a RLModuleConfig once
try:
    from ray.rllib.core.rl_module.rl_module import RLModuleConfig
except ImportError:  # might not exist anymore in the future
    pass
else:
    import logging

    try:
        from ray._common.deprecation import logger as __deprecation_logger  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError:
        from ray.rllib.utils.deprecation import logger as __deprecation_logger  # pyright: ignore[reportMissingImports]

    # This suppresses a deprecation warning from RLModuleConfig
    __old_level = __deprecation_logger.getEffectiveLevel()
    __deprecation_logger.setLevel(logging.ERROR)
    RLModuleConfig()
    __deprecation_logger.setLevel(__old_level)
    del __deprecation_logger
    del logging
    del RLModuleConfig

if "RAY_UTILITIES_NO_MONITOR" not in os.environ:
    os.environ.setdefault("RAY_UTILITIES_NO_MONITOR", "1")
else:
    atexit.register(shutdown_monitor)


def get_runtime_env() -> _RuntimeEnv:
    """Get the runtime environment for Ray tasks and actors."""
    if os.environ.get("_RAY_UTILITIES_RUNTIME_ENV_LOADED", "0") != "1":
        working_dir = Path("outputs/shared/.ray_working_dir").as_posix()
        original_working_dir = os.getcwd()
    else:
        working_dir = None
        original_working_dir = None
    import ray

    if ray.is_initialized() and (runtime_context := ray.get_runtime_context()).runtime_env:
        logger.info("Having access to runtime context: %s", vars(runtime_context))

    # S3 bucket configuration for shared storage (optional)
    # Set RAY_UTILITIES_S3_BUCKET environment variable to enable, e.g.:
    # export RAY_UTILITIES_S3_BUCKET="s3://your-bucket-name/ray-results"
    s3_upload_path = os.environ.get("RAY_UTILITIES_S3_BUCKET")

    # TUNE_RESULT_BUFFER_LENGTH <= trial.checkpoint_freq
    # When pbt should not overstep perturbation interval
    # Checkpoints are also done every 65_536 steps, but at most every 24 iteration
    # normal train (need to end on a checkpoint (min(8-24, checkpoint_frequency)
    # no checkpoints: (no checkpoints, can do 9 buffers)
    # batch_size  | frequency  | iterations to perturbation interval
    # -------------------------------------------------------
    # 64          | 1024        |  2048
    # 128         | 512         |  1024
    # 256         | 256         |  512
    # 512         | 128         |  256
    # 1024        | 64          |  128
    # 2048        | 32          |  64
    # 4096        | 24          |  32
    # 8192        | 24          |  16
    # 16384       | 24          |  8

    try_ = 0
    while try_ < 2:
        try_ += 1
        try:
            runtime_env = _RuntimeEnv(
                # When using working_dir it will be found in TUNE_ORIG_WORKING_DIR
                working_dir=working_dir,
                env_vars={
                    "RAY_UTILITIES_NEW_LOG_FORMAT": "1",
                    "COMET_OFFLINE_DIRECTORY": COMET_OFFLINE_DIRECTORY,
                    "RAY_UTILITIES_SET_COMET_DIR": "0",  # do not warn on remote
                    "RAY_UTILITIES_NO_MONITOR": "1",
                    "ENTRY_POINT": ENTRY_POINT,
                    "RUN_ID": get_run_id(),
                    "RAY_UTILITIES_INITIALIZATION_TIMESTAMP": str(RAY_UTILITIES_INITIALIZATION_TIMESTAMP),
                    "_RAY_UTILITIES_RUNTIME_ENV_LOADED": "1",
                    "ORIGINAL_WORKING_DIR": (
                        original_working_dir
                        if original_working_dir is not None
                        else os.environ.get("ORIGINAL_WORKING_DIR", "original_working_dir_unknown")
                    ),
                    "COMET_GIT_DIRECTORY": os.environ.get(
                        "COMET_GIT_DIRECTORY",
                        (
                            original_working_dir
                            if original_working_dir is not None
                            else os.environ.get(
                                "ORIGINAL_WORKING_DIR", os.environ.get("TUNE_ORIG_WORKING_DIR", os.getcwd())
                            )
                        ),
                    ),
                },
                # pip=["s3fs", "boto3"] if s3_upload_path else None,
            )
        except ValueError as e:
            if "not a valid path" in str(e):
                logger.exception("Got ValueError when creating runtime_env, retrying without working_dir")
                working_dir = None
                continue
            raise
        else:
            return runtime_env
    raise RuntimeError("Failed to create runtime_env after retries")


if not TYPE_CHECKING:
    del Any
del TYPE_CHECKING, atexit
