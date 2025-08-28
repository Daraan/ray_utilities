"""Temporary directory management for Ray Utilities media and logging operations.

This module provides a centralized temporary directory for storing transient files
such as logged videos, checkpoints, and other media files that need temporary
storage before being uploaded to experiment tracking services like Weights & Biases
or Comet ML.

The module automatically creates and manages a temporary directory with proper
cleanup registration, ensuring resources are freed when the application exits.
It supports both mounted memory directories for performance and standard temporary
directories as fallback.

Key Components:
    - ``TEMP_DIR``: Managed temporary directory instance
    - ``TEMP_DIR_PATH``: Absolute path to the temporary directory
    - Automatic cleanup registration for proper resource management

Usage:
    The temporary directory is automatically created when the module is imported
    and can be used throughout the application for storing temporary files.

Example:
    >>> from ray_utilities.temp_dir import TEMP_DIR_PATH
    >>> video_path = os.path.join(TEMP_DIR_PATH, "episode_video.mp4")
    >>> # Use video_path for temporary video storage

Note:
    The module prefers mounted memory directories (``temp_dir``) when available
    for improved I/O performance, falling back to system temporary directories
    otherwise.
"""

import atexit
import os
import tempfile

if os.path.exists("temp_dir"):  # mounted memory
    TEMP_DIR = tempfile.TemporaryDirectory("_utility-temp", dir="temp_dir")
else:
    TEMP_DIR = tempfile.TemporaryDirectory("_utility-temp")
TEMP_DIR_PATH = os.path.abspath(TEMP_DIR.name)


def _cleanup_media_tmp_dir() -> None:
    atexit.register(TEMP_DIR.cleanup)
