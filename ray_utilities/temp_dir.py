"""Temporary directory management for Ray Utilities operations.

Provides a centralized temporary directory for transient files such as videos,
checkpoints, and media files that need temporary storage before upload to
experiment tracking services.

The module creates a managed temporary directory with automatic cleanup and
prefers mounted memory directories when available for improved performance.

Attributes:
    TEMP_DIR: Managed temporary directory instance with automatic cleanup
    TEMP_DIR_PATH: Absolute path to the temporary directory

Example:
    >>> from ray_utilities.temp_dir import TEMP_DIR_PATH
    >>> video_path = os.path.join(TEMP_DIR_PATH, "episode_video.mp4")
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
