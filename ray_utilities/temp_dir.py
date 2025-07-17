"""
This module provides a temporary directory for utility functions, e.g. to store
logged videos to be uploaded to wandb/comet.
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
