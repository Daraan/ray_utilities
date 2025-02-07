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
