"""Constants determined at runtime

Furthermore constants are added to os.environ to be carried over by ray to remote
processes.
"""

import hashlib
import logging
import os
import sys
import time
import warnings
from pathlib import Path

_logger = logging.getLogger(__name__)

__all__ = [
    "COMET_OFFLINE_DIRECTORY",
    "ENTRY_POINT",
    "ENTRY_POINT_ID",
    "RAY_UTILITIES_INITIALIZATION_TIMESTAMP",
    "RUN_ID",
]

RAY_UTILITIES_INITIALIZATION_TIMESTAMP = float(os.environ.get("RAY_UTILITIES_INITIALIZATION_TIMESTAMP", time.time()))
"""float: Unix timestamp of when the Ray Utilities package was first imported.

Useful for tracking package initialization time and calculating elapsed time since import.
"""

os.environ["RAY_UTILITIES_INITIALIZATION_TIMESTAMP"] = str(RAY_UTILITIES_INITIALIZATION_TIMESTAMP)

# Constant for one execution

ENTRY_POINT: str = os.environ.get(
    "ENTRY_POINT",
    (sys.argv[0] if len(sys.argv) > 0 and (sys.argv[0] and not sys.argv[0].endswith("ipython")) else "__console__"),
)
"""The entry point script's filename, i.e. sys.argv[0] or "__console__" if not available."""

os.environ["ENTRY_POINT"] = ENTRY_POINT

ENTRY_POINT_ID: str = hashlib.blake2b(
    os.path.basename(ENTRY_POINT).encode(), digest_size=3, usedforsecurity=False
).hexdigest()
"""Hash of the entry point script's filename, i.e. sys.argv[0]'s basename"""
# Deterministic no need to write to environ

RUN_ID = os.environ.get(
    "RUN_ID",
    (
        ENTRY_POINT_ID
        + time.strftime("%y%m%d%H%M", time.localtime(RAY_UTILITIES_INITIALIZATION_TIMESTAMP))
        + hashlib.blake2b(os.urandom(8) + ENTRY_POINT_ID.encode(), digest_size=2, usedforsecurity=False).hexdigest()
        + "3"
    ),
)
"""
A short partly random created UUID for the current execution. Only containing numbers and letters.
It is build as: <6 chars entry_point_id> + <datetime as yymmddHHMM> + <4 chars random> + <version char in base62>.
It is 6 + 10 + 4 + 1 = 21 characters long.

It can be used to more easily identify trials that have the same entry point and were run
during the same execution.

The last character is the version of the run_id format. It is currently "3".
"""

os.environ["RUN_ID"] = RUN_ID

_COMET_OFFLINE_DIRECTORY_SUGGESTION = (
    Path("../")
    / "outputs"
    / ".cometml-runs"
    / (
        (
            Path(ENTRY_POINT).stem
            + time.strftime(r"_%Y-%m-%d_%H-%M-%S", time.localtime(RAY_UTILITIES_INITIALIZATION_TIMESTAMP))
        )
        if os.environ.get("RAY_UTILITIES_SEPARATE_COMET_LOG_DIRS", "1") == "1"
        else ""
    )
).resolve()
_COMET_OFFLINE_DIRECTORY_SUGGESTION_STR = str(_COMET_OFFLINE_DIRECTORY_SUGGESTION)

# If we run on remote this warning will trigger; but we do not want to change the remote
if (
    os.environ.get("COMET_OFFLINE_DIRECTORY", _COMET_OFFLINE_DIRECTORY_SUGGESTION_STR)
    != _COMET_OFFLINE_DIRECTORY_SUGGESTION_STR
    and "RAY_UTILITIES_SET_COMET_DIR" not in os.environ
):
    # This error might appear during tests, entry point first is test script then worker script.
    _logger.warning(
        "COMET_OFFLINE_DIRECTORY already set to: %s. Overwriting it with %s. "
        "Set RAY_UTILITIES_SET_COMET_DIR=0 or 1 to disable directory change or to silence this warning.",
        os.environ.get("COMET_OFFLINE_DIRECTORY"),
        _COMET_OFFLINE_DIRECTORY_SUGGESTION_STR,
    )
if os.environ.get("RAY_UTILITIES_SET_COMET_DIR", "1").lower() not in ("0", "false", "off"):
    os.environ["COMET_OFFLINE_DIRECTORY"] = _COMET_OFFLINE_DIRECTORY_SUGGESTION_STR

if "COMET_OFFLINE_DIRECTORY" not in os.environ:
    warnings.warn(
        "COMET_OFFLINE_DIRECTORY is not set in os.environ and RAY_UTILITIES_SET_COMET_DIR is disabled. "
        "Comet ML offline experiments may not be saved to a persistent directory.",
        stacklevel=1,
    )

COMET_OFFLINE_DIRECTORY = os.environ.get("COMET_OFFLINE_DIRECTORY", _COMET_OFFLINE_DIRECTORY_SUGGESTION_STR)
"""str: Directory path for storing offline Comet ML experiments.

This directory is where Comet ML stores experiment archives when running in offline mode.
The default location is ``../outputs/.cometml-runs/<entry_point_stem>_<timestamp>``
relative to the current working directory.
The subdir is omitted if RAY_UTILITIES_SEPARATE_COMET_LOG_DIRS is set to anything other than "1".

The location can be changed by setting the ``COMET_OFFLINE_DIRECTORY`` environment variable.
Note, when COMET_OFFLINE_DIRECTORY is already present a UserWarning is logged unless
RAY_UTILITIES_SET_COMET_DIR is set as well.
RAY_UTILITIES_SET_COMET_DIR=1 (default) will set COMET_OFFLINE_DIRECTORY to the default location as describes above,
RAY_UTILITIES_SET_COMET_DIR=0 will leave COMET_OFFLINE_DIRECTORY unchanged.

See Also:
    :class:`ray_utilities.comet.CometArchiveTracker`: For managing offline experiments
"""

_logger.info("Using COMET_OFFLINE_DIRECTORY: %s", COMET_OFFLINE_DIRECTORY)
