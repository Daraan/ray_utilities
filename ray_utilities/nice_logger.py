"""Enhanced logging utilities with colored output for better debugging experience.

This module provides the :func:`nice_logger` function to create loggers with colored,
formatted output that makes debugging Ray Tune and RLlib experiments more pleasant
and informative.

Example:
    Basic usage for setting up a colored logger::

        from ray_utilities.nice_logger import nice_logger

        logger = nice_logger(__name__, level="DEBUG")
        logger.info("This will be colored and nicely formatted")
        logger.warning("Warnings stand out with colors")
        logger.error("Errors are clearly visible")

See Also:
    :mod:`colorlog`: The underlying library used for colored logging
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from contextlib import contextmanager

import colorlog
from typing_extensions import Sentinel

# pyright: enableExperimentalFeatures=true
_NO_PACKAGE = Sentinel("_NO_PACKAGE")

# --- Custom log levels ---
IMPORTANT_INFO = 25  # Between INFO (20) and WARNING (30)
IMPORTANT_WARNING = 35  # Between WARNING (30) and ERROR (40)

logging.addLevelName(IMPORTANT_INFO, "IMPORTANT_INFO")
logging.addLevelName(IMPORTANT_WARNING, "IMPORTANT_WARNING")


class ImportantLogger(logging.Logger):
    """
    Implements two custom log levels: IMPORTANT_INFO and IMPORTANT_WARNING.

    Its methods are also registered to the standard logging.Logger class.
    Use this class via `typing.cast("ImportantLogger", normal_logger).important_info(...)`
    or ImportantLogger.important_info(logger, ...).
    """

    # Can be used in a static method like manner with a normal logger instance
    def important_info(logger: logging.Logger, msg: str, *args, stacklevel=1, **kwargs):  # pyright: ignore[reportSelfClsParameterName] # noqa: N805
        """Log 'msg % args' with severity 'IMPORTANT_INFO'."""
        if logger.isEnabledFor(IMPORTANT_INFO):
            logger._log(IMPORTANT_INFO, msg, args, stacklevel=stacklevel, **kwargs)

    def important_warning(logger: logging.Logger, msg: str, *args, stacklevel=1, **kwargs):  # pyright: ignore[reportSelfClsParameterName] # noqa: N805
        """Log 'msg % args' with severity 'IMPORTANT_WARNING'."""
        if logger.isEnabledFor(IMPORTANT_WARNING):
            logger._log(IMPORTANT_WARNING, msg, args, stacklevel=stacklevel, **kwargs)


logging.Logger.important_info = ImportantLogger.important_info  # type: ignore
logging.Logger.important_warning = ImportantLogger.important_warning  # type: ignore

try:
    _hostname = socket.gethostname()
except:  # noqa: E722
    _hostname = "unknown_host"
if _hostname in os.environ.get("LOG_HOSTNAME_IGNORE", ""):
    _hostname = ""


def _get_colorlog_formatter() -> colorlog.ColoredFormatter:
    """Returns a ColoredFormatter with custom colors for all log levels, including custom ones."""
    return colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s][" + _hostname + " %(filename)s:%(lineno)d, %(funcName)s] :%(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "IMPORTANT_INFO": "bold_green",
            "WARNING": "yellow",
            "IMPORTANT_WARNING": "bold_yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={},
        style="%",
    )


def nice_logger(logger: logging.Logger | str, level: int | str | None = None) -> logging.Logger:
    """Create or modify a logger with colored formatting and enhanced readability.

    This function sets up a logger with colored output using :mod:`colorlog`, making
    it easier to distinguish between different log levels and trace debugging information
    during Ray Tune experiments and RLlib training.

    Args:
        logger: Either a :class:`logging.Logger` instance or a string name to create
            a new logger.
        level: The logging level to set. Can be an integer (from :mod:`logging` module)
            or a string like ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
            If ``None``, the current level is preserved.

    Returns:
        A configured :class:`logging.Logger` with colored formatting that includes:

        - Color-coded log levels for easy identification
        - File name, line number, and function name for debugging
        - Consistent formatting across all log messages

    Warning:
        If the logger already has handlers, a warning will be logged suggesting
        to remove them first to avoid duplicate log messages.

    Example:
        >>> logger = nice_logger("my_experiment", level="DEBUG")
        >>> logger.info("Starting Ray Tune experiment")
        [INFO][ my_file.py:42, main_function] : Starting Ray Tune experiment

        Using with an existing logger::

        >>> import logging
        >>> existing_logger = logging.getLogger("ray.rllib")
        >>> enhanced_logger = nice_logger(existing_logger, level="WARNING")

    Note:
        The colored formatter includes the following information:

        - **Log level** (colored based on severity)
        - **Filename and line number** where the log was called
        - **Function name** where the log was called
        - **Log message** content

        Colors help distinguish between different log levels:
        - DEBUG: Cyan
        - INFO: Green
        - IMPORTANT_INFO: Bold Green
        - WARNING: Yellow
        - IMPORTANT_WARNING: Bold Yellow
        - ERROR: Red
        - CRITICAL: Bold red
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    if isinstance(level, str):
        if sys.version_info >= (3, 11):
            level = logging.getLevelNamesMapping()[level.upper()]
        else:
            level = logging.getLevelName(level.upper())  # pyright: ignore[reportDeprecated]
    if level is not None:
        logger.setLevel(level)
    if logger.hasHandlers():
        logger.warning(
            "Making a richer logger, but logger %s already has handlers, consider removing them first.", logger
        )
    if "SLURM_JOB_ID" in os.environ:
        # On SLURM: use plain, non-colored logging
        utilities_handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s][ %(filename)s:%(lineno)d, %(funcName)s] : %(message)s")
        utilities_handler.setFormatter(formatter)
        logger.addHandler(utilities_handler)
    else:
        utilities_handler = colorlog.StreamHandler()
        formatter = _get_colorlog_formatter()
        utilities_handler.setFormatter(formatter)
        logger.addHandler(utilities_handler)
    return logger


def set_project_log_level(
    pkg_logger: logging.Logger | str, level: int | str, pkg_name: str | None | _NO_PACKAGE = _NO_PACKAGE
) -> None:
    """
    Sets the logging level for a package logger and all its subloggers.

    Args:
        pkg_logger (logging.Logger | str): The package logger instance or its name as a string.
        level (int | str): The logging level to set (e.g., logging.INFO or "INFO").
        pkg_name (str | None | _NO_PACKAGE, optional): The package name. If None, only the given logger is affected.
            If _NO_PACKAGE, the package name is inferred from the logger's name.

    Returns:
        None

    Side Effects:
        - Changes the log level of the specified logger and all its handlers.
        - Recursively sets the log level for all child loggers belonging to the package.
        - Logs an info message when the log level is changed.
        - Logs an exception if a child logger's level cannot be set.

    Raises:
        None
    """
    # Set level on root package logger
    if isinstance(pkg_logger, str):
        pkg_logger = logging.getLogger(pkg_logger)
    if pkg_logger.getEffectiveLevel() not in (level, logging.getLevelName(level)):  # pyright: ignore[reportDeprecated]
        pkg_logger.info(
            "Changing log level of %s logger and all subloggers to %s",
            pkg_logger.name,
            logging.getLevelName(level) if isinstance(level, int) else level,
        )
    pkg_logger.setLevel(level)
    for h in pkg_logger.handlers:
        h.setLevel(level)
    if pkg_name is None:
        return

    if pkg_name is _NO_PACKAGE:
        pkg_name = pkg_logger.name.split(".")[0]

    # Also set level for any already-configured child loggers (skip placeholders)
    # Use copy for rare case during debugging that dict changes during t
    for name, lg in logging.Logger.manager.loggerDict.copy().items():
        if not isinstance(lg, logging.Logger):
            continue
        if lg is pkg_logger:
            continue
        if name == pkg_name or name.startswith(pkg_name + "."):
            try:
                lg.setLevel(level)
                for h in lg.handlers:
                    h.setLevel(level)
            except Exception:  # noqa: BLE001
                lg.exception("Failed to set level for logger '%s'", name)
                # be conservative: ignore any logger that can't be adjusted
                continue


@contextmanager
def change_log_level(logger: logging.Logger, new_level: logging._Level):
    """Context manager to temporarily change a logger's level."""
    old_level = logger.getEffectiveLevel()
    logger.setLevel(new_level)
    try:
        yield
    finally:
        logger.setLevel(old_level)
