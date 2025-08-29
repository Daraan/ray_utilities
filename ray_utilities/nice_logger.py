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
import colorlog


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
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Bold red
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if level is not None:
        logger.setLevel(level)
    if logger.hasHandlers():
        logger.warning(
            "Making a richer logger, but logger %s already has handlers, consider removing them first.", logger
        )
    utilities_handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s][ %(filename)s:%(lineno)d, %(funcName)s] :%(reset)s %(message)s"
    )
    utilities_handler.setFormatter(formatter)
    logger.addHandler(utilities_handler)
    return logger
