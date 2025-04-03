from __future__ import annotations
import logging
import colorlog

def nicer_logging(logger: logging.Logger | str, level: int | str | None= None):
    """Modifies the logger to have a colored formatter and set the level."""
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if level is not None:
        logger.setLevel(level)
    if logger.hasHandlers():
        logger.warning("Making a richer logger, but logger already has handlers, consider removing them first.")
    utilities_handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s][ %(filename)s:%(lineno)d, %(funcName)s] :%(reset)s %(message)s"
    )
    utilities_handler.setFormatter(formatter)
    logger.addHandler(utilities_handler)
    return logger
