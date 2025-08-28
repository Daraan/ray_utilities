"""Argument processing utilities for Ray Tune experiments and logging callbacks.

This module provides utilities for cleaning and processing command-line arguments
before passing them to logging callbacks like Wandb and Comet ML. It helps filter
out arguments that are not relevant for experiment tracking or could cause issues
with logging frameworks.

The main purpose is to remove framework-specific configuration arguments (like
logging settings, parallelization flags, etc.) that should not be logged as
experiment hyperparameters.

Key Components:
    :data:`LOG_IGNORE_ARGS`: Arguments to exclude from logging
    :func:`remove_ignored_args`: Filter arguments for clean logging

Example:
    Cleaning arguments before logging to Wandb::
    
        from ray_utilities.callbacks import remove_ignored_args
        import argparse
        
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--wandb", action="store_true")  # Should not be logged
        parser.add_argument("--silent", action="store_true")  # Should not be logged
        args = parser.parse_args()
        
        # Clean arguments for logging
        clean_args = remove_ignored_args(args)
        # Only {"lr": 0.001} will be passed to Wandb, not the logging flags
        
    Using with Typed Argument Parser (TAP)::
    
        from tap import Tap
        from ray_utilities.callbacks import remove_ignored_args
        
        class MyArgs(Tap):
            lr: float = 0.001
            wandb: bool = False
            comet: bool = False
            
        args = MyArgs().parse_args()
        experiment_params = remove_ignored_args(args)
        # Logging flags are automatically filtered out

Constants:
    :data:`LOG_IGNORE_ARGS`: Tuple of argument names to exclude from logging

See Also:
    :mod:`ray_utilities.config.typed_argument_parser`: For defining experiment arguments
    :class:`ray.air.integrations.wandb.WandbLoggerCallback`: Wandb integration
    :class:`ray.air.integrations.comet.CometLoggerCallback`: Comet ML integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, TypeVar, cast, overload

if TYPE_CHECKING:
    import argparse

    from tap import Tap

__all__ = [
    "LOG_IGNORE_ARGS",
    "remove_ignored_args",
]

_K = TypeVar("_K")
_V = TypeVar("_V")

LOG_IGNORE_ARGS = (
    "wandb",
    "comet",
    "not_parallel",
    "silent",
    "tags",
    "log_level",
    "use_comet_offline",
)
"""tuple[str, ...]: Argument names that should not be processed by logging callbacks.

These arguments control logging and execution behavior rather than experiment
hyperparameters, so they should be excluded when passing parameters to experiment
tracking platforms like Wandb or Comet ML.

Excluded arguments:
    - **wandb**: Enable/disable Wandb logging
    - **comet**: Enable/disable Comet ML logging  
    - **not_parallel**: Disable parallel execution
    - **silent**: Suppress output/logging
    - **tags**: Experiment tags (handled separately by loggers)
    - **log_level**: Logging verbosity level
    - **use_comet_offline**: Use Comet offline mode

See Also:
    :func:`remove_ignored_args`: Uses this tuple for filtering
"""


@overload
def remove_ignored_args(  # pyright: ignore[reportOverlappingOverload]
    args: dict[_K, _V | Callable], *, remove: Iterable[_K | str] = LOG_IGNORE_ARGS
) -> dict[_K, _V]: ...


@overload
def remove_ignored_args(
    args: Tap | argparse.Namespace | Any, *, remove: Iterable[Any] = LOG_IGNORE_ARGS
) -> dict[str, Any]: ...


def remove_ignored_args(
    args: Mapping[_K, Any] | Tap | argparse.Namespace | Any, *, remove: Iterable[_K | str] = LOG_IGNORE_ARGS
) -> Mapping[_K, Any] | dict[str, Any]:
    """Remove logging-specific arguments from experiment parameters.

    This function filters out arguments that control logging and execution behavior
    rather than experiment hyperparameters. It's designed to clean parameter
    dictionaries before passing them to experiment tracking platforms.

    Args:
        args: Arguments to process. Can be:
            
            - Dictionary of parameters
            - :class:`argparse.Namespace` from argument parser
            - :class:`tap.Tap` typed argument parser instance
            - Any object with a ``as_dict()`` method or ``vars()`` support
            
        remove: Iterable of argument names to remove. Defaults to 
            :data:`LOG_IGNORE_ARGS`.

    Returns:
        A dictionary with the specified arguments removed. Callable values
        are also filtered out automatically.

    Example:
        With argparse::
        
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> args = parser.parse_args(["--lr", "0.001", "--wandb"])
        >>> clean_args = remove_ignored_args(args)
        >>> "wandb" in clean_args
        False
        >>> clean_args["lr"]
        0.001
        
        With custom removal list::
        
        >>> params = {"lr": 0.001, "debug": True, "model": "transformer"}
        >>> clean_params = remove_ignored_args(params, remove=["debug"])
        >>> "debug" in clean_params
        False
        
        With TAP (Typed Argument Parser)::
        
        >>> from tap import Tap
        >>> class Args(Tap):
        ...     lr: float = 0.001
        ...     wandb: bool = False
        >>> args = Args()
        >>> clean_args = remove_ignored_args(args)
        >>> "wandb" in clean_args
        False

    Note:
        - Callable values are automatically filtered out regardless of their key names
        - The original argument object/dictionary is not modified
        - Missing keys in the removal list are silently ignored

    See Also:
        :data:`LOG_IGNORE_ARGS`: Default list of arguments to remove
        :mod:`ray_utilities.config.typed_argument_parser`: For structured argument definition
    """
    if not isinstance(args, (dict, Mapping)):
        if hasattr(args, "as_dict"):  # Tap
            args = args.as_dict()
        else:
            args = vars(args)
        args = cast("dict[str, Any]", args)
    return {k: v for k, v in args.items() if k not in remove and not callable(v)}
