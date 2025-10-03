"""Callbacks for Ray RLlib algorithms and Ray Tune experiments.

This module contains callback classes for Ray RLlib algorithms and Ray Tune tuners,
plus utilities for cleaning command-line arguments before passing them to logging
callbacks.

Main Components:
    - Algorithm callbacks in :mod:`~ray_utilities.callbacks.algorithm`
    - Tuner callbacks in :mod:`~ray_utilities.callbacks.tuner`
    - :func:`remove_ignored_args`: Filter arguments for experiment logging
    - :data:`LOG_IGNORE_ARGS`: Arguments to exclude from logging
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, TypeVar, cast, overload

if TYPE_CHECKING:
    import argparse

    from tap import Tap

__all__ = ["LOG_IGNORE_ARGS", "remove_ignored_args"]

_K = TypeVar("_K")
_V = TypeVar("_V")

LOG_IGNORE_ARGS = (
    "wandb",
    "comet",
    "not_parallel",
    "silent",
    "tags",  # handles separately by callbacks
    "use_comet_offline",
)
"""
Arguments that control logging and execution behavior rather than experiment hyperparameters.
These attributes of the `DefaultArgumentParser` or keys in the `Trainable`'s `config`
which should not processed by logging callbacks.
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

    Filters out arguments that control logging and execution behavior rather than
    experiment hyperparameters. Designed for cleaning parameters before passing
    to experiment tracking platforms.

    Args:
        args: Arguments to process (dict, :class:`argparse.Namespace`, :class:`tap.Tap`,
            or object with ``as_dict()`` method).
        remove: Argument names to remove. Defaults to :data:`LOG_IGNORE_ARGS`.

    Returns:
        Dictionary with specified arguments removed. Callable values are also filtered out.

    Example:
        >>> args = argparse.Namespace(lr=0.001, wandb=True, epochs=10)
        >>> clean_args = remove_ignored_args(args)
        >>> "wandb" in clean_args
        False
        >>> clean_args["lr"]
        0.001

    Note:
        The original argument object is not modified.
    """
    if not isinstance(args, (dict, Mapping)):
        if hasattr(args, "as_dict"):  # Tap
            args = args.as_dict()
        else:
            args = vars(args)
        args = cast("dict[str, Any]", args)
    return {k: v for k, v in args.items() if k not in remove and not callable(v)}
