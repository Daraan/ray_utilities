from __future__ import annotations
from typing import Any, Iterable, TypeVar, cast, overload, TYPE_CHECKING


if TYPE_CHECKING:
    from tap import Tap
    import argparse

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
"""
Keys that should not be processed by logging callbacks
"""


@overload
def remove_ignored_args(  # pyright: ignore[reportOverlappingOverload]
    args: dict[_K, _V], *, remove: Iterable[_K | str] = LOG_IGNORE_ARGS
) -> dict[_K, _V]: ...


@overload
def remove_ignored_args(
    args: Tap | argparse.Namespace | Any, *, remove: Iterable[Any] = LOG_IGNORE_ARGS
) -> dict[str, Any]: ...


def remove_ignored_args(
    args: dict[_K, _V] | Tap | argparse.Namespace | Any, *, remove: Iterable[_K | str] = LOG_IGNORE_ARGS
) -> dict[_K, _V] | dict[str, Any]:
    """
    Remove ignored keys from args

    Args:
        args: Arguments to process

    Returns:
        dict: Arguments with ignored keys removed
    """
    if not isinstance(args, dict):
        if hasattr(args, "as_dict"):  # Tap
            args = args.as_dict()
        else:
            args = vars(args)
        args = cast(dict[str, Any], args)
    return {k: v for k, v in args.items() if k not in remove}
