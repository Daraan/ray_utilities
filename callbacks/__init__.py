from __future__ import annotations
from typing import Any, Iterable, TypeVar, overload

__all__ = [
    "LOG_IGNORE_ARGS",
    "remove_ignored_args",
]

_K = TypeVar("_K")
_V = TypeVar("_V")

LOG_IGNORE_ARGS = ("wandb", "comet", "not_parallel", "silent")
"""
Keys that should not be processed by logging callbacks
"""


@overload
def remove_ignored_args(  # pyright: ignore[reportOverlappingOverload]
    args: dict[_K, _V], *, remove: Iterable[_K | str] = LOG_IGNORE_ARGS
) -> dict[_K, _V]: ...


@overload
def remove_ignored_args(args: Any, *, remove: Iterable[Any] = LOG_IGNORE_ARGS) -> dict[str, Any]: ...


def remove_ignored_args(
    args: dict[_K, _V] | Any, *, remove: Iterable[_K | str] = LOG_IGNORE_ARGS
) -> dict[_K, _V] | dict[str, Any]:
    """
    Remove ignored keys from args

    Args:
        args (dict): Arguments to process

    Returns:
        dict: Arguments with ignored keys removed
    """
    if not isinstance(args, dict):
        args = vars(args)
    return {k: v for k, v in args.items() if k not in remove}
