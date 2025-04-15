from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Callable, ParamSpec, TypeVar

    _P = ParamSpec("_P")
    _R = TypeVar("_R")

def type_grad_and_value(func: Callable[_P, _R]) -> Callable[_P, tuple[_R, tuple[Any, ...]]]:
    """
    jax.grad_and_value also returns the gradients of the function in an extra tuple
    Use this wrapper to keep the signature intact and return the gradients as a second return parameter
    """
    return func  # type: ignore[return-type-mismatch]
