from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol, TypedDict, overload

import numpy as np

if TYPE_CHECKING:
    from typing import ParamSpec, TypeVar

    T = TypeVar("T")
    P = ParamSpec("P")

    def cache(func: Callable[P, T], /) -> Callable[P, T]: ...
else:
    from functools import cache

logger = logging.getLogger(__name__)

MIN_DYNAMIC_BATCH_SIZE = 16
MAX_DYNAMIC_BATCH_SIZE = 16384
_MIN_DEFAULT_BATCH_SIZE = 32
_MAX_DEFAULT_BATCH_SIZE = 8192


class UpdateNStepsArgs(Protocol):
    total_steps: int
    n_envs: int
    dynamic_buffer: bool
    static_batch: bool


# NOTE: SYMPOL keeps a copy of this function in the repo (standalone)
def update_buffer_and_rollout_size(
    args: UpdateNStepsArgs,
    *,
    initial_steps: int,
    global_step: int,
    num_increase_factors: int = 8,
    accumulate_gradients_every_initial: int,
):
    """
    Calculates a new rollout and batch size

    Afterwards create Rollout with `n_steps`
    `if args.dynamic_buffer or not args.static_batch:` recalculate
    Then if n_steps != n_steps_old: -> create rollout

    Attention:
        The default value of num_increase_factors=8, does not match with the default values of
        min_size=32 and max_size=8192 in the other functions of this module, as those result in
        9 different values; use num_increase_factors=9 to match the other functions.
    """
    # increase_index = global_step // (args.total_steps//sum(increase_factor_list))
    if global_step + 1 > args.total_steps:
        global_step = args.total_steps  # prevent explosion; limit factor to 128
    increase_factor = int(
        2 ** (np.ceil((((global_step + 1) * num_increase_factors) / (1 + args.total_steps))) - 1)
    )  # int(increase_factor_list_long[increase_index])
    increase_factor_batch = int(
        2 ** (np.ceil((((global_step + 1) * num_increase_factors) / (1 + args.total_steps))) - 1)
    )  # int(increase_factor_list_long[increase_index])
    if args.dynamic_buffer:
        n_steps = initial_steps * increase_factor
    else:
        n_steps = initial_steps
    if not args.static_batch:
        accumulate_gradients_every = int(accumulate_gradients_every_initial * increase_factor_batch)
    else:
        accumulate_gradients_every = int(accumulate_gradients_every_initial)
    # DYNAMIC_BATCH_SIZE
    batch_size = int(args.n_envs * n_steps)  # XXX: Get rid of n_envs; samples_per_step
    # n_iterations = args.total_steps // batch_size
    # eval_freq = max(args.eval_freq // batch_size, 1)
    # logger.debug("updating buffer after step %d / %s to %s. Initial size: %s", global_step, args.total_steps, batch_size, initial_steps)

    return (
        min(MAX_DYNAMIC_BATCH_SIZE, max(MIN_DYNAMIC_BATCH_SIZE, batch_size)),
        accumulate_gradients_every,
        min(MAX_DYNAMIC_BATCH_SIZE, max(MIN_DYNAMIC_BATCH_SIZE, n_steps)),
    )


class SplitBudgetReturnDict(TypedDict):
    total_steps: int
    step_sizes: list[int]
    iterations_per_step_size: list[int]
    total_iterations: int
    """Sum of iterations_per_step_size * step_sizes."""


@cache
def split_timestep_budget(
    total_steps: int = 1_000_000,
    *,
    min_size: int = _MIN_DEFAULT_BATCH_SIZE,
    max_size: int = _MAX_DEFAULT_BATCH_SIZE,
    assure_even: bool = True,
) -> SplitBudgetReturnDict:
    """
    Split the budget into smaller chunks.

    Args:
        total_steps: The total number of steps to split. If assure_even is True, this will be the lower
            bound of steps to assure an en even distribution of steps and iterations from min_size to max_size.
        min_size: The minimum size of the step sizes.
        max_size: The maximum size of the step sizes.
        assure_even: If True, the total_steps will be adjusted to be divisible by the number of increases.
            This will lead to an even distribution of steps and iterations.
            If False, the total_steps will not be adjusted, which may lead to uneven distribution of steps.
    Returns:
        A dictionary with the following keys:
            - total_steps: The total number of steps, adjusted if assure_even is True.
            - step_sizes: A list of step sizes.
            - iterations_per_step_size: A list of iterations per step size.
    """
    next_steps = min_size
    step_sizes: list[int] = []
    while next_steps <= max_size:
        step_sizes.append(next_steps)
        next_steps *= 2
    increases = len(step_sizes)
    if assure_even:
        n = 1
        steps = 0
        while steps < total_steps:
            iterations_per_step_size = np.array([n * 2**i for i in range(increases)][::-1])
            steps_per_increase = iterations_per_step_size * step_sizes  # constant * increases
            steps = sum(steps_per_increase)
            n += 1
        total_steps = steps

    steps_per_increase = total_steps / increases
    if not steps_per_increase.is_integer():
        logger.warning(
            "Total steps %d is not divisible by the number of increases %d. "
            "This may lead to uneven distribution of steps. "
            "Consider increasing the total_steps, start_steps or max_size.",
            total_steps,
            increases,
        )
    iterations_per_step_size_raw = [steps_per_increase / size for size in step_sizes]
    iterations_per_step_size = [max(1, round(i)) for i in iterations_per_step_size_raw]
    if any(not f.is_integer() for f in iterations_per_step_size_raw):
        logger.warning(
            "Some iterations per step size are not integers. This may lead to uneven distribution of steps.",
        )
    return {
        "total_steps": total_steps,
        "step_sizes": step_sizes,
        "iterations_per_step_size": iterations_per_step_size,
        "total_iterations": sum(iterations_per_step_size),
    }


@overload
def calculate_iterations(
    *,
    dynamic_buffer: Literal[False],
    batch_size: int,
    total_steps: int = 1_000_000,
    min_size: int = _MIN_DEFAULT_BATCH_SIZE,
    max_size: int = _MAX_DEFAULT_BATCH_SIZE,
    assure_even: bool = True,
) -> int: ...


@overload
def calculate_iterations(
    *,
    dynamic_buffer: Literal[True],
    batch_size: Optional[int] = None,
    total_steps: int = 1_000_000,
    min_size: int = 32,
    max_size: int = 8192,
    assure_even: bool = True,
) -> int: ...


def calculate_iterations(
    *,
    dynamic_buffer: bool,
    batch_size: Optional[int] = None,
    total_steps: int = 1_000_000,
    min_size: int = _MIN_DEFAULT_BATCH_SIZE,
    max_size: int = _MAX_DEFAULT_BATCH_SIZE,
    assure_even: bool = True,
) -> int:
    """
    Calculate the number of iterations based on the budget of total steps and the batch size.
    Args:
        dynamic_buffer: If True, the budget will be split into smaller chunks.
        batch_size: The batch size to use for the calculation of the iterations if dynamic_buffer is False.
        total_steps: The total number of steps to split. Default is 1_000_000.
        min_size: The minimum size of the step sizes. Not used if dynamic_buffer as well as assure_even is False.
            Default is MIN_DYNAMIC_BATCH_SIZE.
        max_size: The maximum size of the step sizes.
            Default is MAX_DYNAMIC_BATCH_SIZE.
        assure_even: If True, the total_steps will be adjusted to be evenly divisible by the number of increases
            from min_size to max_size.
            If False, the total_steps will not be adjusted, which may lead to uneven distribution of steps.

    See Also:
        split_timestep_budget: For more information on how the budget is split.
    """
    if not dynamic_buffer:
        if batch_size is None:
            raise ValueError("batch_size must be provided when dynamic_buffer is False")
        if not assure_even:
            return math.ceil(total_steps / batch_size)
    budget = split_timestep_budget(
        total_steps=total_steps,
        min_size=min_size,
        max_size=max_size,
        assure_even=assure_even,
    )
    if not dynamic_buffer:
        # Return based on even distribution of steps, even if not using dynamic buffer
        return math.ceil(budget["total_steps"] / batch_size)  # pyright: ignore[reportOperatorIssue]
    iterations = sum(budget["iterations_per_step_size"])
    return iterations


def calculate_steps(iterations, *, total_steps_default: int = 1_000_000, min_step_size: int, max_step_size: int) -> int:
    """
    This calculates the number of steps taken when limiting the number of iterations.

    Use when passing --iterations to calcualte the correct value for the progress bar.
    Args:
        iterations: The number of iterations to limit the steps to.
        total_steps_default: The default total number of steps to use for the complete budget.
        min_step_size: The minimum step size to use for the budget.
        max_step_size: The maximum step size to use for the budget.
    """
    budget = split_timestep_budget(
        total_steps=total_steps_default,
        min_size=min_step_size,
        max_size=max_step_size,
        assure_even=True,
    )
    iterations_left = iterations
    steps_taken = 0
    for i, step_size in enumerate(budget["step_sizes"]):
        steps_taken += step_size * min(iterations_left, budget["iterations_per_step_size"][i])
        iterations_left -= budget["iterations_per_step_size"][i]
        if iterations_left <= 0:
            break
    else:
        # iterations > budget["total_iterations"]
        steps_taken += iterations_left * budget["step_sizes"][-1]
    return steps_taken
