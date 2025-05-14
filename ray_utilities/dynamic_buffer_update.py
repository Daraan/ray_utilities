import math
from types import SimpleNamespace
import numpy as np
from typing import Literal, Protocol

import logging

logger = logging.getLogger(__name__)

MIN_DYNAMIC_BATCH_SIZE = 16
MAX_DYNAMIC_BATCH_SIZE = 16384


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
    accumulate_gradients_every_initial: int,
):
    """
    Calculates a new rollout and batch size

    Afterwards create Rollout with `n_steps`
    `if args.dynamic_buffer or not args.static_batch:` recalculate
    Then if n_steps != n_steps_old: -> create rollout
    """
    # increase_index = global_step // (args.total_steps//sum(increase_factor_list))
    if global_step + 1 > args.total_steps:
        global_step = args.total_steps  # prevent explosion; limit factor to 128
    increase_factor = int(
        2 ** (np.ceil((((global_step + 1) * 8) / (1 + args.total_steps))) - 1)
    )  # int(increase_factor_list_long[increase_index])
    increase_factor_batch = int(
        2 ** (np.ceil((((global_step + 1) * 8) / (1 + args.total_steps))) - 1)
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


def calculate_total_steps(*, training_iterations: int, batch_size: int, dynamic_buffer: bool, increases: int = 8):
    """
    Attention:
        Initial steps should be config.train_batch_size_per_learner without prior modifications.

    TODO:
        This should be calculated via the Callback Method that implements the buffer size increase
    """
    if dynamic_buffer:
        batch_sizes: list[int] = dynamic_batch_sizes(batch_size, increases=increases)
        # NOTE: with ceil the iterations are overestimated
        exponents = [int(math.log2(batch_sizes[i] / batch_sizes[0])) for i in range(increases)]
        amount_steps_per_increase = [max(1, math.floor(training_iterations / (2 ** (e + 1)))) for e in exponents]
        # amount_steps_per_increase = [training_iterations / (2 ** (i + 1)) for i in range(increases)]
        iterations_per_increase = f"{min(amount_steps_per_increase)} - {max(amount_steps_per_increase)}"  # debug
        total_steps = int(sum(amount_steps_per_increase[i] * batch_sizes[i] for i in range(increases)))
    else:
        batch_sizes = [batch_size] * increases
        iterations_per_increase = training_iterations / increases
        amount_steps_per_increase = [batch_size * iterations_per_increase] * increases  # sum is an integer
        # Calculate exactly, instead of summing, to avoid floating point errors
        total_steps = training_iterations * batch_size
        if not math.isclose(sum(amount_steps_per_increase), total_steps, abs_tol=1):
            logger.error(
                "Calculation of total_steps is not correct. %s != %s", sum(amount_steps_per_increase), total_steps
            )
    logger.info(
        "Total steps calculated: %d. Step sizes are %s, with %s iterations in between. Steps at each level: %s",
        total_steps,
        batch_sizes,
        iterations_per_increase,
        amount_steps_per_increase,
    )
    return total_steps


def _exponents_for_batch_sizes(increases: int) -> list[int]:
    """
    This will create an array of [-n, ..., n] of length increases.
    For a non symmetric list the the higher number is first added on the positive side.
    For example:
        [-2, -1, 0, 1, 2, 3] for increases = 6
    """
    low = -math.floor((increases - 1) / 2)
    up = math.ceil((increases + 1) / 2)
    exponents = list(range(low, up))
    return exponents


def dynamic_batch_sizes(batch_size: int, *, increases: int) -> list[int]:
    """Returns a list of batch sizes for the given number of increases and the base batch size."""
    exponents = _exponents_for_batch_sizes(increases)
    batch_sizes = [int(2**i * batch_size) for i in exponents]
    if any(size < MIN_DYNAMIC_BATCH_SIZE for size in batch_sizes):
        logger.warning(
            "Automatic reduction of batch size has values below %d. Setting those to %d. "
            "Consider increasing the batch_size",
            MIN_DYNAMIC_BATCH_SIZE,
            MIN_DYNAMIC_BATCH_SIZE,
        )
        batch_sizes = [max(size, MIN_DYNAMIC_BATCH_SIZE) for size in batch_sizes]
    if any(size > MAX_DYNAMIC_BATCH_SIZE for size in batch_sizes):
        logger.warning(
            "Automatic reduction of batch size has values above %d. Setting those to %d. "
            "Consider decreasing the batch_size",
            MAX_DYNAMIC_BATCH_SIZE,
            MAX_DYNAMIC_BATCH_SIZE,
        )
        batch_sizes = [min(size, MAX_DYNAMIC_BATCH_SIZE) for size in batch_sizes]
    return batch_sizes


def _test_iterations(
    iterations_target,
    total_steps,
    batch_sizes,
    *,
    dynamic_buffer: bool,
    dynamic_batch: bool,
    return_correct: Literal["iterations", "total_steps"] = "iterations",
):
    args: UpdateNStepsArgs = SimpleNamespace(
        total_steps=total_steps,
        n_envs=1,
        dynamic_buffer=dynamic_buffer,
        static_batch=dynamic_batch,
    )  # type: ignore[assignment]
    global_step = 0
    n_steps_old = None
    n_steps = batch_sizes[0]
    iterations = 0  # new batch size is always calculated at the end of the report

    total_steps_until_target = None
    batch_size = None
    while global_step < total_steps:
        global_step += n_steps
        if iterations == iterations_target:
            total_steps_until_target = global_step
        iterations += 1
        batch_size, _, n_steps = update_buffer_and_rollout_size(
            args,
            global_step=global_step,
            accumulate_gradients_every_initial=1,
            initial_steps=batch_sizes[0],
        )
        if n_steps_old != n_steps:
            n_steps_old = n_steps
            logger.debug(
                "Updating at step %d / %s (%f%%) (iter: %d/%d) to '%s x %s=%s' | initially (%s), new batch size=%s",
                global_step,
                total_steps,
                round((global_step / total_steps) * 100, 0),
                iterations,
                iterations_target,
                n_steps,
                1,
                n_steps * 1,
                batch_sizes[0],
                batch_size,
            )
    logger.debug(
        "Final step %d / %s (%f%%) (iter: %d/%d) to '%s x %s=%s' | initially (%s), new batch size=%s",
        global_step,
        total_steps,
        round((global_step / total_steps) * 100, 0),
        iterations,
        iterations_target,
        n_steps,
        1,
        n_steps * 1,
        batch_sizes[0],
        batch_size,
    )
    if iterations != iterations_target:
        logger.warning(
            "Iterations %d != %d. Total steps: %d, batch sizes: %s",
            iterations,
            iterations_target,
            total_steps,
            batch_sizes,
        )
    if return_correct == "iterations":
        return iterations, global_step
    return iterations_target, total_steps_until_target if total_steps_until_target is not None else global_step


def calculate_iterations(*, dynamic_buffer: bool, batch_size: int, total_steps: int, increases=8) -> int:
    if not dynamic_buffer:
        return math.ceil(total_steps / batch_size)
    # TODO: Should be implemented by the chosen Callback method
    batch_sizes: list[int] = dynamic_batch_sizes(batch_size, increases=increases)
    # yes?
    steps_per_increase = total_steps / increases
    iterations_per_step_size = [max(1, round(steps_per_increase / batch_sizes[i])) for i in range(increases)]
    iterations = sum(iterations_per_step_size)
    # iterations, _final_step_count = _test_iterations(iterations, total_steps, batch_sizes, dynamic_buffer=dynamic_buffer, dynamic_batch=True)
    if steps_per_increase < batch_sizes[-1]:
        total_steps_corrected = max(
            sum(iterations_per_step_size[i] * batch_sizes[i] for i in range(increases)), batch_sizes[-1] * increases
        )
        logger.error(
            "total_steps %d is not high enough to fulfill the scaling steps. "
            "Increase it to at least %d or decrease the amount of iterations or the batch_size",
            total_steps,
            total_steps_corrected,
        )
    return iterations
