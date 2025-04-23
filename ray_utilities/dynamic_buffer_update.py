import math
import numpy as np
from typing import Protocol

import logging
logger = logging.getLogger(__name__)

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

    return batch_size, accumulate_gradients_every, n_steps


def calculate_total_steps(*, training_iterations: int, initial_steps: int, dynamic_buffer:bool, increases: int = 8):
    """
    Attention:
        Initial steps should be config.train_batch_size_per_learner without prior modifications.

    TODO:
        This should be calculated via the Callback Method that implements the buffer size increase
    """
    if dynamic_buffer:
        initial_steps = max(16, initial_steps // increases)
        step_ranges: list[int] = [initial_steps * (2 ** i) for i in range(increases)]
        iterations_per_increase = max(1, math.floor(training_iterations / increases))
        amount_steps_per_increase = [
            step_ranges[i] * iterations_per_increase for i in range(increases)
        ]
    else:
        step_ranges = [initial_steps] * increases
        iterations_per_increase = training_iterations / increases
        amount_steps_per_increase = [initial_steps * iterations_per_increase] * increases  # sum is int
    total_steps: int = int(sum(amount_steps_per_increase))
    logger.debug(
        "Total steps calculated: %d. Step ranges are %s, with %d iterations in between. Steps at each level: %s",
        total_steps,
        step_ranges,
        iterations_per_increase,
        amount_steps_per_increase,
    )
    return total_steps
