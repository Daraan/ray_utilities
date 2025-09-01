from __future__ import annotations
import warnings
import logging

logger = logging.getLogger(__name__)


def warn_if_batch_size_not_divisible(*, batch_size: int, num_envs_per_env_runner: int | None) -> None:
    if num_envs_per_env_runner is None:  # satisfies typing as AlgorithmConfig.num_envs_per_env_runner: int | None
        return
    if batch_size % num_envs_per_env_runner != 0:
        suggestion = ((batch_size + num_envs_per_env_runner - 1) // num_envs_per_env_runner) * num_envs_per_env_runner
        warnings.warn(
            f"train_batch_size_per_learner {batch_size} is not divisible by "
            f"num_envs_per_env_runner {num_envs_per_env_runner}. "
            "This will lead to an unexpected number of steps taken. "
            f"Suggested change: Set train_batch_size_per_learner to {suggestion}.",
            UserWarning,
            stacklevel=2,
        )


def warn_if_minibatch_size_not_divisible(*, minibatch_size: int | None, num_envs_per_env_runner: int | None) -> None:
    if (
        minibatch_size is None or num_envs_per_env_runner is None
    ):  # mostly for typing as AlgorithmConfig.minibatch_size: int | None
        return
    if minibatch_size % num_envs_per_env_runner != 0:
        suggestion = (
            (minibatch_size + num_envs_per_env_runner - 1) // num_envs_per_env_runner
        ) * num_envs_per_env_runner
        warnings.warn(
            f"minibatch_size {minibatch_size} is not divisible by "
            f"num_envs_per_env_runner {num_envs_per_env_runner}. "
            "This will lead to not evenly distributed steps sampled during training. "
            f"Suggested change: Set minibatch_size to {suggestion}.",
            UserWarning,
            stacklevel=2,
        )


def warn_about_larger_minibatch_size(
    *, minibatch_size: int | None, train_batch_size_per_learner: int, note_adjustment: bool = True
) -> bool:
    """
    Checks for valid minibatch_size and train_batch_size_per_learner combination.

    Warns if minibatch_size is greater than train_batch_size_per_learner
    or when the train_batch_size_per_learner is not divisible by minibatch_size.

    Args:
        minibatch_size: The size of the mini-batch.
        train_batch_size_per_learner: The training batch size per learner.
        note_adjustment: Modify the warning to note that the adjustment is done automatically.

    Warns:
        UserWarning: if minibatch_size > train_batch_size_per_learner
        Logger Warning: if train_batch_size_per_learner not divisible by minibatch_size

    Returns:
        False if minibatch_size > train_batch_size_per_learner.
        Otherwise True (also if minibatch_size should be None).
    """
    if minibatch_size is None:
        logger.info("minibatch_size is unexpectedly None", stacklevel=2)
        return True
    if train_batch_size_per_learner % minibatch_size != 0:
        logger.warning(
            "train_batch_size_per_learner %d is not divisible by minibatch_size %d.",
            train_batch_size_per_learner,
            minibatch_size,
            stacklevel=2,
        )
    if minibatch_size > train_batch_size_per_learner:
        warnings.warn(
            f"minibatch_size {minibatch_size} is greater than "
            f"train_batch_size_per_learner {train_batch_size_per_learner}. "
            + (
                "Reducing the minibatch_size to the train_batch_size_per_learner."
                if note_adjustment
                else "This can result in an error."
            ),
            UserWarning,
            stacklevel=2,
        )
        return False
    return True
