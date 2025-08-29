"""Random seeding utilities for reproducible experiments across multiple frameworks.

This module provides comprehensive seeding functionality to ensure reproducible results
across Python's :mod:`random`, :mod:`numpy`, PyTorch, TensorFlow, and Gymnasium environments.
It's particularly useful for Ray Tune hyperparameter optimization and RLlib training
where reproducibility is crucial for comparing results.

The main function :func:`seed_everything` handles seeding of all major ML frameworks
with proper seed splitting to avoid correlation between different random number generators.

Example:
    Basic usage for reproducible experiments::

        import gymnasium as gym
        from ray_utilities.random import seed_everything

        env = gym.make("CartPole-v1")
        remaining_seed = seed_everything(env, seed=42)

        # Now all frameworks use properly split seeds derived from 42
        # Use remaining_seed for further seeding if needed

Functions:
    :func:`split_seed`: Generate multiple independent seeds from a single seed
    :func:`seed_everything`: Seed all ML frameworks for reproducible experiments

See Also:
    :mod:`random`: Python's built-in random number generation
    :mod:`numpy.random`: NumPy random number generation
    :mod:`torch`: PyTorch random number generation
    :mod:`tensorflow`: TensorFlow random number generation
    :mod:`gymnasium`: Environment action space seeding
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, TypeVar, overload

import numpy as np

from ray_utilities.constants import GYM_V_0_26

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


_IntOrNone = TypeVar("_IntOrNone", bound=int | None)

__all__ = ["seed_everything"]


@overload
def split_seed(seed: None) -> tuple[None, None]: ...


@overload
def split_seed(seed: int) -> tuple[int, int]: ...


def split_seed(seed: _IntOrNone, n=2) -> tuple[_IntOrNone, ...]:
    """Generate multiple independent seeds from a single seed to avoid correlation.

    This function creates a sequence of independent random seeds from a single input seed,
    which helps prevent correlation issues that can arise when using the same seed
    across multiple random number generators or consecutive operations.

    Args:
        seed: The input seed to split. If ``None``, returns a tuple of ``None`` values.
        n: Number of seeds to generate. Defaults to 2.

    Returns:
        A tuple of ``n`` independent seeds. If input seed is ``None``, all returned
        seeds will be ``None``.

    Example:
        >>> seed1, seed2 = split_seed(42)
        >>> seed1 != seed2  # Seeds are different
        True
        >>> seed1, seed2, seed3 = split_seed(42, n=3)
        >>> len({seed1, seed2, seed3})  # All seeds are unique
        3

        With None input::

        >>> seeds = split_seed(None, n=3)
        >>> seeds
        (None, None, None)

    Note:
        This is particularly important when seeding multiple frameworks to ensure
        they don't produce correlated random sequences. Each framework should
        receive an independent seed derived from the original seed.
    """
    if seed is None:
        return (seed,) * n
    gen = random.Random(seed)
    return tuple(gen.randrange(2**32) for _ in range(n))  # pyright: ignore[reportReturnType]


def seed_everything(
    env: gym.Env | None, seed: _IntOrNone, *, torch_manual=False, torch_deterministic=None
) -> _IntOrNone:
    """Seed all major ML frameworks for fully reproducible experiments.

    This function comprehensively seeds Python's :mod:`random`, :mod:`numpy`, PyTorch,
    TensorFlow, and Gymnasium environments using properly split seeds to avoid
    correlation between different random number generators.

    Args:
        env: A Gymnasium environment to seed, or ``None`` to skip environment seeding.
            The environment's action space will be seeded if provided.
        seed: The master seed to use for generating all framework-specific seeds.
            If ``None``, frameworks will use their default random initialization.
        torch_manual: If ``True``, will call :func:`torch.manual_seed` and
            :func:`torch.cuda.manual_seed_all`. **Warning**: In some cases setting
            this can cause poor model performance, so it defaults to ``False``.
        torch_deterministic: If provided, will set :func:`torch.use_deterministic_algorithms`
            to this value. Useful for fully deterministic PyTorch operations, though
            it may impact performance.

    Returns:
        The remaining seed that can be used for further seeding operations, or ``None``
        if the input seed was ``None``.

    Example:
        Basic seeding for reproducible experiments::

        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> remaining_seed = seed_everything(env, seed=42)
        >>> # All frameworks now use independent seeds derived from 42

        For fully deterministic PyTorch (may impact performance)::

        >>> seed_everything(None, seed=123, torch_manual=True, torch_deterministic=True)

    Note:
        The function seeds frameworks in this order:

        1. **Python random**: Using :func:`random.seed`
        2. **NumPy**: Using :func:`numpy.random.seed`
        3. **PyTorch**: Using :func:`torch.seed` and optionally manual seeding
        4. **TensorFlow**: Using :func:`tf.random.set_seed` and :func:`keras.utils.set_random_seed`
        5. **Gymnasium environment**: Using environment and action space seeding

        Each framework receives an independent seed generated via :func:`split_seed`
        to prevent correlation between random number generators.

    Warning:
        - Setting ``torch_manual=True`` has been observed to cause poor model
          performance in some cases, so use with caution.
        - Setting ``torch_deterministic=True`` ensures full determinism but may
          significantly impact performance due to slower deterministic algorithms.
        - For older Gym versions (< 0.26), the deprecated ``env.seed()`` method
          will be called if available.

    See Also:
        :func:`split_seed`: Used internally to generate independent seeds
        :data:`ray_utilities.constants.GYM_V_0_26`: Version check for Gym compatibility
    """
    # no not reuse seed if its not None
    seed, next_seed = split_seed(seed)
    random.seed(seed)

    seed, next_seed = split_seed(next_seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if next_seed is None:
            torch.seed()
            torch.cuda.seed()
        elif torch_manual:
            seed, next_seed = split_seed(next_seed)
            torch.manual_seed(
                seed,
            )  # setting torch manual seed causes bad models, # ok seed 124
            seed, next_seed = split_seed(next_seed)
            torch.cuda.manual_seed_all(seed)
        if torch_deterministic is not None:
            logger.debug("Setting torch deterministic algorithms to %s", torch_deterministic)
            torch.use_deterministic_algorithms(torch_deterministic)
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if TYPE_CHECKING:
            import keras  # noqa: PLC0415
        else:
            from tensorflow import keras  # noqa: PLC0415
        seed, next_seed = split_seed(next_seed)
        tf.random.set_seed(seed)
        seed, next_seed = split_seed(next_seed)
        keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()

    if env:
        if not GYM_V_0_26:  # old original gym, gymnasium does not have this
            seed, next_seed = split_seed(next_seed)
            env.seed(seed)  # pyright: ignore[reportAttributeAccessIssue]
        seed, next_seed = split_seed(next_seed)
        env.action_space.seed(seed)

    return next_seed
