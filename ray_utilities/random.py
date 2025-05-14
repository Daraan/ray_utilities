from __future__ import annotations

import logging
import random
from typing import TypeVar, overload

from ray_utilities.constants import GYM_V_0_26

logger = logging.getLogger(__name__)


_IntOrNone = TypeVar("_IntOrNone", bound=int | None)


@overload
def _split_seed(seed: None) -> tuple[None, None]: ...


@overload
def _split_seed(seed: int) -> tuple[int, int]: ...


def _split_seed(seed: _IntOrNone, n=2) -> tuple[_IntOrNone, ...]:
    """
    Generate a pair of seeds from a single seed one to be consumed with the next call
    the other to generate further seeds.

    Splitting seeds helps to avoid covariance caused by seed reusal.
    """
    if seed is None:
        return (seed,) * n
    gen = random.Random(seed)
    return tuple(gen.randrange(2**32) for _ in range(n))  # pyright: ignore[reportReturnType]


def seed_everything(env, seed: _IntOrNone, *, torch_manual=False, torch_deterministic=None) -> _IntOrNone:
    """
    Args:
        torch_manual: If True, will set torch.manual_seed and torch.cuda.manual_seed_all
            In some cases setting this causes bad models, so it is False by default
    """
    import numpy as np

    # no not reuse seed if its not None
    seed, next_seed = _split_seed(seed)
    random.seed(seed)

    seed, next_seed = _split_seed(next_seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        if next_seed is None:
            torch.seed()
            torch.cuda.seed()
        elif torch_manual:
            seed, next_seed = _split_seed(next_seed)
            torch.manual_seed(
                seed,
            )  # setting torch manual seed causes bad models, # ok seed 124
            seed, next_seed = _split_seed(next_seed)
            torch.cuda.manual_seed_all(seed)
        if torch_deterministic is not None:
            logger.debug("Setting torch deterministic algorithms to %s", torch_deterministic)
            torch.use_deterministic_algorithms(torch_deterministic)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        seed, next_seed = _split_seed(next_seed)
        tf.random.set_seed(seed)

    if env:
        if not GYM_V_0_26:  # gymnasium does not have this
            seed, next_seed = _split_seed(next_seed)
            env.seed(seed)
        seed, next_seed = _split_seed(next_seed)
        env.action_space.seed(seed)

    return next_seed
