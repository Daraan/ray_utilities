"""Utilities for ray and ray tune to be used across projects."""

# ruff: noqa: PLC0415  # imports at top level of file; safe import time if not needed.

from __future__ import annotations

import datetime
import logging
import colorlog
import random
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypeVar, overload

# fmt: off
try:
    # Import comet early for its monkey patch
    import comet_ml  # noqa: F401
except ImportError:
    pass
# fmt: on

from ray.experimental import tqdm_ray
from tqdm import tqdm
from typing_extensions import TypeIs

from ray_utilities.constants import GYM_V_0_26, RAY_UTILITIES_INITALIZATION_TIMESTAMP

from .typing.algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData

if TYPE_CHECKING:
    from ray.tune.experiment import Trial


_T = TypeVar("_T")

__all__ = [
    "AlgorithmReturnData",
    "StrictAlgorithmReturnData",
    "is_pbar",
    "run_tune",
    "seed_everything",
    "trial_name_creator",
]

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    utilities_handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s][ %(filename)s:%(lineno)d, %(funcName)s] :%(reset)s %(message)s"
    )
    utilities_handler.setFormatter(formatter)
    logger.addHandler(utilities_handler)
logger.info("Ray utilities imported")
logger.debug("Ray utilities logger debug level set")


def trial_name_creator(trial: Trial) -> str:
    start_time = datetime.datetime.fromtimestamp(trial.run_metadata.start_time or RAY_UTILITIES_INITALIZATION_TIMESTAMP)
    start_time_str = start_time.strftime("%Y-%m-%d_%H:%M")
    return "_".join(
        [
            trial.trainable_name,
            trial.config["env"],
            trial.config["module"],
            start_time_str,
            trial.trial_id,
        ]
    )


def is_pbar(pbar: Iterable[_T]) -> TypeIs[tqdm_ray.tqdm | tqdm[_T]]:
    return isinstance(pbar, (tqdm_ray.tqdm, tqdm))


@overload
def _split_seed(seed: None) -> tuple[None, None]: ...


@overload
def _split_seed(seed: int) -> tuple[int, int]: ...


def _split_seed(seed: Optional[int]) -> tuple[int, int] | tuple[None, None]:
    """
    Generate a pair of seeds from a single seed one to be consumed with the next call
    the other to generate further seeds.

    Splitting seeds helps to avoid covariance caused by seed reusal.
    """
    if seed is None:
        return None, None
    gen = random.Random(seed)
    return gen.randrange(2**32), gen.randrange(2**32)


_IntOrNone = TypeVar("_IntOrNone", int, None)


def seed_everything(
    env, seed: _IntOrNone, *, torch_manual=False, torch_deterministic=None
) -> tuple[_IntOrNone, _IntOrNone]:
    """
    Args:
        torch_manual: If True, will set torch.manual_seed and torch.cuda.manual_seed_all
            In some cases setting this causes bad models, so it is False by default
    """
    import numpy as np
    import torch

    # no not reuse seed if its not None
    seed, next_seed = _split_seed(seed)
    random.seed(seed)

    seed, next_seed = _split_seed(next_seed)
    np.random.seed(seed)

    # os.environ["PYTHONHASHSEED"] = str(seed)
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
    if env:
        if not GYM_V_0_26:  # gymnasium does not have this
            seed, next_seed = _split_seed(next_seed)
            env.seed(seed)
        seed, next_seed = _split_seed(next_seed)
        env.action_space.seed(seed)

    seed, next_seed = _split_seed(next_seed)
    return seed, next_seed


def flat_dict_to_nested(metrics: dict[str, Any]) -> dict[str, Any | dict[str, Any]]:
    nested_metrics = metrics.copy()
    for key_orig, v in metrics.items():
        k = key_orig
        subdict = nested_metrics
        while "/" in k:
            parent, k = k.split("/", 1)
            subdict = subdict.setdefault(parent, {})
        subdict[k] = v
        if key_orig != k:
            del nested_metrics[key_orig]
    return nested_metrics


# Circular import
from ray_utilities.runfiles.run_tune import run_tune  # noqa: E402
