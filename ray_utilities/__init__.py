"""Utilities for ray and ray tune to be used across projects."""

# ruff: noqa: PLC0415  # imports at top level of file; safe import time if not needed.

from __future__ import annotations

import datetime
import random
import time
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypeVar, overload

import gymnasium as gym
from packaging.version import Version
from packaging.version import parse as parse_version
from ray.experimental import tqdm_ray
from tqdm import tqdm
from typing_extensions import TypeIs

from .typing.algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData

if TYPE_CHECKING:
    from ray.tune.experiment import Trial

GYM_VERSION = parse_version(gym.__version__)
GYM_V_0_26: bool = GYM_VERSION >= Version("0.26")
"""First gymnasium version and above"""
GYM_V1: bool = GYM_VERSION >= Version("1.0.0")
"""Gymnasium version 1.0.0 and above"""

_T = TypeVar("_T")

_SCRIPT_TIMESTAMP = time.time()

__all__ = [
    "GYM_V1",
    "GYM_VERSION",
    "GYM_V_0_26",
    "AlgorithmReturnData",
    "StrictAlgorithmReturnData",
    "is_pbar",
    "seed_everything",
    "trial_name_creator",
]


def trial_name_creator(trial: Trial) -> str:
    start_time = datetime.datetime.fromtimestamp(trial.run_metadata.start_time or _SCRIPT_TIMESTAMP)
    start_time_str = start_time.strftime("%Y-%m-%d_%H:%M")
    return "_".join(
        [
            trial.trainable_name,
            trial.evaluated_params["env"],
            trial.evaluated_params["module"],
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


def seed_everything(env, seed: Optional[int], *, torch_manual=False):
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
