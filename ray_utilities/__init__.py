"""Utilities for ray and ray tune to be used across projects."""

# ruff: noqa: PLC0415  # imports at top level of file; safe import time if not needed.

from __future__ import annotations

import datetime
import logging
import random
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, TypeVar, overload

import colorlog

# fmt: off
try:
    # Import comet early for its monkey patch
    import comet_ml  # noqa: F401
except ImportError:
    pass
# fmt: on

import gymnasium as gym
from ray.experimental import tqdm_ray
from tqdm import tqdm
from typing_extensions import TypeIs

from ray_utilities.constants import GYM_V_0_26, RAY_UTILITIES_INITALIZATION_TIMESTAMP
from .typing.algorithm_return import AlgorithmReturnData, StrictAlgorithmReturnData

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.tune.experiment import Trial


_T = TypeVar("_T")

__all__ = [
    "AlgorithmReturnData",
    "StrictAlgorithmReturnData",
    "create_default_trainable",
    "default_trainable",
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
    module = trial.config["module"]
    if module is None:
        module = trial.config["cli_args"]["agent_type"]
    fields = [
        trial.trainable_name,
        trial.config["env"],
        module,
        start_time_str,
        trial.trial_id,
    ]
    return "_".join(fields)


def is_pbar(pbar: Iterable[_T]) -> TypeIs[tqdm_ray.tqdm | tqdm[_T]]:
    return isinstance(pbar, (tqdm_ray.tqdm, tqdm))


@overload
def episode_iterator(args: dict[str, Any], hparams: Any, *, use_pbar: Literal[False]) -> range: ...


@overload
def episode_iterator(args: dict[str, Any], hparams: dict[Any, Any], *, use_pbar: Literal[True]) -> tqdm_ray.tqdm: ...


def episode_iterator(args: dict[str, Any], hparams: dict[str, Any], *, use_pbar=True) -> tqdm_ray.tqdm | range:
    """Creates an iterator for `args["episodes"]`

    Will create a `tqdm` if `use_pbar` is True, otherwise returns a range object.
    """
    if use_pbar:
        return tqdm_ray.tqdm(range(args["episodes"]), position=hparams.get("process_number", None))
    return range(args["episodes"])


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


_IntOrNone = TypeVar("_IntOrNone", bound=int | None)


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


def create_env_for_config(config: AlgorithmConfig, env_spec: str | gym.Env):
    if isinstance(config.env, str) and config.env != "seeded_env":
        init_env = gym.make(config.env)
    elif config.env == "seeded_env":
        if isinstance(env_spec, str):
            init_env = gym.make(env_spec)
        else:
            init_env = env_spec
    else:
        assert not TYPE_CHECKING or config.env
        init_env = gym.make(config.env.unwrapped.spec.id)  # pyright: ignore[reportOptionalMemberAccess]
    return init_env


# Circular import
from ray_utilities.runfiles.run_tune import run_tune  # noqa: E402
from ray_utilities.default_trainable import default_trainable, create_default_trainable  # noqa: E402
