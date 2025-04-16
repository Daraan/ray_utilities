from __future__ import annotations

import datetime
import functools
from typing import TYPE_CHECKING, TypeVar

from ray.experimental import tqdm_ray
from tqdm import tqdm
from typing_extensions import Iterable, TypeIs

from ray_utilities.constants import RAY_UTILITIES_INITIALIZATION_TIMESTAMP

if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.tune.experiment import Trial

_T = TypeVar("_T")

def trial_name_creator(trial: Trial) -> str:
    start_time = datetime.datetime.fromtimestamp(trial.run_metadata.start_time or RAY_UTILITIES_INITIALIZATION_TIMESTAMP)
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


def get_trainable_name(trainable: Callable) -> str:
    """If a trainable is wrap return its name"""
    last = None
    while last != trainable:
        last = trainable
        while isinstance(trainable, functools.partial):
            trainable = trainable.func
        while hasattr(trainable, "__wrapped__"):
            trainable = trainable.__wrapped__  # type: ignore[attr-defined]
    return trainable.__name__


def is_pbar(pbar: Iterable[_T]) -> TypeIs[tqdm_ray.tqdm | tqdm[_T]]:
    """TypeIs-guard for tqdm or tqdm_ray.tqdm."""
    return isinstance(pbar, (tqdm_ray.tqdm, tqdm))
