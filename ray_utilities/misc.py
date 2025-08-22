from __future__ import annotations

import datetime
import functools
import re
from typing import TYPE_CHECKING, Any, TypeVar

from exceptiongroup import ExceptionGroup
from ray.experimental import tqdm_ray
from ray.tune.result_grid import ResultGrid
from tqdm import tqdm
from typing_extensions import Iterable, TypeIs

from ray_utilities.constants import RAY_UTILITIES_INITIALIZATION_TIMESTAMP

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ray.tune.experiment import Trial

_T = TypeVar("_T")

RE_GET_TRIAL_ID = re.compile("id=(?P<trial_id>[a-zA-Z0-9]+_[0-9]+)")
"""
Regex Pattern to extract the id of a trial.

Assumes the id is in the format 'id=<part1>_<sample_number>'.
That each block should have five characters is not checked for.
"""


def trial_name_creator(trial: Trial) -> str:
    start_time = datetime.datetime.fromtimestamp(
        trial.run_metadata.start_time or RAY_UTILITIES_INITIALIZATION_TIMESTAMP
    )
    start_time_str = start_time.strftime("%Y-%m-%d_%H:%M")
    module = trial.config.get("module", None)
    if module is None and "cli_args" in trial.config:
        module = trial.config["cli_args"]["agent_type"]
    fields = [
        trial.trainable_name,
        trial.config["env"],
        module,
        start_time_str,
        "id=" + trial.trial_id,
    ]
    if "cli_args" in trial.config and trial.config["cli_args"]["from_checkpoint"]:
        match = RE_GET_TRIAL_ID.match(trial.config["cli_args"]["from_checkpoint"])
        if match:
            fields.append("from_checkpoint=" + match.group("trial_id"))
    setup_cls = trial.config.get("setup_cls", None)
    if setup_cls is not None:
        fields.insert(0, setup_cls)
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


def deep_update(mapping: dict[str, Any], *updating_mappings: dict[str, Any]) -> dict[str, Any]:
    """
    Taken from pydantic:
    https://github.com/pydantic/pydantic/blob/main/pydantic/_internal/_utils.py
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def raise_tune_errors(result: ResultGrid | Sequence[Exception], msg: str = "Errors encountered during tuning") -> None:
    if isinstance(result, ResultGrid):
        if not result.errors:
            return
        if len(result.errors) == 1:
            raise result.errors[0]
        errors = result.errors
    else:
        errors = result
    raise ExceptionGroup(msg, errors)


class AutoInt(int):
    """An integer created from an "auto" string in the args."""
