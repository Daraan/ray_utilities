"""Miscellaneous utilities for Ray RLlib workflows.

Provides various utility functions for working with Ray Tune experiments,
progress bars, and data structures. Includes functions for trial naming,
trainable introspection, dictionary operations, and error handling.
"""

from __future__ import annotations

import datetime
import functools
import re
import sys
from typing import TYPE_CHECKING, Any, TypeVar

if sys.version_info < (3, 11):
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
"""Regex pattern to extract the trial ID from checkpoint paths.

This pattern assumes the trial ID is in the format 'id=<part1>_<sample_number>'.
The length of each block is not validated.

Example:
    >>> match = RE_GET_TRIAL_ID.search("path/to/checkpoint/id=abc123_456")
    >>> match.group("trial_id") if match else None
    'abc123_456'
"""


def trial_name_creator(trial: Trial) -> str:
    """Create a descriptive name for a Ray Tune trial.

    Generates a human-readable trial name that includes the trainable name,
    environment, module, start time, and trial ID. Optionally includes
    checkpoint information if the trial was restored from a checkpoint.

    Args:
        trial: The :class:`ray.tune.experiment.Trial` object to create a name for.

    Returns:
        A formatted string containing trial information, with fields separated by underscores.
        Format: ``<setup_cls>_<trainable_name>_<env>_<module>_<start_time>_id=<trial_id>``
        with optional ``[_from_checkpoint=<checkpoint_id>]`` suffix.

    Example:
        >>> # For a PPO trial on CartPole started at 2023-01-01 12:00
        >>> trial_name_creator(trial)
        'PPO_CartPole-v1_ppo_2023-01-01_12:00_id=abc123_456'
    """
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
    """Extract the original name from a potentially wrapped trainable function.

    Unwraps :func:`functools.partial` objects and functions with ``__wrapped__``
    attributes to find the original function name. This is useful for identifying
    the underlying trainable when it has been decorated or partially applied.

    Args:
        trainable: A callable that may be wrapped with decorators or partial application.

    Returns:
        The ``__name__`` attribute of the unwrapped function.

    Example:
        >>> import functools
        >>> def my_trainable():
        ...     pass
        >>> wrapped = functools.partial(my_trainable, arg=1)
        >>> get_trainable_name(wrapped)
        'my_trainable'
    """
    last = None
    while last != trainable:
        last = trainable
        while isinstance(trainable, functools.partial):
            trainable = trainable.func
        while hasattr(trainable, "__wrapped__"):
            trainable = trainable.__wrapped__  # type: ignore[attr-defined]
    return trainable.__name__


def is_pbar(pbar: Iterable[_T]) -> TypeIs[tqdm_ray.tqdm | tqdm[_T]]:
    """Type guard to check if an iterable is a tqdm progress bar.

    This function serves as a :class:`typing_extensions.TypeIs` guard to narrow
    the type of an iterable to either :class:`ray.experimental.tqdm_ray.tqdm` or
    :class:`tqdm.tqdm`.

    Args:
        pbar: An iterable that might be a progress bar.

    Returns:
        ``True`` if the object is a tqdm or tqdm_ray progress bar, ``False`` otherwise.

    Example:
        >>> from tqdm import tqdm
        >>> progress = tqdm(range(10))
        >>> if is_pbar(progress):
        ...     # Type checker now knows progress is a tqdm object
        ...     progress.set_description("Processing")
    """
    return isinstance(pbar, (tqdm_ray.tqdm, tqdm))


def deep_update(mapping: dict[str, Any], *updating_mappings: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a dictionary with one or more updating dictionaries.

    This function performs a deep merge of dictionaries, where nested dictionaries
    are recursively merged rather than replaced. Non-dictionary values are overwritten.

    Note:
        This implementation is adapted from `Pydantic's internal utilities
        <https://github.com/pydantic/pydantic/blob/main/pydantic/_internal/_utils.py>`_.

    Args:
        mapping: The base dictionary to update.
        *updating_mappings: One or more dictionaries to merge into the base mapping.

    Returns:
        A new dictionary containing the merged result. The original dictionaries
        are not modified.

    Example:
        >>> base = {"a": {"x": 1, "y": 2}, "b": 3}
        >>> update = {"a": {"y": 20, "z": 30}, "c": 4}
        >>> deep_update(base, update)
        {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}
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
    """Raise errors from Ray Tune results as a single ExceptionGroup.

    Processes errors from Ray Tune training results and raises them in a structured way.
    If only one error is present, it's raised directly. Multiple errors are grouped
    using :class:`ExceptionGroup`.

    Args:
        result: Either a :class:`ray.tune.result_grid.ResultGrid` containing errors,
            or a sequence of exceptions to raise.
        msg: Custom message for the ExceptionGroup. Defaults to
            "Errors encountered during tuning".

    Raises:
        Exception: The single error if only one is present.
        ExceptionGroup: Multiple errors grouped together with the provided message.

    Returns:
        None if no errors are found in the ResultGrid.

    Example:
        >>> from ray.tune import ResultGrid
        >>> # Assuming result_grid contains errors from failed trials
        >>> raise_tune_errors(result_grid, "Training failed")
    """
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
    """An integer subclass that represents an automatically determined value.

    This class extends :class:`int` to provide a semantic distinction for values
    that were originally specified as "auto" in command-line arguments or configuration,
    but have been resolved to specific integer values.

    The class maintains the same behavior as a regular integer but can be used
    for type checking and to track the origin of automatically determined values.

    Example:
        >>> value = AutoInt(42)  # Originally "auto", resolved to 42
        >>> isinstance(value, int)  # True
        >>> isinstance(value, AutoInt)  # True
        >>> value + 10  # 52
    """
