"""Miscellaneous utilities for Ray RLlib workflows.

Provides various utility functions for working with Ray Tune experiments,
progress bars, and data structures. Includes functions for trial naming,
trainable introspection, dictionary operations, and error handling.
"""

from __future__ import annotations

import datetime
import functools
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any, TypeVar

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

from ray.experimental import tqdm_ray
from ray.tune.error import TuneError
from ray.tune.result_grid import ResultGrid
from tqdm import tqdm
from typing_extensions import Iterable, TypeIs

from ray_utilities.constants import (
    DEFAULT_EVAL_METRIC,
    EVAL_METRIC_RETURN_MEAN,
    NEW_LOG_EVAL_METRIC,
    RAY_UTILITIES_INITIALIZATION_TIMESTAMP,
    RUN_ID,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ray.tune.experiment import Trial

_T = TypeVar("_T")

_logger = logging.getLogger(__name__)

RE_GET_TRIAL_ID = re.compile(r"id=(?P<trial_id>(?P<trial_id_part1>[a-zA-Z0-9]{5,6})(?:_(?P<trial_number>[0-9]{5}))?)")
"""Regex pattern to extract the trial ID from checkpoint paths.

This pattern assumes the trial ID is in the format 'id=<part1>[_<trial_number>]',
with the trial number being optional. The length of each block is not validated.

Example:
    >>> match = RE_TRIAL_ID_FROM_CHECKPOINT.search("path/to/checkpoint/id=abc123_000001")
    >>> match.group("trial_id") if match else None
    'abc123'
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
    start_time_str = start_time.strftime("%Y-%m-%d_%H-%M")
    module = trial.config.get("module", None)
    if module is None and "cli_args" in trial.config:
        module = trial.config["cli_args"]["agent_type"]
    fields = [
        trial.config["env"],
        trial.trainable_name,
        module,
        "id=" + trial.trial_id,
        start_time_str,
    ]
    if "cli_args" in trial.config and trial.config["cli_args"]["from_checkpoint"]:
        match = RE_GET_TRIAL_ID.match(trial.config["cli_args"]["from_checkpoint"])
        if match:
            fields.append("from_checkpoint=" + match.group("trial_id"))
    setup_cls = trial.config.get("setup_cls", None)
    if setup_cls is not None:
        fields.insert(0, setup_cls)
    return "_".join(fields)


_NOT_FOUND = object()


def _is_key(key: str):
    return key.startswith("<") and key.endswith(">")


def _format_key(key: str) -> str:
    assert key[0] == "<" and key[-1] == ">", "Key must be wrapped in angle brackets."
    key = key[1:-1]  # remove angle brackets
    if key == "batch_size":
        return "train_batch_size_per_learner"
    return key


def extend_trial_name(
    insert: Iterable[str] = (), *, prepend: Iterable[str] = (), append: Iterable[str] = ()
) -> Callable[[Trial], str]:
    """
    Inserts strings or values from the trials config into the trial name.

    Values to be extracted from the config must be wrapped in angle brackets,
    e.g. "<my_param>". The function returns a new trial name creator that can be
    used in place of the default one.

    Args:
        insert: Iterable of strings or config keys to insert before the "_id=" part.
        prepend: Iterable of strings or config keys to prepend at the start.
        append: Iterable of strings or config keys to append at the end.

    Example:
        name_creator = extend_trial_name(insert=["<param1>"], prepend=["NEW"], append=["<param2>"])
        # This will create trial names like "NEW_<param1>_..._id=..._<param2>"

    Hint:
        For `train_batch_size_per_learner`, you can use the shorthand `<batch_size>` instead of the full key.

    Returns:
        A callable that takes a :class:`ray.tune.experiment.Trial` and returns
        a modified trial name string with the specified insertions.
    """
    if isinstance(insert, str):
        insert = (insert,)
    if isinstance(prepend, str):
        prepend = (prepend,)
    if isinstance(append, str):
        append = (append,)

    def extended_trial_name_creator(trial: Trial) -> str:
        base = trial_name_creator(trial)

        start, end = base.split("_id=")
        for key in insert:
            if _is_key(key):
                value = trial.config.get(_format_key(key), _NOT_FOUND)
                if value is _NOT_FOUND:
                    _logger.warning("Key %s not found in trial config, skipping insertion into trial name.", key)
                    continue
                start += f"_{key[1:-1]}={value}"
            else:
                start += f"_{key}"
        for key in prepend:
            if _is_key(key):
                value = trial.config.get(_format_key(key), _NOT_FOUND)
                if value is _NOT_FOUND:
                    _logger.warning("Key %s not found in trial config, skipping prepending into trial name.", key)
                    continue
                start = f"{key[1:-1]}={value}_" + start
            else:
                start = f"{key}_" + start
        for key in append:
            if _is_key(key):
                value = trial.config.get(_format_key(key), _NOT_FOUND)
                if value is _NOT_FOUND:
                    _logger.warning("Key %s not found in trial config, skipping appending into trial name.", key)
                    continue
                end += f"_{key[1:-1]}={value}"
            else:
                end += f"_{key}"
        return start + "_id=" + end

    return extended_trial_name_creator


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


def make_experiment_key(trial: Trial, trials_created: int) -> str:
    """
    Build a unique experiment key for a trial, making use of the :attr:`RUN_ID`,
    :attr:`~ray.tune.experiment.Trial.trial_id <Trial.trial_id>`, and the number of trials created so far.

    It has the format::
        {RUN_ID:x<26}xX{trial_id}X{trials_created:0>3} with all underscores replaced by "xx".

    References:
        - :attr:`RUN_ID <ray_utilities.constants.RUN_ID>`: A unique identifier for the current execution.
        - :attr:`~ray.tune.experiment.Trial.trial_id <Trial.trial_id>`: The unique ID of the trial.

    Note:
        The resulting key replaces underscores with "xx". For compatibility it should only contain
        numbers and letters.
        For comet it should be at most 50 characters long. This is not enforced here.
    """
    return f"{RUN_ID:x<26}xX{trial.trial_id}X{trials_created:0>3}".replace("_", "xx")


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
        for i, error in enumerate(result.errors):
            if not isinstance(error, BaseException):
                _logger.debug("Error %d is not an exception: %s", i, error)
                exception = TuneError("Error(s) occurred:\n" + str(error))
                result.errors[i] = exception
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


_new_log_format_used: bool | None = None


def new_log_format_used() -> bool:
    """Check if the new log format is enabled via environment variable.

    This function checks the environment variable
    :envvar:`RAY_UTILITIES_NEW_LOG_FORMAT` to determine if the new logging format
    should be used. The new format changes how metrics are structured in logs that are,
    for example, sent to WandB or Comet ML.

    Returns:
        ``True`` if the environment variable is set to a truthy value (not "0", "false", or "off"),
        ``False`` otherwise.

    Note:
        The result is cached after the first call.
    """
    global _new_log_format_used  # noqa: PLW0603
    if _new_log_format_used is not None:
        return _new_log_format_used
    _new_log_format_used = "RAY_UTILITIES_NEW_LOG_FORMAT" in os.environ and os.environ[
        "RAY_UTILITIES_NEW_LOG_FORMAT"
    ].lower() not in (
        "0",
        "false",
        "off",
    )
    return _new_log_format_used


def resolve_default_eval_metric(eval_metric: str | DEFAULT_EVAL_METRIC | None = None) -> str:
    """Resolve the default evaluation metric based on log format.

    This function determines the evaluation metric key to use
    based on whether the new log format is enabled. If the provided
    `eval_metric` is ``None``, it defaults to either
    :attr:`EVAL_METRIC_RETURN_MEAN <ray_utilities.constants.EVAL_METRIC_RETURN_MEAN>` or
    :attr:`NEW_LOG_EVAL_METRIC <ray_utilities.constants.NEW_LOG_EVAL_METRIC>` depending on
    the log format.

    Args:
        eval_metric: The evaluation metric key to resolve, or ``None`` to use the default.

    Returns:
        The resolved evaluation metric key as a string.

    Attention:
        This function is intended to be used with loggers like the WandB or Comet logger
        where the log metrics are changed for better human interpretation and not
        for other callbacks where the original metric keys are expected.
    """
    if eval_metric is not DEFAULT_EVAL_METRIC and eval_metric is not None:
        return eval_metric
    if new_log_format_used():
        return NEW_LOG_EVAL_METRIC
    return EVAL_METRIC_RETURN_MEAN
