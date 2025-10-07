"""Miscellaneous utilities for Ray RLlib workflows.

Provides various utility functions for working with Ray Tune experiments,
progress bars, and data structures. Includes functions for trial naming,
trainable introspection, dictionary operations, and error handling.
"""

from __future__ import annotations

import datetime
from enum import Enum
import functools
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any, Mapping, Optional, TypeVar

import base62
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
    RE_PARSE_FORK_FROM,
    RUN_ID,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ray.tune.experiment import Trial

    from ray_utilities.typing import ForkFromData

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

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


def extract_trial_id_from_checkpoint(ckpt_path: str) -> str | None:
    """Extract the trial ID from a checkpoint path.

    This function uses a regex pattern to extract the trial ID from a given
    checkpoint path. The expected format is 'id=<part1>[_<trial_number>]',
    where the trial number is optional.

    Args:
        ckpt_path: The checkpoint path string to extract the trial ID from.

    Note:
        As a fallback, the function also checks for an older format without the 'id=' prefix.
        In this case it looks for a pattern like '<part1>_<trial_number>'.
        Where trial_number is exactly 5 digits and part1 is at least 5 alphanumeric characters.

    Returns:
        The extracted trial ID as a string, or ``None`` if no valid ID is found
    """
    # TODO possibly, trial id might contain _forkof_/fork_from in the future.
    match = RE_GET_TRIAL_ID.search(ckpt_path)
    # get id of run
    if match:
        return match.groupdict()["trial_id"]
    # Deprecated:
    # possible old format without id=
    match = re.search(r"(?:id=)?(?P<trial_id>[a-zA-Z0-9]{5,}_[0-9]{5})", ckpt_path)
    if match:
        return match.groupdict()["trial_id"]
    return None


def parse_fork_from(fork_from: str) -> tuple[str, int | None] | None:
    """Parse a forked trial identifier into its components.

    This function takes a forked trial identifier string and splits it into
    the original trial ID and the step at which the fork occurred. The expected
    format is '<trial_id>?_step=<step>', where the step is optional.

    Args:
        fork_from: The forked trial identifier string to parse.
            Example: "abc123?_step=10" or "abc123"
    Returns:
        A tuple containing the trial ID as a string and the step as an integer
        or ``None`` if the step is not specified.
        Or None if the input string does not match the expected format.
    """
    # Note: only used for wandb currently, possibly move there
    # NOTE: could also just do split("?_step=") here
    match = RE_PARSE_FORK_FROM.match(fork_from)
    if not match:
        return None
    trial_id = match.group("fork_id")
    step_str = match.group("fork_step")
    step = int(step_str) if step_str is not None else None
    return trial_id, step


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


class ExperimentKey(str, Enum):
    """Lookup for replacements in experiment keys when using :func:`make_experiment_key`."""

    REPLACE_UNDERSCORE = ""
    """All underscores in the trial ID are replaced by this string."""

    REPLACE_3ZEROS = ""  # noqa: PIE796
    """All occurrences of "000" in the trial ID's count part are replaced by this string to shorten it."""

    RIGHT_PAD_CHAR = "Z"
    """Right padding character to reach at least 32 characters in length."""

    MIN_LENGTH = 32
    """
    Minimum length of the experiment key for non-forks. :attr:`RIGHT_PAD_CHAR` is used to pad the key to this length.
    Should be at least 32 characters long to be valid for Comet.
    """

    NO_ITERATION_DATA = "NaN"
    """
    String to use when no iteration data is available.

    Key will end with "SNaN" in this case instead of "S####"
    """

    RUN_ID_SEPARATOR = "X"
    """Separator between RUN_ID and trial ID."""

    FORK_SEPARATOR = "F"
    """Separator between trial ID and fork information."""

    COUNT_SEPARATOR = "C"
    """Separator between trial ID and trial count part."""

    STEP_SEPARATOR = "S"

    @classmethod
    def _make_non_fork_experiment_key(cls, trial: Trial) -> str:
        trial_base, *trial_number = trial.trial_id.split("_")
        if len(trial_number) > 1:
            _logger.warning(
                "Unexpected trial_id format '%s'. Expected format '<id>_<number>'.",
                trial.trial_id,
            )
        if trial_number:
            trial_number = cls.COUNT_SEPARATOR + "".join(trial_number).replace("000", cls.REPLACE_UNDERSCORE)
        else:  # empty list
            trial_number = ""
        base_key = f"{RUN_ID}{cls.RUN_ID_SEPARATOR}{trial_base}{trial_number}".replace("_", cls.REPLACE_UNDERSCORE)
        # Pad at the end with Z to be at least 32 characters long.
        # Use uppercase letters as trial_id is lowercase alphanumeric only.
        base_key = f"{base_key:{cls.RIGHT_PAD_CHAR}<{cls.MIN_LENGTH}}"
        return base_key

    @classmethod
    def _make_fork_experiment_key(cls, base_key: str, fork_data: ForkFromData) -> str:
        base_key = base_key.rstrip(cls.RIGHT_PAD_CHAR)
        parent_id = fork_data["parent_id"]
        # Prefer training iteration for experiment key (stable across frameworks)
        ft = fork_data.get("parent_time")
        # ft is a NamedTuple[time_attr, time]; only use numeric time
        parent_iteration = ft[1] if ft[0] == "current_step" else fork_data["parent_training_iteration"]

        fork_base, *fork_number = parent_id.split("_")
        if fork_number:
            fork_number = cls.COUNT_SEPARATOR + "".join(fork_number).replace("000", cls.REPLACE_3ZEROS)
        else:
            fork_number = ""

        if parent_iteration is None:  # pyright: ignore[reportUnnecessaryComparison]
            iteration_data = cls.NO_ITERATION_DATA  # rare chance that NaN is actually encoded
            r_pad = 0
            _logger.warning("parent_iteration is None, using 'NaN' in experiment key.")
        else:
            r_pad = 4
            iteration_data = base62.encode(parent_iteration)
        return (
            f"{base_key}{cls.FORK_SEPARATOR}{fork_base}{fork_number}{cls.STEP_SEPARATOR}{iteration_data:0>{r_pad}}"
        ).replace("_", cls.REPLACE_UNDERSCORE)

    @classmethod
    def make_experiment_key(cls, trial: Trial, fork_data: Optional[ForkFromData] = None) -> str:
        """
        Build a unique experiment key for a trial, making use of the :attr:`RUN_ID`,
        :attr:`~ray.tune.experiment.Trial.trial_id <Trial.trial_id>`, and optional fork information.

        It has the format of two forms:
            - If not forked: "<RUN_ID of 21 chars>X<trial_id>[Z* up to 32 chars in total]"
            it is prolonged by ``Z`` (:attr:``RIGHT_PAD``) to be at least 32 chars long.
            - If forked: "<RUN_ID of 21 chars>X<trial_id>F<trial_id>S<step>"
            The <step> is in base62 encoded and right aligned to 4 characters with leading zeros.

            Both have in common that the potential underscore in the trial ID is removed

            and all "000" in the trials
            count part of the trial ID are removed, e.g. ``abcd0001_00005`` becomes ``abcd0001C05``.

        References:
            - :attr:`RUN_ID <ray_utilities.constants.RUN_ID>`: A unique identifier for the current execution.
            - :attr:`~ray.tune.experiment.Trial.trial_id <Trial.trial_id>`: The unique ID of the trial.

        Note:
            The resulting key underscores replaced by C. For compatibility it should only contain
            numbers and letters.
            For comet it must be between 32 and 50 characters long. This is not enforced here.

            For comet support, to not exceed 50 characters, only 99 parallel trials are supported, e.g. <trial_id>_00099.
            The highest supported forking step is 14_776_335.

            It is assumed that the trial ID contains only lowercase alphanumeric characters and underscores.
        """
        base_key = cls._make_non_fork_experiment_key(trial)
        if not fork_data:
            return base_key
        return cls._make_fork_experiment_key(base_key, fork_data)


make_experiment_key = ExperimentKey.make_experiment_key


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


def get_value_by_path(tree_dict: Mapping[Any, Any], path: tuple[Any, ...]) -> Any:
    """Extract value from nested dict using a tuple path."""
    value = tree_dict
    for key in path:
        value = value[key]
    return value


def flatten_mapping_with_path(d, path=()) -> list[tuple[tuple[Any, ...], Any]]:
    """Similar to :func:`tree.flatten_with_path` but only for mappings."""
    items = []
    if isinstance(d, Mapping):
        for k, v in d.items():
            items.extend(flatten_mapping_with_path(v, (*path, k)))
    else:
        items.append((path, d))
    return items


# Build a nested dict with lists at the leaves
def build_nested_dict(paths: list[tuple[Any, ...]], values: dict[tuple[Any, ...], list]) -> dict:
    """Build a nested dict from paths and corresponding values."""
    nested = {}
    for path in paths:
        d = nested
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = values[path]
    return nested


def flat_dict_to_nested(metrics: dict[str, Any]) -> dict[str, Any | dict[str, Any]]:
    """Convert a flat dictionary with slash-separated keys to a nested dictionary structure.

    This function transforms dictionary keys containing forward slashes into nested
    dictionary structures, useful for organizing Ray Tune/RLlib metrics that are
    typically logged with hierarchical key names.

    Args:
        metrics: A dictionary with potentially slash-separated keys (e.g.,
            ``{"eval/return_mean": 100, "train/loss": 0.5}``).

    Returns:
        A nested dictionary structure where slash-separated keys become nested levels
        (e.g., ``{"eval": {"return_mean": 100}, "train": {"loss": 0.5}}``).

    Example:
        >>> metrics = {
        ...     "train/episode_return_mean": 150.0,
        ...     "eval/env_runner_results/episode_return_mean": 200.0,
        ...     "timesteps_total": 10000,
        ... }
        >>> nested = flat_dict_to_nested(metrics)
        >>> nested["train"]["episode_return_mean"]
        150.0
        >>> nested["eval"]["env_runner_results"]["episode_return_mean"]
        200.0

    Note:
        This is particularly useful when working with Ray Tune's result dictionaries
        which often contain hierarchical metrics with slash-separated key names.
    """
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
