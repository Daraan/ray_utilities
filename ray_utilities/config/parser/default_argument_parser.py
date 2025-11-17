"""Typed argument parser classes and utilities for Ray RLlib experiments.

This module provides extended argument parser classes that support type annotations,
meta-annotations for checkpoint restoration, and integration with Ray Tune experiments.
It includes mixins for various argument types and utilities for handling "auto" values.

The module extends the :mod:`tap` (Typed Argument Parser) library with additional
functionality specific to machine learning experiments and checkpointing workflows.
"""

from __future__ import annotations

# pyright: enableExperimentalFeatures=true
import argparse
import logging
import math
import os
import shlex
import sys
from ast import literal_eval
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Collection, Generic, Optional, Sequence, cast

from typing_extensions import Self, TypeIs, TypeVar

from ray_utilities.config.parser.subcommand import SubcommandMixin
from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN

try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False

from tap import Tap
from typing_extensions import Annotated, Literal, Sentinel, get_args, get_origin, get_type_hints

from ray_utilities.config.parser.pbt_scheduler_parser import PopulationBasedTrainingParser
from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations, split_timestep_budget
from ray_utilities.misc import AutoInt
from ray_utilities.nice_logger import ImportantLogger, set_project_log_level
from ray_utilities.warn import (
    warn_about_larger_minibatch_size,
    warn_if_batch_size_not_divisible,
    warn_if_minibatch_size_not_divisible,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


logger = logging.getLogger(__name__)


def _auto_int_transform(x) -> int | Literal["auto"]:
    """Transform string values to integers, preserving "auto" literal.

    Args:
        x: String value that is either a number or "auto".

    Returns:
        Integer value if parseable, otherwise the literal "auto".
    """
    return int(x) if x != "auto" else x


_T = TypeVar("_T")
_ParserT = TypeVar("_ParserT", bound="argparse.ArgumentParser | Tap")

_NO_DEFAULT = Sentinel("_NO_DEFAULT")
NO_VALUE = Sentinel("NO_VALUE")

NeverRestore = Annotated[_T, "NeverRestore"]
"""Type annotation marker for fields that should never be restored from checkpoints.

When a field is annotated with this marker, it will always be reset to its default
value when restoring from a checkpoint, regardless of what value was saved.

Example:
    >>> from typing_extensions import Annotated
    >>> class Args:
    ...     temp_setting: NeverRestore[bool] = False
"""

AlwaysRestore = Annotated[_T, "AlwaysRestore"]
"""Type annotation marker for fields that should always be restored from checkpoints.

Fields marked with this annotation will always use the value from the checkpoint,
even if there are conflicting values from other sources like configuration files.

Example:
    >>> from typing_extensions import Annotated
    >>> class Args:
    ...     model_path: AlwaysRestore[str] = "default_model"
"""

RestoreIfDefault = Annotated[_T, "RestoreIf", NO_VALUE]  # note do not use a generic here
"""Type annotation marker for conditional restoration (currently not in use).

This marker would restore a value only if the current value matches the given default.
"""

NotAModelParameter = Annotated[_T, "NotAModelParameter"]
"""Type annotation marker for arguments that are not model parameters.

Arguments marked with this annotation will be excluded from the model's configuration
and will not be included in the ``"cli_args"`` key of the trainable's config.
This prevents these values from being uploaded to experiment tracking systems
like Weights & Biases or Comet.

Example:
    >>> class Args:
    ...     num_jobs: NotAModelParameter[int] = 4  # Infrastructure setting, not a hyperparameter

See Also:
    :func:`clean_args_to_hparams`: Function that processes these annotations.
    :func:`remove_ignored_args`: Function that filters out these arguments.
    :data:`LOG_IGNORE_ARGS`: Configuration for logging exclusions.
"""

AcceptsBoolAsString = Annotated[_T, "AcceptsBoolAsString"]
"""
Notes that an option can handle: --option True/False as string inputs.
This suppresses a warning when using :class:`PatchArgsMixin`.

See Also:
    :class:`PatchArgsMixin`
"""

RestoreOverride = Annotated[_T, "RestoreOverride"]
"""
When using --restore_path this value will override the restored value.

May not be used with :data:`NeverRestore` or :data:`AlwaysRestore`, when used the current_value (if it exists)
will always be used regardless of the restored or default value.
"""


class SupportsMetaAnnotations(Tap):
    """Mixin class for argument parsers that support meta annotations for checkpoint handling.

    This class extends :class:`tap.Tap` to provide support for special type annotations
    that control how arguments are handled during checkpoint restoration and experiment
    configuration. It processes annotations at configuration time and provides methods
    to query and restore arguments based on their annotations.

    Supported meta annotations:
        - :data:`AlwaysRestore`: Always restore the value from a checkpoint
        - :data:`NeverRestore`: Never restore from checkpoint, always use default
        - :data:`NotAModelParameter`: Exclude from model configuration

    Attributes:
        _always_restore: Set of argument names marked with AlwaysRestore
        _never_restore: Set of argument names marked with NeverRestore
        _non_cli_args: Set of argument names marked with NotAModelParameter

    Raises:
        ValueError: If a NeverRestore argument lacks a default value
    """

    def configure(self) -> None:
        """Configure the parser by processing meta annotations from type hints.

        This method analyzes all type annotations from the current class and its
        parent classes to identify arguments with special restoration behavior.
        It populates internal sets for tracking different annotation types.

        Raises:
            ValueError: If an argument is marked NeverRestore but has no default value.
        """
        super().configure()
        complete_annotations = self._get_from_self_and_super(
            extract_func=lambda super_class: dict(get_type_hints(super_class, include_extras=True))
        )
        # get literals dynamically to be future proof
        always_restore: Literal["AlwaysRestore"] = get_args(AlwaysRestore)[-1]
        never_restore: Literal["NeverRestore"] = get_args(NeverRestore)[-1]

        self._always_restore: set[str] = {
            k for k, v in complete_annotations.items() if get_origin(v) is Annotated and always_restore in get_args(v)
        }
        self._never_restore: set[str] = {
            k for k, v in complete_annotations.items() if get_origin(v) is Annotated and never_restore in get_args(v)
        }
        for k in self._never_restore:
            if getattr(self, k, _NO_DEFAULT) is _NO_DEFAULT:
                raise ValueError(
                    f"Argument '{k}' is annotated with NeverRestore but has no default value set. "
                    "Please provide a default value or remove the NeverRestore annotation."
                )

        # non cli args
        non_a_hp: Literal["NotAModelParameter"] = get_args(NotAModelParameter)[-1]
        self._non_cli_args: set[str] = {
            k for k, v in complete_annotations.items() if get_origin(v) is Annotated and non_a_hp in get_args(v)
        }
        # Accepts bool as string is not handled by this class (needs class level)

        restore_override_str: Literal["RestoreOverride"] = get_args(RestoreOverride)[-1]
        self._restore_override_args: set[str] = {
            k
            for k, v in complete_annotations.items()
            if get_origin(v) is Annotated and restore_override_str in get_args(v)
        }
        restore_conflicts_with_never = self._restore_override_args & self._never_restore
        restore_conflicts_with_always = self._restore_override_args & self._always_restore
        if restore_conflicts_with_never:
            raise ValueError(
                f"Arguments {restore_conflicts_with_never} cannot be annotated with both NeverRestore and RestoreOverride."
            )
        if restore_conflicts_with_always:
            raise ValueError(
                f"Arguments {restore_conflicts_with_always} cannot be annotated with both AlwaysRestore and RestoreOverride."
            )

    def get_restore_overrides(self) -> dict[str, Any]:
        """Get a dictionary of argument names and their current values for restoration overrides.

        Returns:
            Dictionary mapping argument names marked with :data:`RestoreOverride`
            to their current values in the parser instance.
        """
        return {name: getattr(self, name) for name in self._restore_override_args}

    def get_to_restore_values(self) -> set[str]:
        """Get the set of argument names that should always be restored from checkpoints.

        Returns:
            Set of argument names marked with :data:`AlwaysRestore` annotation.
        """
        return self._always_restore

    def get_non_cli_args(self) -> set[str]:
        """Get the set of argument names that are not model parameters.

        Returns:
            Set of argument names marked with :data:`NotAModelParameter` annotation.
            These arguments will be excluded from model configuration and experiment tracking.
        """
        return self._non_cli_args

    def restore_arg(self, name: str, *, restored_value: Any | NO_VALUE, default: Any = NO_VALUE) -> Any | NO_VALUE:
        """Restore an argument value based on its meta annotations.

        Determines the appropriate value for an argument during checkpoint restoration
        based on its meta annotations. The restoration logic follows these rules:

        1. If annotated with :data:`NeverRestore`: Returns the class default value
        2. If annotated with :data:`AlwaysRestore`: Returns the restored value
        3. Otherwise: Uses standard restoration logic
        Note: RestoreOverride is handled separately when a Setup is initialized with restore_path.

        Args:
            name: The name of the argument to restore.
            restored_value: The value from the checkpoint, or :data:`NO_VALUE` if not found.
            default: The default value to use, or :data:`NO_VALUE` if not provided.

        Returns:
            The value to use for this argument, or :data:`NO_VALUE` if no value
            should be set.

        Example:
            >>> parser = MyParser()
            >>> # For a NeverRestore argument, always gets default
            >>> value = parser.restore_arg("temp_dir", restored_value="/tmp/old", default="/tmp/new")
            >>> # Returns "/tmp/new" regardless of restored_value
        """
        current_value = getattr(self, name, NO_VALUE)
        if name in self._restore_override_args and current_value is not NO_VALUE:
            return current_value
        current_value_is_default = current_value == getattr(type(self), name, None)
        if name in self._always_restore:
            if current_value is not NO_VALUE and current_value != restored_value:  # pyright: ignore[reportOperatorIssue]
                logger.log(
                    logging.DEBUG if current_value_is_default else logging.WARNING,
                    "Restoring AlwaysRestore argument '%s' from checkpoint: replacing %s (%s) with %s",
                    name,
                    current_value,
                    "default" if current_value_is_default else "explicitly passed",
                    restored_value,
                )
            return restored_value
        if name in self._never_restore:
            # return default
            if default is not NO_VALUE:
                return default
            default = getattr(type(self), name, NO_VALUE)
            if default is not NO_VALUE:
                return default
            raise ValueError(
                f"Argument '{name}' is annotated with NeverRestore but has no default value set. "
                "Please provide a default value or remove the NeverRestore annotation."
            )
        if restored_value is NO_VALUE:
            if current_value is not NO_VALUE:
                return current_value
            if default is not NO_VALUE:
                return default
            return getattr(type(self), name, NO_VALUE)
        return restored_value


class PatchArgsMixin(Tap):
    """Mixin class that allows temporarily overriding command line arguments.

    This mixin provides a context manager that can merge additional arguments
    with the existing :data:`sys.argv`, while preserving the priority of
    command-line arguments passed directly to the script.

    The patching mechanism allows setting default values for arguments that
    can still be overridden by explicit command-line arguments.

    Example:
        >>> # python script.py --another_arg cli_value
        >>> with MyParser.patch_args(
        ...     "--my_arg",
        ...     "default_value",
        ...     "--another_arg",
        ...     "patch_value",
        ... ):
        ...     parser = MyParser()
        ...     # parser.my_arg == "default_value"
        ...     # parser.another_arg == "cli_value"  # CLI takes priority
    """

    @classmethod
    @contextmanager
    def patch_args(cls, *args: str | Any, config_files: Optional[Sequence[str]] = None):
        """Context manager to temporarily merge additional arguments with sys.argv.

        Arguments present in the original :data:`sys.argv` will take higher priority
        than the patched arguments. This allows setting default values while
        preserving user-specified command-line arguments.

        Args:
            *args: Alternating argument names and values to patch.
                Should be in the format: ``"--arg1", "value1", "--arg2", "value2"``.

        Yields:
            None. The context modifies :data:`sys.argv` temporarily.

        Example:
            >>> with MyParser.patch_args("--lr", 0.001, "--batch_size", 32):
            ...     parser = MyParser()
            ...     # parser now has lr=0.001 and batch_size=32 as defaults

        See Also:
            :func:`ray_utilities.testing_utils.patch_args`: Alternative implementation
                for testing scenarios.
        """
        original_argv = sys.argv.copy()
        # Parse the original and patch args separately
        parser_argv = cls(config_files=config_files)
        patch_parser = cls(config_files=config_files)
        NO_VALUE = object()
        subparser_dests = set()
        for action in patch_parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparser_dests.add(action.dest)
            action.default = NO_VALUE
        if patch_parser._subparsers:
            for subparser in cast(
                "argparse._SubParsersAction[argparse.ArgumentParser]", patch_parser._subparsers
            ).choices.values():
                for action in subparser._actions:
                    action.default = NO_VALUE
        for action in parser_argv._actions:
            action.default = NO_VALUE
        if parser_argv._subparsers:
            for subparser in cast(
                "argparse._SubParsersAction[argparse.ArgumentParser]", parser_argv._subparsers
            ).choices.values():
                for action in subparser._actions:
                    action.default = NO_VALUE

        # Parse original CLI args (excluding script name)
        argv_ns, orig_unknown = parser_argv.parse_known_args(original_argv[1:])
        if [unknown for unknown in orig_unknown if not unknown.startswith("--tag:")]:
            ImportantLogger.important_warning(
                logger, "Passed unknown args for this parser via sys.argv: %s", orig_unknown, stacklevel=3
            )

        # Parse patch args
        config_args = [
            arg
            for args_from_config in patch_parser.args_from_configs
            for arg in shlex.split(args_from_config, comments=True)
        ]

        patch_ns, patch_unknown = patch_parser.parse_known_args(config_args + list(map(str, args)))
        if [unknown for unknown in patch_unknown if not unknown.startswith("--tag:")]:
            logger.warning("Patching with unknown args: %s", patch_unknown, stacklevel=3)

        # Remove NO_VALUE entries to keep those that were actually passed:
        passed_argv = {dest: v for dest, v in vars(argv_ns).items() if v is not NO_VALUE}
        passed_patch = {dest: v for dest, v in vars(patch_ns).items() if v is not NO_VALUE}
        # argv has highest priority
        merged_args = {**passed_patch, **passed_argv}

        # actions that were used
        used_actions = {
            action: merged_args[action.dest] for action in patch_parser._actions if action.dest in merged_args
        }

        complete_annotations = cls._get_from_self_and_super(
            extract_func=lambda super_class: dict(get_type_hints(super_class, include_extras=True))
        )

        new_args = []
        # Move subparser actions to the end:
        used_actions: dict[argparse.Action, Any] = {
            k: v for k, v in used_actions.items() if not isinstance(k, argparse._SubParsersAction)
        } | {k: v for k, v in used_actions.items() if isinstance(k, argparse._SubParsersAction)}
        for action, value in used_actions.items():
            cls._recus_add_actions(
                action,
                value,
                new_args=new_args,
                original_argv=original_argv,
                args=merged_args,
                complete_annotations=complete_annotations,
                merged_args=merged_args,
            )
        patched_argv = [original_argv[0], *map(str, new_args), *orig_unknown]
        sys.argv = patched_argv

        try:
            yield
        finally:
            sys.argv = original_argv

    @classmethod
    def _recus_add_actions(
        cls,
        action: argparse.Action,
        value: Any,
        *,
        new_args: list,
        original_argv,
        args,
        complete_annotations,
        merged_args,
    ):
        """Constructs argument list recursively for nested subparsers."""
        option = None
        args_option: str | None = None
        dd_option: str | None = None
        for option in action.option_strings:
            # find the one that was used:
            if option in original_argv:
                break
            if option in args:
                # could still be overwritten in argv
                args_option = option
            if option.startswith("--"):
                dd_option = option
        else:
            # did not find in patch
            if args_option is not None:
                option = args_option
            elif dd_option is not None:
                option = dd_option
        if option is None and not isinstance(action, argparse._SubParsersAction):
            return  # should not happen            # consider n_args and store_true/false
        if isinstance(value, bool):
            # action was used so add it, do not need to check store_true/false
            if action.nargs in (None, 0):
                new_args.append(option)
            else:
                # cannot pass bool as str
                # possible has some type conversion we cannot guess
                assert option is not None
                ok = (
                    option.lstrip("-") in complete_annotations
                    and get_origin(complete_annotations[option.lstrip("-")]) is Annotated
                    and "AcceptsBoolAsString" in get_args(complete_annotations[option.lstrip("-")])
                )
                if ok:
                    logger.debug(
                        "%s is annotated as AcceptsBoolAsString. '%s %s' will be passed as argument.",
                        option,
                        option,
                        str(value),
                    )
                else:
                    logger.warning(
                        "Cannot safely convert boolean value to string for option '%s'. "
                        "Make sure that the parser can handle '%s %s' and annotate it with AcceptsBoolAsString",
                        option,
                        option,
                        str(value),
                    )
                new_args.extend((option, str(value)))
            return

        if action.nargs in (None, 1):
            new_args.extend([option, value])
        elif action.nargs == 0:
            new_args.append(option)
        elif action.nargs == "?":
            new_args.extend([option, value] if value is not None else [option])
        elif action.nargs in ("*", "+") or isinstance(action.nargs, int):
            if isinstance(value, str):
                new_args.extend([option, value])
            else:
                new_args.extend([option, *(value if value is not None else [])])
        elif action.nargs == argparse.PARSER:
            new_args.append(value)
            if action.choices is None:
                logger.warning("No choices found for subparser action '%s'", option)
                return
            action = cast("argparse._SubParsersAction[argparse.ArgumentParser]", action)
            for sub_action in action.choices[value]._actions:
                sub_value = merged_args.get(sub_action.dest)
                if sub_value is None:
                    continue
                cls._recus_add_actions(
                    sub_action,
                    sub_value,
                    new_args=new_args,
                    original_argv=original_argv,
                    args=args,
                    complete_annotations=complete_annotations,
                    merged_args=merged_args,
                )
        else:
            logger.warning("Unexpected nargs value for option '%s': %s", option, action.nargs)
            new_args.extend([option] + (value if value is not None else []))


Subparsers = TypeVar("Subparsers", bound=argparse.ArgumentParser | None, default=Any | None)


class SubcommandHandlerBase(Tap, Generic[Subparsers]):
    _command_str = ""
    __command = None

    @property
    def command_str(self) -> str | Literal[""]:
        return self._command_str

    @property
    def command(self) -> Subparsers:  # type hint is per default union with | None
        return self.__command  # pyright: ignore[reportReturnType]

    @command.setter
    def command(self, value: Subparsers) -> None:
        self.__command = value

    _skip_subparser_fields = ["help", "description", "_parsed"]  # noqa: RUF012

    class __NO_VALUE:  # noqa: N801
        """Serilizable NO_VALUE Sentinel for Delegator classes."""

    def configure(self) -> None:
        super().configure()
        self.add_subparsers(dest="_command_str", help="Sub-command to run")

    def process_args(self) -> None:
        super().process_args()
        self._process_subparsers()
        self.post_subparser_process()

    def post_subparser_process(self) -> None:
        """Hook that is called after subparser processing.

        Can be overridden by subclasses to perform additional processing
        after the subparser has been set up.
        """

    def _process_subparsers(self):
        if not self.command_str:
            return
        assert isinstance(self._subparsers, argparse._SubParsersAction)
        # subparser is not parsed
        command: Subparsers | argparse.ArgumentParser = self._subparsers.choices[self.command_str]
        assert command is not None
        if isinstance(command, Tap):
            type(command).process_args(self)
        # Full sync, just use for duck-typing
        self.command = self.create_delegator(command)  # pyright: ignore[reportAttributeAccessIssue]

    def create_delegator(self, subparser: _ParserT) -> _ParserT:
        DelegatorCommand = type(
            subparser.__class__.__name__ + "Delegator",
            (subparser.__class__, SubcommandMixin),
            {
                "_parsed": True,
                "help": getattr(subparser, "help", None),
                "description": getattr(subparser, "description", self.__doc__),
                "_SubcommandMixin__parent": self,
            },
        )
        for action in subparser._actions:
            if action.dest in self._skip_subparser_fields:
                continue

            @property
            def delegator_getter(_delegator: Any, dest=action.dest) -> Any:  # noqa: PLR0206
                attr = getattr(self, dest, self.__NO_VALUE)
                if attr is self.__NO_VALUE:
                    if hasattr(_delegator.__class__, dest):
                        # we are possibly in __setstate__ where attributes are not yet set
                        return self.__NO_VALUE
                    raise AttributeError(
                        f"'{dest}' not found on '{self.__class__}' or delegator.__class__ '{_delegator.__class__}'"
                    )
                return attr

            @delegator_getter.setter
            def delegator_setter(_delegator: Any, value: Any, dest=action.dest) -> None:
                setattr(self, dest, value)

            setattr(DelegatorCommand, action.dest, delegator_setter)

        def attr_gettr(_delegator: Any, key: str) -> Any:
            # Optional: remove check for full Delegator -> main parser forwarding
            if not hasattr(subparser, key):
                raise AttributeError(f"'{key}' not found on subparser {subparser}")
            return getattr(self, key)

        DelegatorCommand.__getattr__ = attr_gettr

        # def attr_settr(_delegator: Any, key: str, value: Any) -> None:
        #    setattr(_delegator, key, value)
        # DelegatorCommand.__setattr__ = attr_settr
        # Also call process_args if its a Tap class
        if isinstance(subparser, Tap):
            # subparser.process_args()
            # Process args on parser, choose one:
            subparser._parsed = True
            # type(subparser).process_args(parser)
            DelegatorCommand.process_args(self)
        delegator = DelegatorCommand()
        delegator._parsed = True
        return delegator


class _GoalParser(Tap):
    mode: Literal["min", "max"] = "max"
    """One of {min, max}.Determines whether objective is minimizing or maximizing the metric attribute."""
    metric: str = EVAL_METRIC_RETURN_MEAN
    """The metric to be optimized as flat key, e.g. 'evaluation/env_runners/episode_return_mean'."""


class _DefaultSetupArgumentParser(_GoalParser, Tap):
    agent_type: AlwaysRestore[str] = "mlp"
    """Agent Architecture"""

    env_type: AlwaysRestore[str] = "cart"
    """Environment to run on"""

    iterations: NeverRestore[int | Literal["auto"]] = 1000  # NOTE: Overwritten by Extra
    """
    How many iterations to run.

    An iteration consists of *n* iterations over the PPO batch, each further
    divided into minibatches of size `minibatch_size`.
    """
    total_steps: int = 1_000_000  # NOTE: Overwritten by Extra

    seed: int | None = None
    test: NeverRestore[bool] = False

    extra: Optional[list[str]] = None

    from_checkpoint: NeverRestore[Optional[str]] = None

    restore_path: NotAModelParameter[NeverRestore[Optional[str]]] = None
    """
    Path to a previous Ray Tune experiment to restore from.

    When set, uses :meth:`tune.Tuner.restore` to continue a previously interrupted
    or paused experiment. The path should point to the experiment directory
    (e.g., ``~/ray_results/experiment_name``).

    Note:
        This is different from ``from_checkpoint`` which restores a single trial.
        This restores the entire Tune experiment including all trials and search state.
    """

    restore_errored: NotAModelParameter[RestoreOverride[Literal["restart", "resume", False]]] = "resume"
    """When restore_path is used controls the behavior of Tuner.restore resume_errored and restart_errored."""

    def configure(self) -> None:
        # Short hand args
        super().configure()
        self.add_argument("-a", "--agent_type")
        self.add_argument("-env", "--env_type")
        self.add_argument("--seed", default=None, type=int)
        # self.add_argument("--test", nargs="*", const=True, default=False)
        self.add_argument("--iterations", "-it", default="auto", type=_auto_int_transform)
        self.add_argument("--total_steps", "-ts")
        self.add_argument(
            "--from_checkpoint", "-cp", "-load", default=None, type=str, help="Path to the checkpoint to load from."
        )


class _EnvRunnerParser(Tap):
    num_env_runners: NeverRestore[int] = 0
    """Number of CPU workers to use for training"""

    evaluation_num_env_runners: NeverRestore[int] = 0
    """Number of CPU workers to use for evaluation"""

    num_envs_per_env_runner: int = 8
    """Number of parallel environments per env runner"""

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "-n_envs",
            "--num_envs_per_env_runner",
            type=int,
            required=False,
        )
        self.add_argument(
            "-n_runners",
            "--num_env_runners",
            type=int,
            required=False,
        )
        self.add_argument(
            "-n_eval_runners",
            "--evaluation_num_env_runners",
            type=int,
            required=False,
        )


def _parse_lr(value: str) -> float | list[tuple[int, float]]:
    try:
        # Try to parse as a float
        return float(value)
    except ValueError:
        # If it fails, try to parse as a list of tuples or lists
        try:
            result = literal_eval(value)
        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(f"Invalid learning rate format: {value}") from e
        else:
            for item in result:
                if not (isinstance(item, (list, tuple)) and all(isinstance(x, (float, int)) for x in item)):
                    raise argparse.ArgumentTypeError(
                        f"Invalid learning rate: Each item must be a list or tuple of floats or ints, got: {item}"
                    )
            return result


def _literal_tuple_or_none(value: str) -> tuple[int, int] | None:
    if value.lower() == "none":
        return None
    try:
        result = literal_eval(value)
        if isinstance(result, (tuple, list)) and len(result) == 2 and all(isinstance(x, int) for x in result):
            return tuple(result)
        raise argparse.ArgumentTypeError(f"Invalid format: Expected a tuple of two integers or 'None', got: {value}")
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid format: Could not parse '{value}'") from e


class RLlibArgumentParser(_EnvRunnerParser):
    """Attributes of this class have to be attributes of the AlgorithmConfig."""

    train_batch_size_per_learner: int = 2048  # batch size that ray samples
    minibatch_size: int = 128
    """Minibatch size used for backpropagation/optimization"""

    minibatch_scale: float | None = None
    """If set, minibatch_size will be scaled as: minibatch_size = int(train_batch_size_per_learner * minibatch_scale).
    Must be in range (0, 1]. A value of 1.0 sets minibatch_size equal to train_batch_size_per_learner."""

    lr: float | list[tuple[int, float]] = 1e-4

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--batch_size",
            dest="train_batch_size_per_learner",
            type=int,
            required=False,
        )
        self.add_argument(
            "--lr",
            "-lr",
            type=_parse_lr,
        )

    def process_args(self):
        # Apply minibatch_scale setting
        if self.minibatch_scale is not None:
            if self.minibatch_scale > 1.0:
                raise ValueError(f"minibatch_scale must be <= 1.0, got {self.minibatch_scale}")
            if self.minibatch_scale <= 0.0:
                raise ValueError(f"minibatch_scale must be > 0.0, got {self.minibatch_scale}")
            self.minibatch_size = int(self.train_batch_size_per_learner * self.minibatch_scale)

        # Emit warnings:
        warn_if_batch_size_not_divisible(
            batch_size=self.train_batch_size_per_learner, num_envs_per_env_runner=self.num_envs_per_env_runner
        )
        if self.minibatch_size > self.train_batch_size_per_learner:
            warn_about_larger_minibatch_size(
                minibatch_size=self.minibatch_size,
                train_batch_size_per_learner=self.train_batch_size_per_learner,
                note_adjustment=True,
            )
            self.minibatch_size = self.train_batch_size_per_learner
        warn_if_minibatch_size_not_divisible(
            minibatch_size=self.minibatch_size, num_envs_per_env_runner=self.num_envs_per_env_runner
        )
        return super().process_args()


class DefaultResourceArgParser(Tap):
    num_jobs: NotAModelParameter[NeverRestore[int]] = 0
    """
    Trials to run in parallel

    Use 0 to apply no limit and use the available resources.

    Note:
        When using PBT, you should set the limit via cpu limits and set this to 0.
    """

    num_samples: NotAModelParameter[NeverRestore[int]] = 1
    """
    Number of times to sample from hyperparameter space for grid search this value is multiplied by grid options.
    If None, same as num_jobs
    """

    num_cpus: NotAModelParameter[NeverRestore[int | None]] = None
    """Amount of cpus cores to allocate per worker.

    Note, this is only a parser option and has to be implemented in the
    resource allocation logic.
    """

    ray_address: NotAModelParameter[NeverRestore[str | None]] = "auto"

    object_store_memory: NotAModelParameter[NeverRestore[int | None]] = None
    """Amount of object store memory to allocate per worker in bytes."""

    gpu: NeverRestore[bool] = False

    # num_cpus_per_learner: NeverRestore[int | Literal["auto"]] = "auto"
    # """auto: 1 if num_gpus_per_learner == 0 else 0"""

    num_learners: NeverRestore[int] = 0
    """Number of learner workers"""

    parallel: NeverRestore[bool] = False
    """Use multiple CPUs per worker"""

    not_parallel: NotAModelParameter[NeverRestore[bool]] = False
    """
    Do not run multiple models in parallel, i.e. the Tuner will execute one job only.
    This is similar to num_jobs=1, but one might skip the Tuner setup.
    """

    hostname_selector: NotAModelParameter[RestoreOverride[Optional[str]]] = None
    """
    Scheduling selector for Ray cluster nodes based on hostname labels.

    Note:
        clusters must have been started with the --labels "hostname=<value>" option

    | Operator | Description | Example syntax |
    | --- | --- | --- |
    | Equals | Label matches exactly one value. | `"{"hostname": “<value>”}"`
    | Not equal | Label matches anything by one value. | `"{"hostname": “<!value>”}"`
    | In | Label matches one of the provided values. | `"{"hostname": “<in(val1,val2)>”}"`
    | Not in | Label matches none of the provided values. | `"{"hostname": “!<in(val1,val2)>”}"`
    """

    node_id_selector: NotAModelParameter[RestoreOverride[Optional[str]]] = None
    """
    Scheduling selector for Ray cluster nodes based on its ray.io/node-id labels.
    This label is added by default to all nodes

    | Operator | Description | Example syntax |
    | --- | --- | --- |
    | Equals | Label matches exactly one value. | `"{ray.io/node-id”: “<value>”}"`
    | Not equal | Label matches anything by one value. | `"{ray.io/node-id”: “<!value>”}"`
    | In | Label matches one of the provided values. | `"{ray.io/node-id”: “<in(val1,val2)>”}"`
    | Not in | Label matches none of the provided values. | `"{ray.io/node-id”: “!<in(val1,val2)>”}"`
    """

    def process_args(self) -> None:
        super().process_args()
        if self.num_samples is None:  # pyright: ignore[reportUnnecessaryComparison]
            self.num_samples = self.num_jobs

    def configure(self) -> None:
        super().configure()
        self.add_argument("-J", "--num_jobs")
        self.add_argument("-gpu", "--gpu")
        self.add_argument("-p", "--parallel")
        self.add_argument("-np", "--not_parallel")
        self.add_argument("--num_samples", "-n", type=int, default=None)


class DefaultEnvironmentArgParser(Tap):
    render_mode: NeverRestore[Optional[Literal["human", "rgb_array", "ansi"]]] = None
    """Render mode"""

    env_seeding_strategy: Literal["random", "constant", "same", "sequential"] = "sequential"
    """
    Options:

            - random: subsequent and repeated trials are independent
                (use ``make_seeded_env_callback(None)``)

            - constant: use a constant seed for all trials
                (use ``make_seeded_env_callback(0)``)

            - same: identical to the ``seed`` option
                (use ``make_seeded_env_callback(args.seed)``)

            - sequential: use different, but deterministic, seeds for each trial.
                The first trial will always use the same seed, but a different one from
                subsequent trials.

    Usage:

            .. code-block:: python

                    make_seeded_env_callback(env_seed)
                    seed_environments_for_config(config, env_seed)

    """

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--render_mode",
            "-render",
            nargs="?",
            const="rgb_array",
            type=str,
            default=None,
            choices=["human", "rgb_array", "ansi"],
        )
        self.add_argument(
            "--env_seeding_strategy",
            "-ess",
            type=str,
            default="sequential",
            choices=["random", "constant", "same", "sequential"],
        )


OnlineLoggingOption = Literal["offline", "offline+upload", "offline+upload@end", "online", "off", False]
"""off -> NO LOGGING; offline -> offline logging but no upload; offline+upload@end -> upload only at experiment end"""

LogStatsChoices = Literal["minimal", "more", "timers", "learners", "timers+learners", "most", "all"]
LOG_STATS = "log_stats"


class DefaultLoggingArgParser(Tap):
    log_level: NeverRestore[
        Literal["DEBUG", "INFO", "IMPORTANT_INFO", "WARNING", "IMPORTANT_WARNING", "ERROR", "CRITICAL"]
    ] = "INFO"
    wandb: NeverRestore[NotAModelParameter[AcceptsBoolAsString[OnlineLoggingOption]]] = False
    comet: NeverRestore[NotAModelParameter[AcceptsBoolAsString[OnlineLoggingOption]]] = False
    comment: Optional[str] = None
    tags: NotAModelParameter[list[str]] = []  # noqa: RUF012
    """
    Variadic argument to add tags to the experiment

    Tip:
        A second way to add tags is via a ``--tag:my_tag`` argument which can be used multiple times.
        Those tags are useful when combining tags from different sources, e.g. a config file.
        But, for this to work you must use ``known_only=True`` in :meth:`tap.Tap.parse_args`.

    Tip:
        If you structure a tag like ``my_tag:my_value`` or ``my_tag=my_value``
        a repeated tag with the same base (e. g., here ``my_tag``) will be overwritten.
        If a tag is provided in ``--tags`` it has the highest priority.
    """

    log_stats: LogStatsChoices = "minimal"
    """Log all metrics and do not reduce them to the most important ones"""

    _change_log_level = True
    """Whether to apply :func:`set_project_log_level` in :meth:`process_args`."""

    offline_loggers: list[Literal["csv", "tensorboard", "tb", "tbx", "json", "all", "1", "0"]] | bool | None = None
    """
    Which offline loggers to enable when using offline logging.

    A value of "all", "1" will be cast to True, "0" to False.

    Note:
        Its recommended to always have the json logger active when using wandb or comet for the verification
        of the uploads.
        ``TUNE_DISABLE_AUTO_CALLBACK_LOGGERS`` environment variable has to be set for tune to not add
        csv, json, and tensorboard loggers automatically.

        When not at least three values, ``all``, or ``1`` is used
        the flag ``TUNE_DISABLE_AUTO_CALLBACK_LOGGERS`` will be set.

        None behavior (default):
            - if comet or wandb is used -> set to ["json"]
            - if neither comet or wandb is used#
              - test mode: set to 0
              - normal mode: set to ["csv", "tensorboard", "json"]
    """

    @classmethod
    def _get_safe_str_patches(cls):
        try:
            oks = super()._get_safe_str_patches()  # pyright: ignore[reportAttributeAccessIssue]
        except AttributeError as e:
            if "has no attribute '_get_safe_str_patches'" not in str(e):
                raise
            oks = {}
        oks["--wandb"] = ["--wandb False", "--wandb True"]
        return oks

    @property
    def use_comet_offline(self) -> bool:
        return self.comet and self.comet.lower().startswith("offline")

    def __setstate__(self, d: dict[str, Any]) -> None:
        d.pop("use_comet_offline", None)  # do not set property
        d.pop("command_str", None)  # Cannot set subparser field
        return super().__setstate__(d)

    def _parse_logger_choices(  # noqa: PLR6301  # could be static
        self, value: OnlineLoggingOption
    ) -> OnlineLoggingOption | Literal[False]:
        if value in {"0", "False", "off"}:  # off -> no logging
            return False
        if value in {"1", "True", "on"}:
            return "online"
        return value

    @staticmethod
    def __parse_offline_loggers(value: str) -> Literal["csv", "tensorboard", "tb", "tbx", "json"] | bool:
        if value in {"0", "False", "off"}:  # off -> no logging
            return False
        if value in {"1", "True", "on", "all"}:
            return True
        if value not in {"csv", "tensorboard", "tb", "tbx", "json"}:
            raise argparse.ArgumentTypeError(
                f"Invalid offline logger: {value}. "
                f"Expected one of 'csv', 'tensorboard', 'tb', 'tbx', 'json', 'all', '1', '0'."
            )
        return value  # pyright: ignore[reportReturnType]

    def configure(self) -> None:
        super().configure()
        self.add_argument("--log_level")
        logger_choices: tuple[OnlineLoggingOption] = get_args(OnlineLoggingOption)
        self.add_argument(
            "--wandb",
            "-wb",
            nargs="?",
            const="online",
            default=False,
            choices=logger_choices,
            type=self._parse_logger_choices,
        )
        self.add_argument(
            "--comet",
            nargs="?",
            const="online",
            default=False,
            choices=logger_choices,
            type=self._parse_logger_choices,
        )
        self.add_argument("--comment", "-c", type=str, default=None)
        self.add_argument("--tags", nargs="+", default=[])
        self.add_argument(
            "--" + LOG_STATS, nargs="?", const="more", default="minimal", choices=get_args(LogStatsChoices)
        )
        self.add_argument("--offline_loggers", nargs="+", default=None, type=self.__parse_offline_loggers)

    # region tags

    def _add_extra_tags(self):
        if self.extra_args:
            extra_to_remove = []
            tags_to_add = []
            for arg in self.extra_args:
                if arg.startswith("--tag:"):
                    tag = arg.split(":", 1)[1]
                    if tag not in self.tags:
                        tags_to_add.append(tag)
                    extra_to_remove.append(arg)
            if extra_to_remove:
                self.extra_args = [arg for arg in self.extra_args if arg not in extra_to_remove]
            if self.tags:
                tags_to_add.extend(self.tags)
            self.tags = tags_to_add

    @staticmethod
    def organize_subtags(tags: Iterable[str], allow_multiple: Collection[str] = ()) -> list[str]:
        """Ensure that for tag:value or tag=val, only the last occurrence per key is kept.
        'key' and 'key:' are allowed both to be present; but 'key:' and 'key=' are considered duplicates.
        """
        tag_map: dict[str, str] = {}
        tags = list(tags)
        for tag in tags:
            if ":" in tag:
                key = tag.split(":", 1)[0] + ":"
                normalized_key = key[:-1]
            elif "=" in tag:
                key = tag.split("=", 1)[0] + "="
                normalized_key = key[:-1]
            else:
                key = tag
                normalized_key = None  # plain key, not normalized

            if normalized_key is not None and normalized_key not in allow_multiple:
                tag_map.pop(f"{normalized_key}:", None)  # remove 'key:' if exists
                tag_map.pop(f"{normalized_key}=", None)  # remove 'key=' if exists
                tag_map[key] = tag  # set new with : or =
            else:
                # allow both 'key' and 'key:' to be present
                tag_map[tag] = tag
        if tag_map:
            tags = list(tag_map.values())
        return tags

    # endregion

    def process_args(self) -> None:
        self._add_extra_tags()
        self.tags = self.organize_subtags(self.tags)
        super().process_args()
        if self._change_log_level:
            set_project_log_level(logging.getLogger("ray_utilities"), self.log_level)
            # If we set this before ray.init it will work automatically for all remote
            os.environ["RAY_UTILITIES_LOG_LEVEL"] = self.log_level
        if self.offline_loggers is None:
            if self.comet or self.wandb:
                self.offline_loggers = ["json"]
            elif getattr(self, "test", False):
                self.offline_loggers = ["json"]
            else:
                self.offline_loggers = True
        # We should have a list otherwise; but default could be overwritten with a bool
        elif isinstance(self.offline_loggers, list):
            any_false_offline_loggers = False in self.offline_loggers
            any_true_offline_loggers = True in self.offline_loggers
            if any_false_offline_loggers and any_true_offline_loggers:
                raise argparse.ArgumentError(
                    next(a for a in self._actions if a.dest == "offline_loggers"),
                    f"Cannot mix False and True in offline_loggers {self.offline_loggers}",
                )
            if (any_false_offline_loggers or any_true_offline_loggers) and len(self.offline_loggers) > 1:
                raise argparse.ArgumentError(
                    next(a for a in self._actions if a.dest == "offline_loggers"),
                    f"Cannot mix boolean and string values in offline_loggers {self.offline_loggers}",
                )
            if any_false_offline_loggers:
                self.offline_loggers = False
            elif any_true_offline_loggers:
                self.offline_loggers = True
        if self.offline_loggers is False or (
            self.offline_loggers is not True and len(cast("list", self.offline_loggers)) < 3
        ):
            # "all", "1" already converted to True
            # NOTE: if ray is initialized before this it will not work!
            import ray  # noqa: PLC0415

            if not ray.is_initialized():
                ImportantLogger.important_info(
                    logger,
                    "Requested < 3 offline loggers: %s. "
                    "Setting TUNE_DISABLE_AUTO_CALLBACK_LOGGERS='1' "
                    "to disable automatic addition of csv, json, and tensorboard loggers.",
                )
            else:
                pass
                # TODO: Do not print when called on remote
                # logger.warning(
                #    "Ray is already initialized. Setting TUNE_DISABLE_AUTO_CALLBACK_LOGGERS may not have an effect."
                # )
            os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"


class DefaultExtraArgs(Tap):
    extra: Optional[list[str]] = None

    def configure(self) -> None:
        super().configure()
        self.add_argument("--extra", help="extra arguments", nargs="+")


class CheckpointConfigArgumentParser(Tap):
    checkpoint_frequency: int | None = 65_536  # 8192*8
    """
    Frequency of checkpoints in steps (or iterations, see checkpoint_frequency_unit)
    0 or None for no checkpointing
    """

    checkpoint_frequency_unit: Literal["steps", "iterations"] = "steps"
    """Unit for checkpoint_frequency, either after # steps or iterations"""

    num_to_keep: NotAModelParameter[int | None] = None
    """The number of checkpoints to keep. None to keep all checkpoints."""

    buffer_length: Optional[int | Literal["auto", "save"]] = None
    """
    If an integer of save sets TUNE_RESULT_BUFFER_LENGTH to the min of the given value and checkpoint_frequency

    If auto, then each trainable will call itself with train_buffered with an appropriate value.
    """

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--buffer_length", default=None, type=lambda x: int(x) if x not in ("auto", "save", None) else x
        )

    def process_args(self) -> None:
        if self.num_to_keep is not None and self.num_to_keep <= 0:
            raise ValueError(f"num_to_keep must be a positive integer or None. Not {self.num_to_keep}.")
        # Handle buffer_length auto in DefaultArgumentParser on in Trainable
        if isinstance(self.buffer_length, (int, float)) and self.buffer_length <= 0:
            raise ValueError(f"buffer_length must be a positive integer, 'auto', or None. Not {self.buffer_length}.")
        return super().process_args()


class OptionalExtensionsArgs(RLlibArgumentParser):
    dynamic_buffer: AlwaysRestore[bool] = False
    """Use DynamicBufferCallback. Increases env steps sampled and batch size"""

    dynamic_batch: AlwaysRestore[bool] = False
    """Use dynamic batch, scales batch size via gradient accumulation"""

    iterations: NeverRestore[int | AutoInt | Literal["auto"]] = "auto"
    total_steps: int = 1_179_648  # 2**20 + 2**17
    min_step_size: int = 64
    """min_dynamic_buffer_size"""
    max_step_size: int = 8192
    """max_dynamic_buffer_size"""

    use_exact_total_steps: AlwaysRestore[bool] = False
    """
    If True, the total_steps are a lower bound, independently of dynamic_buffer are they adjusted to
    be divisible by max_step_size and min_step_size. In case of a dynamic buffer, this results in
    evenly distributed fractions of the total_steps size for each dynamic batch size.
    """

    no_exact_sampling: AlwaysRestore[bool] = False
    """
    Set to not add the exact_sampling_callback to the AlgorithmConfig.

    If this is True this is Rllib's default behavior which might sample a minor amount of more steps
    than required for the batch_size.
    For exactness this callback will trim the sampled data to the exact batch size.
    """

    keep_masked_samples: AlwaysRestore[bool] = False
    """
    Wether to not add the RemoveMaskedSamplesConnector to the AlgorithmConfig.

    Set to True to enable RLlibs's default behavior which inserts masked samples into the learner
    that do not contribute to the loss.
    """

    accumulate_gradients_every: int = 1
    """
    Number of accumulation steps for the gradient update.
    The accumulated gradients will be averaged before backpropagation.
    """

    no_dynamic_eval_interval: AlwaysRestore[bool] = False
    """
    Does not add the :class:`.DynamicEvalInterval` callback that is added per default

    Dynamic evaluation adjust the evaluation interval depending on the current batch size,
    otherwise it is every 16 iterations independent of the batch size.

    See also:
        :attr:`evaluate_every_n_steps_before_step` to evaluate more often at the beginning.
    """

    evaluate_every_n_steps_before_step: AlwaysRestore[tuple[int, int] | None] = (512, 32784)  # 8196 * 4
    """
    The default evaluation interval is every 16 iterations if :attr:`no_dynamic_eval_interval` is used.

    With dynamic evaluation intervals it scales depending on the current batch size.

    The command line argument has to be a string that can be literally evaluated to a tuple of two integers
    or that is falsy to disable this feature.

    Note:
        Only supported when :attr:`no_dynamic_eval_interval` is not set.
    """

    def process_args(self) -> None:
        super().process_args()
        budget = split_timestep_budget(
            total_steps=self.total_steps,
            min_size=self.min_step_size,
            max_size=self.max_step_size,
            assure_even=not self.use_exact_total_steps,
        )
        # eval_intervals = get_dynamic_evaluation_intervals(budget["step_sizes"], batch_size=self.train_batch_size_per_learner, eval_freq=4)
        self.total_steps = budget["total_steps"]
        if self.iterations == "auto":  # for testing reduce this number
            iterations = calculate_iterations(
                dynamic_buffer=self.dynamic_buffer,
                batch_size=self.train_batch_size_per_learner,  # <-- if adjusted manually afterwards iterations will be wrong  # noqa: E501
                total_steps=self.total_steps,
                assure_even=not self.use_exact_total_steps,
                min_size=self.min_step_size,
                max_size=self.max_step_size,
            )
            iterations = AutoInt(iterations)
        else:
            iterations = self.iterations
        self.iterations = iterations
        # TODO / NOTE: When adjusting the train_batch_size_per_learner afterwards the amount of
        # iterations will be wrong to reach total steps (at least shown in CLI).

    def configure(self) -> None:
        super().configure()
        self.add_argument("--iterations", "-it", default="auto", type=_auto_int_transform)
        self.add_argument("--evaluate_every_n_steps_before_step", type=_literal_tuple_or_none)


class ScalingPBTSubparser(OptionalExtensionsArgs, SubcommandHandlerBase[PopulationBasedTrainingParser | Subparsers]):
    @staticmethod
    def _is_pbt_used(parser: Self) -> TypeIs[PopulationBasedTrainingParser]:  # pyright: ignore[reportGeneralTypeIssues] # noqa: N805
        return isinstance(parser.command, PopulationBasedTrainingParser)

    def process_args(self) -> None:
        super().process_args()
        if not self._is_pbt_used(self):
            return
        if isinstance(self.perturbation_interval, float):
            if self.perturbation_interval.is_integer() and self.perturbation_interval > 1:
                # Do not modify it, but warn
                logger.warning(
                    "Passing a integer equal float %s as perturbation_interval that is outside of [0,1]. ",
                    self.perturbation_interval,
                )
            elif 0 <= self.perturbation_interval <= 1:
                if not isinstance(self.iterations, (float, int)):
                    raise ValueError(
                        "When passing perturbation_interval as fraction (float in [0,1]), "
                        f"the iterations argument must be numerical, not {self.iterations}"
                    )
                # Should be divisible by min_step_size and max_step_size (actually all)
                relative_interval = math.ceil(self.perturbation_interval * self.total_steps)
                # Find the closest integer to relative_interval that is divisible by both min_step_size and max_step_size
                min_step = getattr(self, "min_step_size", 1)
                max_step = getattr(self, "max_step_size", 1)

                def lcm(a: int, b: int) -> int:
                    return abs(a * b) // math.gcd(a, b) if a and b else 0

                step_lcm = lcm(min_step, max_step)
                if step_lcm > 0:
                    # Round relative_interval to the nearest multiple of step_lcm
                    relative_interval = int(round(relative_interval / step_lcm) * step_lcm)
                if relative_interval <= 0:
                    relative_interval = step_lcm
                logger.info(
                    "Interpreting perturbation_interval=%s as fraction of total steps: "
                    "Setting perturbation_interval=%s (rounded to be divisible by min/max_step_size).",
                    self.perturbation_interval,
                    relative_interval,
                )
                self.perturbation_interval = relative_interval
            # Else assume it is some float time attr

        max_step_size = getattr(self, "max_step_size", None)
        if max_step_size is None:
            return
        if self.perturbation_interval % max_step_size != 0:
            logger.warning(
                "The perturbation_interval (%s) is not a multiple of max_step_size (%s). "
                "This will lead to overstepping behavior and uneven perturbation intervals.",
                self.perturbation_interval,
                max_step_size,
            )

    def configure(self) -> None:
        super().configure()
        self.add_subparser("pbt", PopulationBasedTrainingParser, help="Enable Population Based Training")


#####


def _parse_tune_choices(
    value: str | Literal[False],
) -> str | Literal[False]:
    if value is False or value.lower() in ("false", "0", "none"):
        return False
    return value


class OptunaArgumentParser(_GoalParser, Tap):
    optimize_config: NotAModelParameter[NeverRestore[bool]] = (
        False  # legacy argument name; possible replace with --tune later
    )
    # FIXME: Change to use keys from create_tune_parameters.
    # NOTE: Need to be defined in add_argument below as well
    tune: NeverRestore[
        list[
            Literal[
                "all",
                "accumulate_gradients_every",
                "batch_size",
                "lr",
                "minibatch_size",
                "minibatch_scale",
                "num_envs_per_env_runner",
            ]
        ]
        | Literal[False]
    ] = False
    """List of dynamic parameters to be tuned"""

    pruner_min_trials: NotAModelParameter[NeverRestore[int]] = 3
    """When optimizing with Optuna, minimum number of trials before pruning is started."""
    pruner_warmup_steps: NotAModelParameter[NeverRestore[int]] = 40
    """When optimizing with Optuna, number of steps to wait before pruning is started."""

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--tune",
            nargs="+",
            default=False,
            choices=[
                "all",
                "accumulate_gradients_every",
                "batch_size",
                "lr",
                "minibatch_size",
                "minibatch_scale",
                "num_envs_per_env_runner",
            ],
            type=_parse_tune_choices,
        )

    def process_args(self) -> None:
        super().process_args()
        if self.optimize_config and not self.tune:
            logger.warning(
                "The `--optimize_config` argument is deprecated. When using a non-legacy setup, "
                "use `--tune param1 param2 ...` to specify parameters to tune."
            )
            return
        if self.tune:
            self.optimize_config = True


class ConfigFilePreParser(Tap):
    config_files: list[str] = []  # noqa: RUF012
    """
    Files with additional commands that should be added to the commands passed
    with a lower priority.
    """

    def configure(self) -> None:
        self.allow_abbrev = False
        super().configure()
        self.add_argument("--config_files", "-cfg", nargs="+", default=[])


class DefaultArgumentParser(
    SupportsMetaAnnotations,
    ScalingPBTSubparser[Subparsers],  # needs be before  OptionalExtensionsArgs
    OptionalExtensionsArgs,  # Needs to be before _DefaultSetupArgumentParser
    RLlibArgumentParser,
    OptunaArgumentParser,
    _DefaultSetupArgumentParser,
    CheckpointConfigArgumentParser,
    DefaultResourceArgParser,
    DefaultEnvironmentArgParser,
    DefaultLoggingArgParser,
    DefaultExtraArgs,
    PatchArgsMixin,
):
    def configure(self) -> None:
        self.allow_abbrev = False
        super().configure()

    @classmethod
    def enable_completion(cls) -> None:
        """Enable shell completion for this parser using argcomplete.

        This method should be called before parsing arguments if shell completion
        is desired. To activate in your shell, you need to run:

        ```bash
        pip install argcomplete
        eval "$(register-python-argcomplete your_script.py)"
        ```

        Or for all Python scripts:
        ```bash
        eval "$(register-python-argcomplete python)"
        ```
        """
        if ARGCOMPLETE_AVAILABLE:
            # Create a temporary parser instance to get the argparse object
            temp_parser = cls()
            temp_parser.parse_args([])
            # Apply argcomplete to the underlying argparse parser
            argcomplete.autocomplete(temp_parser)  # pyright: ignore[reportPossiblyUnboundVariable]
            logger.debug("Shell completion enabled with argcomplete")
        else:
            logger.warning("argcomplete not available. Install with 'pip install argcomplete' for shell completion.")

    def suggest_buffer_length(self) -> Optional[int | Literal["auto"]]:
        budget = split_timestep_budget(
            total_steps=self.total_steps,
            min_size=self.min_step_size,
            max_size=self.max_step_size,
            assure_even=not self.use_exact_total_steps,
        )
        if self.buffer_length == "save":
            max_buffer_suggestion = min(budget["iterations_per_step_size"])
            if self.checkpoint_frequency is not None and self.checkpoint_frequency > 0:
                max_buffer_suggestion = min(max_buffer_suggestion, self.checkpoint_frequency)
            if self._is_pbt_used(self.command):
                # Pass buffer results to subparser
                if self.command.time_attr == "training_iteration":
                    max_buffer_suggestion = min(max_buffer_suggestion, int(self.command.perturbation_interval))
            self.buffer_length = max_buffer_suggestion
            logger.info("Set 'save' buffer_length to %s", self.buffer_length)
        if isinstance(self.buffer_length, int):
            if self.buffer_length > min(budget["iterations_per_step_size"]):
                ImportantLogger.important_warning(
                    logger,
                    f"Requested buffer_length {self.buffer_length} is larger than "
                    f"the maximum save value of {min(budget['iterations_per_step_size'])}.",
                )
            if self.checkpoint_frequency is not None and self.checkpoint_frequency > 0:
                if self.buffer_length > self.checkpoint_frequency:
                    ImportantLogger.important_warning(
                        logger,
                        f"Requested buffer_length {self.buffer_length} is larger than "
                        f"the checkpoint_frequency of {self.checkpoint_frequency}. Setting to checkpoint_frequency, "
                        "otherwise values will be dropped.",
                    )
                    self.buffer_length = self.checkpoint_frequency
        return self.buffer_length

    def post_subparser_process(self) -> None:
        super().post_subparser_process()
        if self.buffer_length not in (None, "auto"):  # An "auto" value we will handle by the trainable
            self.buffer_length = self.suggest_buffer_length()
            ImportantLogger.important_info(
                logger,
                f"Setting TUNE_RESULT_BUFFER_LENGTH to {self.buffer_length}. "
                "Make sure perturbation interval and checkpoint frequency are divisible by it.",
            )
            os.environ["TUNE_RESULT_BUFFER_LENGTH"] = str(self.buffer_length)
