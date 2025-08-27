from __future__ import annotations

# pyright: enableExperimentalFeatures=true
import logging
import sys
from contextlib import contextmanager
from typing import Any, Optional, TypeVar

from tap import Tap
from typing_extensions import Annotated, Literal, Sentinel, get_args, get_origin, get_type_hints

from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations, split_timestep_budget
from ray_utilities.misc import AutoInt
from ray_utilities.warn import (
    warn_about_larger_minibatch_size,
    warn_if_batch_size_not_divisible,
    warn_if_minibatch_size_not_divisible,
)

logger = logging.getLogger(__name__)


def _auto_int_transform(x) -> int | Literal["auto"]:
    return int(x) if x != "auto" else x


_T = TypeVar("_T")

_NO_DEFAULT = Sentinel("_NO_DEFAULT")
NO_VALUE = Sentinel("NO_VALUE")

NeverRestore = Annotated[_T, "NeverRestore"]
"""Marks a field that is never restored and always reset to its default value when restoring from a checkpoint."""
AlwaysRestore = Annotated[_T, "AlwaysRestore"]
"""
Marks a field that should always be restored and not ignored
e.g. it should not be superseeded by a get_args_from_config.
"""
RestoreIfDefault = Annotated[_T, "RestoreIf", NO_VALUE]  # note do not use a generic here
"""NOT IN USE. Restore if the current value is the given default value"""

NotAModelParameter = Annotated[_T, "NotAModelParameter"]
"""
Marks a key do be not available to the model, i.e. ``num_jobs``.

These values will not be included in the `"cli_args"` key of the trainable's config and
therefore not uploaded to wandb / comet.

See Also:
    ``clean_args_to_hparams``
    ``remove_ignored_args``
    ``LOG_IGNORE_ARGS``
"""


class SupportsMetaAnnotations(Tap):
    """
    Mixin class for argument parsers that support meta annotations:

    - `AlwaysRestore`: Always restore the value from a checkpoint.
    - `NeverRestore`: Never restore the value from a checkpoint, always use the default value
    - `NotAModelParameter`: Mark the argument as not a model parameter, i.e. not included in the model's config.
    """

    def configure(self) -> None:
        super().configure()
        complete_annotations = self._get_from_self_and_super(
            extract_func=lambda super_class: dict(get_type_hints(super_class, include_extras=True))
        )
        # get literals dynamically to be future proof
        always_restore: Literal["AlwaysRestore"] = get_args(AlwaysRestore)[-1]
        self._always_restore: set[str] = {
            k for k, v in complete_annotations.items() if get_origin(v) is Annotated and always_restore in get_args(v)
        }
        never_restore: Literal["NeverRestore"] = get_args(NeverRestore)[-1]
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

    def get_to_restore_values(self):
        return self._always_restore

    def get_non_cli_args(self):
        return self._non_cli_args

    def restore_arg(self, name: str, *, restored_value: Any | NO_VALUE, default: Any = NO_VALUE) -> Any | NO_VALUE:
        """
        If a value is annotated with NeverRestore its default value is returned as defined in the class.
        An argument with AlwaysRestore will ignore the value stored on this instance and
        return the `restored_value` instead.

        Use the attribute stored on this parser?

        AlwaysRestore: Return restored_value
        NeverRestore: return default or the default value set in the class
        Otherwise: return the restored_value of the argument.
            However if it is explicitly set to NO_VALUE, return the current_value.
            In case the attribute does not exist return the `default` value.
            This might be the Sentinel value NO_VALUE.

        Note:
            Depending on the setup, e.g. when config_from_args is used, AlwaysRestore
            values need to be checked again afterwards.
        """
        current_value = getattr(self, name, NO_VALUE)
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
    """
    Mixin class that allows to provide parameters that can be overwritten by the command line arguments.

    Example:
        # python file.py --another_arg cli_value
        with DefaultArgumentParser.patch_args(
            "--my_arg", "value",
            "--another_arg", "another_value",
        ):
            # Inside this context, the arguments are patched. Custom provided CLI args have the highest priority.
            # For example, if `--another_arg cli_value` is provided in the command line,
            # it will overwrite the value "another_value".
    """

    @classmethod
    @contextmanager
    def patch_args(cls, *args: str | Any):
        """
        Context manager to temporarily *merge* sys.argv with additional arguments.
        Arguments present in sys.argv will take a *higher* priority.

        See Also:
            - testing_utils.patch_args as an alternative
        """
        original_argv = sys.argv[:]
        # Parse the original and patch args separately
        parser_argv = cls()
        patch_parser = cls()
        NO_VALUE = object()
        for action in patch_parser._actions:
            action.default = NO_VALUE
        for action in parser_argv._actions:
            action.default = NO_VALUE

        # Parse original CLI args (excluding script name)
        argv_ns, orig_unknown = parser_argv.parse_known_args(original_argv[1:])

        # Parse patch args
        patch_ns, patch_unknown = patch_parser.parse_known_args(list(map(str, args)))

        # Remove NO_VALUE entries to keep those that were actually passed:
        passed_argv = {dest: v for dest, v in vars(argv_ns).items() if v is not NO_VALUE}
        passed_patch = {dest: v for dest, v in vars(patch_ns).items() if v is not NO_VALUE}
        # argv has highest priority
        merged_args = {**passed_patch, **passed_argv}

        # actions that were used
        used_actions = {
            action: merged_args[action.dest] for action in patch_parser._actions if action.dest in merged_args
        }

        new_args = []
        for action, value in used_actions.items():
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
            if option is None:
                continue  # should not happen            # consider n_args and store_true/false
            if isinstance(value, bool):
                # action was used so add it, do not need to check store_true/false
                if action.nargs in (None, 0):
                    new_args.append(option)
                else:
                    # cannot pass bool as str
                    # possible has some type conversion we cannot guess
                    logger.warning("Cannot safely convert boolean value to string for option '%s'", option)
                    new_args.extend((option, str(value)))
                continue

            if action.nargs in (None, 1):
                new_args.extend([option, value])
            elif action.nargs == 0:
                new_args.append(option)
            elif action.nargs == "?":
                new_args.extend([option, value] if value is not None else [option])
            elif action.nargs in ("*", "+") or isinstance(action.nargs, int):
                new_args.extend([option] + (value if value is not None else []))
            else:
                logger.warning("Unexpected nargs value for option '%s': %s", option, action.nargs)
                new_args.extend([option] + (value if value is not None else []))
        patched_argv = [original_argv[0], *map(str, new_args)]
        sys.argv = patched_argv

        try:
            yield
        finally:
            sys.argv = original_argv


class _DefaultSetupArgumentParser(Tap):
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
    num_env_runners: int = 0
    """Number of CPU workers to use for training"""

    evaluation_num_env_runners: int = 0
    """Number of CPU workers to use for evaluation"""

    num_envs_per_env_runner: int = 4
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


class RLlibArgumentParser(_EnvRunnerParser):
    """Attributes of this class have to be attributes of the AlgorithmConfig."""

    train_batch_size_per_learner: int = 2048  # batch size that ray samples
    minibatch_size: int = 128
    """Minibatch size used for backpropagation/optimization"""

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--batch_size",
            dest="train_batch_size_per_learner",
            type=int,
            required=False,
        )

    def process_args(self):
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
    num_jobs: NotAModelParameter[NeverRestore[int]] = 5
    """Trials to run in parallel"""

    num_samples: NotAModelParameter[NeverRestore[int]] = 1
    """Number of samples to run in parallel, if None, same as num_jobs"""

    gpu: NeverRestore[bool] = False

    parallel: NeverRestore[bool] = False
    """Use multiple CPUs per worker"""

    not_parallel: NotAModelParameter[NeverRestore[bool]] = False
    """
    Do not run multiple models in parallel, i.e. the Tuner will execute one job only.
    This is similar to num_jobs=1, but one might skip the Tuner setup.
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
            `make_seeded_env_callback(None)`
        - constant: use a constant seed for all trials
            `make_seeded_env_callback(0)`
        - same: Identical to `seed` option.
            `make_seeded_env_callback(args.seed)`
        - sequential: use different, but deterministic, seeds for each trial,
            i.e. the first trial will always use the same seed, but a different one from
            the subsequent trials.

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


OnlineLoggingOption = Literal["offline", "offline+upload", "online", "off", False]
"""off -> NO LOGGING; offline -> offline logging but no upload"""

LogStatsChoices = Literal["minimal", "more", "timers", "learners", "timers+learners", "most", "all"]
LOG_STATS = "log_stats"


class DefaultLoggingArgParser(Tap):
    log_level: NeverRestore[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = "INFO"
    wandb: NeverRestore[NotAModelParameter[OnlineLoggingOption]] = False
    comet: NeverRestore[NotAModelParameter[OnlineLoggingOption]] = False
    comment: Optional[str] = None
    tags: NotAModelParameter[list[str]] = []  # noqa: RUF012
    log_stats: LogStatsChoices = "minimal"
    """Log all metrics and do not reduce them to the most important ones"""

    @property
    def use_comet_offline(self) -> bool:
        return self.comet and self.comet.lower().startswith("offline")

    def __setstate__(self, d: dict[str, Any]) -> None:
        d.pop("use_comet_offline", None)  # do not set property
        return super().__setstate__(d)

    def _parse_logger_choices(  # noqa: PLR6301  # could be static
        self, value: OnlineLoggingOption
    ) -> OnlineLoggingOption | Literal[False]:
        if value in {"0", "False", "off"}:  # off -> no logging
            return False
        if value in {"1", "True", "on"}:
            return "online"
        return value

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

    def process_args(self) -> None:
        super().process_args()
        logging.getLogger("ray_utilities").setLevel(self.log_level)


class DefaultExtraArgs(Tap):
    extra: Optional[list[str]] = None

    def configure(self) -> None:
        super().configure()
        self.add_argument("--extra", help="extra arguments", nargs="+")


class CheckpointConfigArgumentParser(Tap):
    checkpoint_frequency: NotAModelParameter[int | None] = 50_000
    """
    Frequency of checkpoints in steps (or iterations, see checkpoint_frequency_unit)
    0 or None for no checkpointing
    """

    checkpoint_frequency_unit: NotAModelParameter[Literal["steps", "iterations"]] = "steps"
    """Unit for checkpoint_frequency, either after # steps or iterations"""

    num_to_keep: NotAModelParameter[int | None] = None
    """The number of checkpoints to keep. None to keep all checkpoints."""

    def process_args(self) -> None:
        if self.num_to_keep is not None and self.num_to_keep <= 0:
            raise ValueError(f"num_to_keep must be a positive integer or None. Not {self.num_to_keep}.")
        return super().process_args()


class OptionalExtensionsArgs(RLlibArgumentParser):
    dynamic_buffer: AlwaysRestore[bool] = False
    """Use DynamicBufferCallback. Increases env steps sampled and batch size"""

    dynamic_batch: AlwaysRestore[bool] = False
    """Use dynamic batch, scales batch size via gradient accumulation"""

    iterations: NeverRestore[int | AutoInt | Literal["auto"]] = "auto"
    total_steps: int = 1_000_000
    min_step_size: int = 32
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


def _parse_tune_choices(
    value: str | Literal[False],
) -> Literal["batch_size", "rollout_size", "all", False]:
    return value  # type: ignore[return-value]


class OptunaArgumentParser(Tap):
    optimize_config: NotAModelParameter[NeverRestore[bool]] = (
        False  # legacy argument name; possible replace with --tune later
    )
    tune: NeverRestore[list[Literal["batch_size", "rollout_size", "all"]] | Literal[False]] = False
    """List of dynamic parameters to be tuned"""

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--tune", nargs="+", default=False, choices=["batch_size", "rollout_size", "all"], type=_parse_tune_choices
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


class DefaultArgumentParser(
    SupportsMetaAnnotations,
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
        super().configure()
