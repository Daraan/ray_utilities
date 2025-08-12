from __future__ import annotations
# pyright: enableExperimentalFeatures=true

import logging
from typing import Any, Optional, TypeVar

from tap import Tap
from typing_extensions import Annotated, Literal, get_type_hints, get_args, get_origin, Sentinel

from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations, split_timestep_budget

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
RestoreIfDefault = Annotated[_T, "RestoreIf", NO_VALUE]


class SupportsRestoreParser(Tap):
    def configure(self) -> None:
        super().configure()
        complete_annotations = self._get_from_self_and_super(
            extract_func=lambda super_class: dict(get_type_hints(super_class, include_extras=True))
        )
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

    def get_to_restore_values(self):
        return self._always_restore

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
            if current_value is not NO_VALUE and current_value != restored_value:
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


class RLlibArgumentParser(Tap):
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
        if self.minibatch_size > self.train_batch_size_per_learner:
            logger.warning(
                "minibatch_size %d is larger than train_batch_size_per_learner %d, this can result in an error. "
                "Reducing the minibatch_size to the train_batch_size_per_learner.",
                self.minibatch_size,
                self.train_batch_size_per_learner,
            )
            self.minibatch_size = self.train_batch_size_per_learner
        return super().process_args()


class DefaultResourceArgParser(Tap):
    num_jobs: NeverRestore[int] = 5
    """Trials to run in parallel"""

    num_samples: NeverRestore[int] = 1
    """Number of samples to run in parallel, if None, same as num_jobs"""

    gpu: NeverRestore[bool] = False

    parallel: NeverRestore[bool] = False
    """Use multiple CPUs per worker"""

    not_parallel: NeverRestore[bool] = False
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
        self.add_argument("-mp", "--parallel")
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
    wandb: NeverRestore[OnlineLoggingOption] = False
    comet: NeverRestore[OnlineLoggingOption] = False
    comment: Optional[str] = None
    tags: list[str] = []  # noqa: RUF012
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


class DefaultExtraArgs(Tap):
    extra: Optional[list[str]] = None

    def configure(self) -> None:
        super().configure()
        self.add_argument("--extra", help="extra arguments", nargs="+")


class CheckpointConfigArgumentParser(Tap):
    checkpoint_frequency: int | None = 50_000
    """Frequency of checkpoints in steps (or iterations, see checkpoint_frequency_unit), 0 or None for no checkpointing"""

    checkpoint_frequency_unit: Literal["steps", "iterations"] = "steps"
    """Unit for checkpoint_frequency, either after # steps or iterations"""

    num_to_keep: int | None = None
    """The number of checkpoints to keep. None to keep all checkpoints."""

    def process_args(self) -> None:
        if self.num_to_keep is not None and self.num_to_keep <= 0:
            raise ValueError(f"num_to_keep must be a positive integer or None. Not {self.num_to_keep}.")
        return super().process_args()


class OptionalExtensionsArgs(RLlibArgumentParser):
    dynamic_buffer: AlwaysRestore[bool] = False
    """Use DynamicBufferCallback"""

    dynamic_batch: AlwaysRestore[bool] = False
    """Use dynamic batch"""

    iterations: NeverRestore[int | Literal["auto"]] = "auto"
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
            assure_even=self.use_exact_total_steps,
        )
        # eval_intervals = get_dynamic_evaluation_intervals(budget["step_sizes"], batch_size=self.train_batch_size_per_learner, eval_freq=4)
        self.total_steps = budget["total_steps"]
        if self.iterations == "auto":  # for testing reduce this number
            iterations = calculate_iterations(
                dynamic_buffer=self.dynamic_buffer,
                batch_size=self.train_batch_size_per_learner,  # <-- if adjusted manually afterwards iterations will be wrong  # noqa: E501
                total_steps=self.total_steps,
                assure_even=self.use_exact_total_steps,
                min_size=self.min_step_size,
                max_size=self.max_step_size,
            )
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
    optimize_config: NeverRestore[bool] = False  # legacy argument name; possible replace with --tune later
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
    SupportsRestoreParser,
    OptionalExtensionsArgs,  # Needs to be before _DefaultSetupArgumentParser
    RLlibArgumentParser,
    OptunaArgumentParser,
    _DefaultSetupArgumentParser,
    CheckpointConfigArgumentParser,
    DefaultResourceArgParser,
    DefaultEnvironmentArgParser,
    DefaultLoggingArgParser,
    DefaultExtraArgs,
):
    def configure(self) -> None:
        super().configure()
