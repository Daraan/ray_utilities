from __future__ import annotations

import logging
from typing import Any, Optional, get_args

from tap import Tap
from typing_extensions import Literal

from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations, split_timestep_budget

logger = logging.getLogger(__name__)


def _auto_int_transform(x) -> int | Literal["auto"]:
    return int(x) if x != "auto" else x


class _DefaultSetupArgumentParser(Tap):
    agent_type: str = "mlp"
    """Agent Architecture"""

    env_type: str = "cart"
    """Environment to run on"""

    iterations: int | Literal["auto"] = 1000  # NOTE: Overwritten by Extra
    """
    How many iterations to run.

    An iteration consists of *n* iterations over the PPO batch, each further
    divided into minibatches of size `minibatch_size`.
    """
    total_steps: int = 1_000_000  # NOTE: Overwritten by Extra

    seed: int | None = None
    test: bool = False

    extra: Optional[list[str]] = None

    def configure(self) -> None:
        # Short hand args
        super().configure()
        self.add_argument("-a", "--agent_type")
        self.add_argument("-env", "--env_type")
        self.add_argument("--seed", default=None, type=int)
        # self.add_argument("--test", nargs="*", const=True, default=False)
        self.add_argument("--iterations", "-it", default="auto", type=_auto_int_transform)
        self.add_argument("--total_steps", "-ts")


class RLlibArgumentParser(Tap):
    train_batch_size_per_learner: int = 2048  # batch size that ray samples
    minibatch_size: int = 128
    """Minibatch size used for backpropagation/optimization"""

    from_checkpoint: Optional[str] = None

    def configure(self) -> None:
        super().configure()
        self.add_argument(
            "--batch_size",
            dest="train_batch_size_per_learner",
            type=int,
            required=False,
        )

        self.add_argument(
            "--from_checkpoint", "-cp", "-load", default=None, type=str, help="Path to the checkpoint to load from."
        )

    def process_args(self):
        if self.minibatch_size > self.train_batch_size_per_learner:
            logger.error(
                "minibatch_size %d is larger than train_batch_size_per_learner %d, this can result in an error. "
                "Reducing the minibatch_size to the train_batch_size_per_learner.",
                self.minibatch_size,
                self.train_batch_size_per_learner,
            )
            self.minibatch_size = self.train_batch_size_per_learner
        return super().process_args()


class DefaultResourceArgParser(Tap):
    num_jobs: int = 5
    """Trials to run in parallel"""

    num_samples: int
    """Number of samples to run in parallel, if None, same as num_jobs"""

    gpu: bool = False

    parallel: bool = False
    """Use multiple CPUs per worker"""

    not_parallel: bool = False
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
    render_mode: Optional[Literal["human", "rgb_array", "ansi"]] = None
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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    wandb: OnlineLoggingOption = False
    comet: OnlineLoggingOption = False
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


class OptionalExtensionsArgs(RLlibArgumentParser):
    dynamic_buffer: bool = False
    """Use DynamicBufferCallback"""

    dynamic_batch: bool = False
    """Use dynamic batch"""

    iterations: int | Literal["auto"] = "auto"
    total_steps: int = 1_000_000
    min_step_size: int = 32
    """min_dynamic_buffer_size"""
    max_step_size: int = 8192
    """max_dynamic_buffer_size"""

    use_exact_total_steps: bool = False
    """
    If True, the total_steps are a lower bound, independently of dynamic_buffer are they adjusted to
    be divisible by max_step_size and min_step_size. In case of a dynamic buffer, this results in
    evenly distributed fractions of the total_steps size for each dynamic batch size.
    """

    no_exact_sampling: bool = False
    """
    Set to not add the exact_sampling_callback to the AlgorithmConfig.

    If this is True this is Rllib's default behavior which might sample a minor amount of more steps
    than required for the batch_size.
    For exactness this callback will trim the sampled data to the exact batch size.
    """

    keep_masked_samples: bool = False
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
    optimize_config: bool = False  # legacy argument name; possible replace with --tune later
    tune: list[Literal["batch_size", "rollout_size", "all"]] | Literal[False] = False
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
    OptionalExtensionsArgs,  # Needs to be before _DefaultSetupArgumentParser
    RLlibArgumentParser,
    OptunaArgumentParser,
    _DefaultSetupArgumentParser,
    DefaultResourceArgParser,
    DefaultEnvironmentArgParser,
    DefaultLoggingArgParser,
    DefaultExtraArgs,
):
    def configure(self) -> None:
        return super().configure()
