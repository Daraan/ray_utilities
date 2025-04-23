from __future__ import annotations

import math
from typing import Any, Optional, get_args

from tap import Tap
from typing_extensions import Literal

from ray_utilities.dynamic_buffer_update import calculate_total_steps

def _auto_int_transform(x):
    return int(x) if x != "auto" else x

class _DefaultSetupArgumentParser(Tap):
    agent_type: str
    """Agent Architecture"""

    env_type: str = "cart"
    """Environment to run on"""

    episodes: int = 1000
    """How many episodes"""

    iterations: int | Literal["auto"] = 1000  # NOTE: Overwritten by Extra
    """
    How many iterations to run.

    An iteration consists of *n* iterations over the PPO batch, each further
    divided into minibatches of size `minibatch_size`.
    """
    total_steps: int | Literal["auto"] = 1_000_000  # NOTE: Overwritten by Extra

    seed: int | None = None
    test: bool = False

    extra: Optional[list[str]] = None

    def configure(self) -> None:
        # Short hand args
        super().configure()
        self.add_argument("-a", "--agent_type")
        self.add_argument("-env", "--env_type")
        self.add_argument("-e", "--episodes")
        self.add_argument("--seed", default=None, type=int)
        # self.add_argument("--test", nargs="*", const=True, default=False)
        self.add_argument("--iterations", "-it", default="auto", type=_auto_int_transform)
        self.add_argument("--total_steps", "-ts", default="auto", type=_auto_int_transform)

class RLlibArgumentParser(Tap):
    train_batch_size_per_learner: int = 4096  # batch size that ray samples
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

class DefaultResourceArgParser(Tap):
    num_jobs: int = 5
    """Trials to run in parallel"""

    gpu: bool = False

    parallel: bool = False
    """Use multiple CPUs per worker"""

    not_parallel: bool = False
    """
    Do not run multiple models in parallel, i.e. the Tuner will execute one job only.
    This is similar to num_jobs=1, but one might skip the Tuner setup.
    """

    def configure(self) -> None:
        super().configure()
        self.add_argument("-J", "--num_jobs")
        self.add_argument("-gpu", "--gpu")
        self.add_argument("-mp", "--parallel")
        self.add_argument("-np", "--not_parallel")


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
            `make_seeded_env_callback(run_seed)`
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

class OptionalExtenionsArgs(RLlibArgumentParser):
    dynamic_buffer: bool = False
    """Use DynamicBufferCallback"""
    # static_batch: bool = True

    iterations: int | Literal["auto"] = "auto"
    total_steps: int | Literal["auto"] = "auto"
    _total_steps_default: int = 1_000_000

    def process_args(self) -> None:
        super().process_args()
        increases = 8
        if self.iterations == "auto":  # for testing reduce this number
            if self.dynamic_buffer:  # TODO: should orient on batch_size
                # TODO: Should be implemented by the choosen Callback method
                low = -math.floor((increases-1) / 2)
                up = math.ceil((increases+1) / 2)
                exponents = list(range(low, up))
                batch_sizes = [2**i * self.train_batch_size_per_learner for i in exponents]
                avg_batch_sizes = sum(batch_sizes) / len(exponents)
                iterations = math.ceil(self._total_steps_default / avg_batch_sizes)
            else:
                iterations = math.ceil(self._total_steps_default / self.train_batch_size_per_learner)
        else:
            iterations = self.iterations
        if self.total_steps == "auto":
            self.total_steps = calculate_total_steps(
                training_iterations=iterations,
                initial_steps=self.train_batch_size_per_learner,
                dynamic_buffer=self.dynamic_buffer,
                increases=8,
            )
            if self.iterations == "auto" and self.total_steps < self._total_steps_default:
                iterations += 1
                self.total_steps += batch_sizes[-1] if self.dynamic_buffer else self.train_batch_size_per_learner  # pyright: ignore[reportPossiblyUnboundVariable]
        self.iterations = iterations

class DefaultArgumentParser(
    OptionalExtenionsArgs,  # Needs to be before _DefaultSetupArgumentParser
    RLlibArgumentParser,
    _DefaultSetupArgumentParser,
    DefaultResourceArgParser,
    DefaultEnvironmentArgParser,
    DefaultLoggingArgParser,
    DefaultExtraArgs,
):
    def configure(self) -> None:
        return super().configure()
