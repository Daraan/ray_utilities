from __future__ import annotations

from typing import Any, Optional

from tap import Tap
from typing_extensions import Literal


class _DefaultSetupArgumentParser(Tap):
    agent_type: str
    """Agent Architecture"""

    env_type: str = "cart"
    """Environment to run on"""

    episodes: int = 1000
    """How many episodes"""

    seed: int | None = None
    test: bool = False

    extra: Optional[list[str]] = None

    def configure(self) -> None:
        # Short hand args
        super().configure()
        self.add_argument("-a", "--agent_type")
        self.add_argument("-env", "--env_type")
        self.add_argument("-e", "--episodes")
        self.add_argument("-s", "--seed", default=None, type=int)


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


class DefaultLoggingArgParser(Tap):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    wandb: OnlineLoggingOption = False
    comet: OnlineLoggingOption = False
    comment: Optional[str] = None
    tags: list[str] = []  # noqa: RUF012

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
        logger_choices: list[OnlineLoggingOption] = ["offline", "offline+upload", "online", "off"]
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


class DefaultExtraArgs(Tap):
    extra: Optional[list[str]] = None

    def configure(self) -> None:
        super().configure()
        self.add_argument("--extra", help="extra arguments", nargs="+")


class DefaultArgumentParser(
    _DefaultSetupArgumentParser,
    DefaultResourceArgParser,
    DefaultEnvironmentArgParser,
    DefaultLoggingArgParser,
    DefaultExtraArgs,
):
    def configure(self) -> None:
        return super().configure()
