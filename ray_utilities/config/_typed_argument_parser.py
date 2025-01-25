from __future__ import annotations

from typing import Optional
from typing_extensions import Literal
from tap import Tap


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


class DefaultResssourceArgParser(Tap):
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


class DefaultLoggingArgParser(Tap):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    wandb: bool = False
    comet: Literal["offline", "offline+upload", "on", False] = False
    comment: Optional[str] = None
    tags: list[str] = []  # noqa: RUF012

    def _parse_comet(
        self, value: Literal["offline", "offline+upload", "on"]
    ) -> Literal["offline", "offline+upload", "on", False]:
        if value in ("0", "False", "off"):
            return False
        if value in ("1", "True", "on"):
            return "on"
        return value

    def configure(self) -> None:
        super().configure()
        self.add_argument("--log_level")
        self.add_argument("--wandb", "-wb")
        self.add_argument(
            "--comet",
            nargs="?",
            const="on",
            default="off",
            choices=["offline", "offline+upload", "off", "on"],
            type=self._parse_comet,
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
    DefaultResssourceArgParser,
    DefaultEnvironmentArgParser,
    DefaultLoggingArgParser,
    DefaultExtraArgs,
):
    def configure(self) -> None:
        return super().configure()
