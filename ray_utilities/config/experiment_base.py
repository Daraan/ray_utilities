from __future__ import annotations

# pyright: enableExperimentalFeatures=true

import argparse
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence, TypedDict
from typing_extensions import TypeForm

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig

class ExperimentSetupBase(ABC):
    """
    Methods:
    - create_parser
    - create_config
    - trainable_from_config
    - trainable_return_type
    """

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default="ddt")
        parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default="cart")
        parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
        parser.add_argument("-s", "--seed", help="Seed", default=None, type=int)
        parser.add_argument("--test", "--dry-run", help="Do not save any models", action="store_true", default=False)
        self._add_resource_args(parser)
        self._add_environment_args(parser)
        self._add_logging_args(parser)

        parser.add_argument("--extra", help="extra arguments", nargs="+", choices=[])
        return parser

    def _add_resource_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("-J", "--num_jobs", help="Amount of jobs the Tuner does start", default=5, type=int)
        parser.add_argument("-gpu", "--gpu", help="run on GPU?", action="store_true")
        parser.add_argument(
            "-mp",
            "--parallel",
            help="Use multiple CPUs per worker",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "-np",
            "--not_parallel",
            help="Do not run multiple models in parallel, i.e. the Tuner will execute one job only. "
            "This is equivalent to num_jobs=1",
            action="store_true",
            default=False,
        )
        return parser

    def _add_environment_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--render_mode",
            "--render",
            nargs="?",
            const="rgb_array",
            help="Render mode",
            type=str,
            default=None,
            choices=["human", "rgb_array", "ansi"],
        )
        return parser

    def _add_logging_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--log_level",
            help="Set the log level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )
        parser.add_argument("--wandb", "-wb", help="Log to WandB", action="store_true", default=False)
        parser.add_argument(
            "--comet",
            nargs="?",
            help="Log to Comet",
            const="1",
            default="off",
            choices=["offline", "offline+upload", "0", "1", "False", "off", "on"],
            type=str,
        )
        parser.add_argument("--comment", "-c", help="Add comment to this run", type=str, default="")
        parser.add_argument(
            "--tags", nargs="+", help="Add tags to this run to be used with wandb and comet", default=()
        )
        return parser

    @abstractmethod
    def create_config(self, args: argparse.Namespace) -> AlgorithmConfig: ...

    @abstractmethod
    def trainable_from_config(self, config: AlgorithmConfig) -> Callable[[dict[str, Any]], dict[str, Any]]: ...

    @property
    @abstractmethod
    def trainable_return_type(self) -> TypeForm[type[TypedDict]] | Sequence[str] | dict[str, Any]:
        """Keys or a TypedDict of the return type of the trainable function."""
        ...
