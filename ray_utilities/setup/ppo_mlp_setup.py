from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypeVar

from ray_utilities.config.parser.mlp_argument_parser import MLPArgumentParser, SimpleMLPParser
from ray_utilities.setup.algorithm_setup import AlgorithmSetup, PPOSetup
from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co

if TYPE_CHECKING:
    from argparse import Namespace

    from ray.rllib.algorithms.ppo import PPO, PPOConfig


ParserType_co = TypeVar("ParserType_co", covariant=True, bound="MLPArgumentParser", default="MLPArgumentParser")


class MLPSetup(AlgorithmSetup[ParserType_co, ConfigType_co, AlgorithmType_co]):
    """Setup for MLP-based algorithms."""

    def create_parser(self, config_files=None):
        self.parser = MLPArgumentParser(config_files=config_files, allow_abbrev=False)
        return self.parser

    @classmethod
    def _model_config_from_args(cls, args: Namespace | ParserType_co) -> dict[str, Any] | None:
        base = super()._model_config_from_args(args) or {}
        return base | {
            # Use Attributes from SimpleMLPParser for the choice
            k: getattr(args, k)
            for k in SimpleMLPParser().parse_args([]).as_dict().keys()
            if not k.startswith("_") and hasattr(args, k)
        }


class PPOMLPSetup(PPOSetup[ParserType_co], MLPSetup[ParserType_co, "PPOConfig", "PPO"]):
    """Setup for MLP-based PPO algorithms."""


if TYPE_CHECKING:  # Check ABC
    MLPSetup()
