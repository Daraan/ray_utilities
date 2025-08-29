from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from ray_utilities.config.model_config_parsers import MLPConfigParser
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.setup.algorithm_setup import AlgorithmSetup, PPOSetup
from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co

if TYPE_CHECKING:
    from argparse import Namespace

    from ray.rllib.algorithms.ppo import PPO, PPOConfig


class MLPArgumentParser(MLPConfigParser, DefaultArgumentParser):
    pass


ParserType_co = TypeVar("ParserType_co", covariant=True, bound=MLPArgumentParser)


class MLPSetup(AlgorithmSetup[ParserType_co, ConfigType_co, AlgorithmType_co]):
    """Setup for MLP-based algorithms."""

    def create_parser(self):
        self.parser = MLPArgumentParser(allow_abbrev=False)
        return self.parser

    @classmethod
    def _model_config_from_args(cls, args: Namespace | ParserType_co) -> dict[str, Any] | None:
        base = super()._model_config_from_args(args) or {}
        return base | {
            # Use Attributes from MLPConfigParser for the choice
            k: getattr(args, k)
            for k in MLPConfigParser().parse_args([]).as_dict().keys()
            if not k.startswith("_") and hasattr(args, k)
        }


class PPOMLPSetup(PPOSetup[ParserType_co], MLPSetup[ParserType_co, "PPOConfig", "PPO"]):
    """Setup for MLP-based PPO algorithms."""


if TYPE_CHECKING:  # Check ABC
    MLPSetup()
