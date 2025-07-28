from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from typing_extensions import TypeVar

from ray_utilities.config import add_callbacks_to_config
from ray_utilities.config.create_algorithm import create_algorithm_config
from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co, ExperimentSetupBase, ParserType_co
from ray_utilities.setup.extensions import SetupWithDynamicBatchSize, SetupWithDynamicBuffer
from ray_utilities.training.default_class import DefaultTrainable

if TYPE_CHECKING:
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray_utilities.typing import StrictAlgorithmReturnData

    from ray_utilities.typing import TrainableReturnData

__all__ = ["AlgorithmSetup", "AlgorithmType_co", "ConfigType_co", "PPOSetup", "ParserType_co"]


TrainableT = TypeVar("TrainableT", bound=Callable[..., "TrainableReturnData"] | type["DefaultTrainable"])


class AlgorithmSetup(
    SetupWithDynamicBuffer[ParserType_co, ConfigType_co, AlgorithmType_co],
    SetupWithDynamicBatchSize[ParserType_co, ConfigType_co, AlgorithmType_co],
    ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co],
):
    """
    Base class for algorithm setup in Ray RLlib experiments.

    This class is used to define the setup for RLlib algorithms, including configuration and callbacks.
    It inherits from ExperimentSetupBase to provide a common interface for experiment setups.

    Most basic complete ExperimentSetupBase
    """

    PROJECT = "Unnamed Project"
    # FIXME: Need at least PPO to use run_tune with this class
    config_class: type[ConfigType_co] = PPOConfig  # evaluate the forward ref of ConfigType.__default__
    algo_class: type[AlgorithmType_co] = PPO

    @property
    def group_name(self) -> str:
        return "algorithm_setup"

    def _create_trainable(self) -> type[DefaultTrainable[ParserType_co, ConfigType_co, AlgorithmType_co]]:
        """
        Create a trainable instance for the algorithm setup.

        Returns:
            A callable that returns a dictionary with training data.
        """
        return DefaultTrainable.define(self)

    @classmethod
    def _config_from_args(cls, args, base: Optional[ConfigType_co] = None) -> ConfigType_co:
        config, _module_spec = create_algorithm_config(
            args=args,
            module_class=None,
            catalog_class=None,
            model_config=None,
            framework="torch",
            config_class=cls.config_class,
            base=base,
        )
        config.evaluation(evaluation_interval=1)  # required to not fail on the cheap default trainable
        add_callbacks_to_config(config, cls.get_callbacks_from_args(args))
        return config

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        return super()._get_callbacks_from_args(args)


class PPOSetup(AlgorithmSetup[ParserType_co, "PPOConfig", "PPO"]):
    """
    A specific setup for PPO algorithms.
    This class can be extended to customize PPO configurations and callbacks.
    """

    config_class = PPOConfig
    algo_class = PPO


if TYPE_CHECKING:  # check ABC
    AlgorithmSetup()
