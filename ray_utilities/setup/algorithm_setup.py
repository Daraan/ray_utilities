from __future__ import annotations

from typing import TYPE_CHECKING

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray_utilities.config import DefaultArgumentParser, add_callbacks_to_config
from ray_utilities.config.create_algorithm import create_algorithm_config
from ray_utilities.setup import ExperimentSetupBase
from ray_utilities.setup.extensions import SetupWithDynamicBatchSize, SetupWithDynamicBuffer

if TYPE_CHECKING:
    from ray.rllib.callbacks.callbacks import RLlibCallback

    from ray_utilities.typing import TrainableReturnData


class AlgorithmSetup(
    SetupWithDynamicBuffer, SetupWithDynamicBatchSize, ExperimentSetupBase[DefaultArgumentParser, AlgorithmConfig]
):
    """
    Base class for algorithm setup in Ray RLlib experiments.

    This class is used to define the setup for RLlib algorithms, including configuration and callbacks.
    It inherits from ExperimentSetupBase to provide a common interface for experiment setups.

    Most basic complete ExperimentSetupBase
    """

    PROJECT = "Unnamed Project"

    @property
    def group_name(self) -> str:
        return "algorithm_setup"

    def create_trainable(self):
        """
        Create a trainable instance for the algorithm setup.

        Returns:
            A callable that returns a dictionary with training data.
        """

        def trainable(params) -> TrainableReturnData:  # noqa: ARG001
            # This is a placeholder for the actual implementation of the trainable.
            # It should return a dictionary with training data.
            return self.config.build().train()  # type: ignore

        return trainable

    @classmethod
    def _config_from_args(cls, args):
        config, _module_spec = create_algorithm_config(
            args=args,
            module_class=None,
            catalog_class=None,
            model_config=None,
            framework="torch",
        )
        add_callbacks_to_config(config, cls.get_callbacks_from_args(args))
        return config

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        return super()._get_callbacks_from_args(args)


if TYPE_CHECKING:  # check ABC
    AlgorithmSetup()
