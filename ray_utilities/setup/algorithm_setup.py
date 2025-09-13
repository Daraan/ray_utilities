"""Algorithm setup classes for Ray RLlib experiments with dynamic configuration.

This module provides concrete implementations of experiment setups for Ray RLlib
algorithms, with built-in support for dynamic batch sizing, experience buffer
management, and trainable class instantiation.

The main classes extend :class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase`
with algorithm-specific functionality and configuration management, making it easy
to set up and run reinforcement learning experiments with minimal boilerplate code.

Key Components:
    - :class:`AlgorithmSetup`: Base setup class with dynamic buffer and batch size support
    - :class:`PPOSetup`: Ready-to-use setup for PPO algorithm experiments
    - Type definitions for trainable classes and configuration flexibility

These classes integrate with Ray Tune for hyperparameter optimization and provide
a standardized interface for algorithm configuration across different RL algorithms.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from typing_extensions import TypeVar

from ray_utilities.config import add_callbacks_to_config
from ray_utilities.config.create_algorithm import create_algorithm_config
from ray_utilities.setup.experiment_base import (
    AlgorithmType_co,
    ConfigType_co,
    ExperimentSetupBase,
    NamespaceType,
    ParserType_co,
)
from ray_utilities.setup.extensions import SetupWithDynamicBatchSize, SetupWithDynamicBuffer
from ray_utilities.training.default_class import DefaultTrainable

if TYPE_CHECKING:
    from ray.rllib.callbacks.callbacks import RLlibCallback

    from ray_utilities.typing import TrainableReturnData

__all__ = ["AlgorithmSetup", "AlgorithmType_co", "ConfigType_co", "PPOSetup", "ParserType_co"]


TrainableT = TypeVar("TrainableT", bound=Callable[..., "TrainableReturnData"] | type["DefaultTrainable"])
"""TypeVar for the two trainable types. Note that default values of generic DefaultTrainable are applied here"""


class AlgorithmSetup(
    SetupWithDynamicBuffer[ParserType_co, ConfigType_co, AlgorithmType_co],  # use before the other setup
    SetupWithDynamicBatchSize[ParserType_co, ConfigType_co, AlgorithmType_co],
    ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co],
):
    """Concrete base class for Ray RLlib algorithm experiment setups.

    This class provides a complete, ready-to-use implementation of
    :class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase` with
    built-in support for dynamic configuration adjustments. It combines
    multiple mixins to provide dynamic batch sizing and experience buffer
    management capabilities.

    The class serves as the foundation for algorithm-specific setups and
    can be used directly for basic PPO experiments or extended for other
    algorithms by overriding the ``config_class`` and ``algo_class`` attributes.

    Features:
        - Dynamic batch size adjustment based on available resources
        - Dynamic experience buffer sizing for sample efficiency
        - Automatic trainable class creation with proper type hints
        - Integration with Ray Tune for hyperparameter optimization
        - Built-in callback and configuration management

    Attributes:
        config_class: RLlib configuration class (defaults to :class:`ray.rllib.algorithms.ppo.PPOConfig`)
        algo_class: RLlib algorithm class (defaults to :class:`ray.rllib.algorithms.ppo.PPO`)

    Example:
        >>> setup = AlgorithmSetup()
        >>> config = setup.create_config(args)
        >>> trainable = setup._create_trainable()

    Note:
        This is the most basic complete implementation of :class:`ExperimentSetupBase`.
        For production use, consider using :class:`PPOSetup` or creating algorithm-specific
        subclasses with proper ``PROJECT`` names and configuration customizations.

    See Also:
        :class:`PPOSetup`: Specialized setup for PPO algorithms
        :class:`~ray_utilities.setup.extensions.SetupWithDynamicBatchSize`: Dynamic batch sizing mixin
        :class:`~ray_utilities.setup.extensions.SetupWithDynamicBuffer`: Dynamic buffer sizing mixin
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
        return DefaultTrainable.define(
            self,
            model_config=None,  # TODO: allow porting, but check that it works
            log_level=self.args.log_level,
        )

    @classmethod
    def _model_config_from_args(cls, args: NamespaceType[ParserType_co]) -> dict[str, Any] | None:  # noqa: ARG003
        """Returns a model_config to be used with an RLModule. Return None for default option."""
        return None

    @classmethod
    def _config_from_args(cls, args, base: Optional[ConfigType_co] = None) -> ConfigType_co:
        learner_class = None
        if args.accumulate_gradients_every > 1 or args.dynamic_batch:
            # import lazy as currently not used elsewhere
            from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import (  # noqa: PLC0415
                PPOTorchLearnerWithGradientAccumulation,
            )

            learner_class = PPOTorchLearnerWithGradientAccumulation
        config, _module_spec = create_algorithm_config(
            args=args,
            module_class=None,
            catalog_class=None,
            model_config=cls._model_config_from_args(args),
            learner_class=learner_class,
            framework="torch",
            config_class=cls.config_class,
            base_config=base,
        )
        add_callbacks_to_config(config, cls.get_callbacks_from_args(args))
        return config

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        return super()._get_callbacks_from_args(args)


class PPOSetup(AlgorithmSetup[ParserType_co, "PPOConfig", "PPO"]):
    """Specialized setup class for Proximal Policy Optimization (PPO) experiments.

    This class provides a ready-to-use setup specifically configured for PPO
    algorithms in Ray RLlib. It inherits all the dynamic configuration capabilities
    from :class:`AlgorithmSetup` while ensuring type safety with PPO-specific
    algorithm and configuration types.

    The setup automatically configures PPO-specific features like gradient
    accumulation when requested through command-line arguments, and provides
    sensible defaults for PPO experiments.

    Features:
        - Type-safe PPO configuration and algorithm classes
        - Automatic gradient accumulation learner selection
        - Built-in evaluation interval configuration
        - Inherits dynamic batch sizing and buffer management
        - Compatible with Ray Tune hyperparameter optimization

    Attributes:
        config_class: Set to :class:`ray.rllib.algorithms.ppo.PPOConfig`
        algo_class: Set to :class:`ray.rllib.algorithms.ppo.PPO`

    Example:
        >>> setup = PPOSetup()
        >>> parser = setup.create_parser()
        >>> args = parser.parse_args(["--env", "CartPole-v1"])
        >>> config = setup.create_config(args)
        >>> trainable = setup._create_trainable()

    Note:
        This class can be extended to customize PPO configurations and callbacks
        for specific experiment requirements. Override methods like ``create_config``
        or ``_get_callbacks_from_args`` to add custom behavior.

    See Also:
        :class:`AlgorithmSetup`: Base algorithm setup class
        :class:`ray.rllib.algorithms.ppo.PPO`: The PPO algorithm implementation
        :class:`ray.rllib.algorithms.ppo.PPOConfig`: PPO configuration class
    """

    config_class = PPOConfig
    algo_class = PPO


if TYPE_CHECKING:  # check ABC
    AlgorithmSetup()
