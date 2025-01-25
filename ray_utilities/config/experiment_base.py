from __future__ import annotations

import logging

# pyright: enableExperimentalFeatures=true
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Sequence, TypeAlias, TypedDict

from typing_extensions import TypeVar

from ._typed_argument_parser import DefaultArgumentParser

if TYPE_CHECKING:
    import argparse

    import gymnasium as gym
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.env.multi_agent_env import EnvType
    from tap.tap import Tap
    from typing_extensions import Never, NoReturn, TypeForm

__all__ = [
    "DefaultArgumentParser",
    "ExperimentSetupBase",
    "NamespaceType",
    "Parser",
]

logger = logging.getLogger(__name__)

ParserType = TypeVar("ParserType", bound="Tap", default="DefaultArgumentParser")
Parser: TypeAlias = "argparse.ArgumentParser | ParserType"
NamespaceType: TypeAlias = "argparse.Namespace | ParserType"  # Generic

_ConfigType = TypeVar("_ConfigType", bound="AlgorithmConfig")


class TrainableReturnData(TypedDict, total=False):
    pass


class ExperimentSetupBase(ABC, Generic[_ConfigType, ParserType]):
    """
    Methods:
    - create_parser
    - create_config
    - trainable_from_config
    - trainable_return_type
    """

    def __new__(cls: Never) -> NoReturn:
        raise NotImplementedError("This class is not meant to be instantiated.")

    @classmethod
    def create_parser(cls) -> Parser[ParserType]:
        return DefaultArgumentParser()

    @classmethod
    def postprocess_args(cls, args: NamespaceType[ParserType]) -> NamespaceType[ParserType]:
        """
        Post-process the arguments.

        Note:
            This is not an abstract method
        """
        return args

    @classmethod
    def parse_args(cls, args: Sequence[str] | None = None) -> NamespaceType[ParserType]:
        parser = cls.create_parser()
        parsed = parser.parse_args(args)
        return cls.postprocess_args(parsed)

    @classmethod
    @abstractmethod
    def create_config(cls, args: NamespaceType[ParserType]) -> _ConfigType:
        """Creates the config for the experiment."""

    @classmethod
    def create_config_and_module_spec(
        cls,
        args: NamespaceType[ParserType],
        *,
        env: Optional[EnvType] = None,
        spaces: Optional[dict[str, gym.Space]] = None,
        inference_only: Optional[bool] = None,
    ) -> tuple[_ConfigType, RLModuleSpec]:
        """Creates the config and module spec for the experiment."""
        config = cls.create_config(args)
        module_spec = config.get_rl_module_spec(env=env, spaces=spaces, inference_only=inference_only)
        if not module_spec.action_space:
            logger.warning(
                "No action space found in the module spec. "
                "Adjust your create_config method or pass env or spaces to create_config_and_module_spec."
            )
        if not module_spec.observation_space:
            logger.warning(
                "No observation space found in the module spec. "
                "Adjust your create_config method or pass env or spaces to create_config_and_module_spec."
            )
        return config, module_spec

    @classmethod
    @abstractmethod
    def trainable_from_config(
        cls, *, args: NamespaceType[ParserType], config: _ConfigType
    ) -> Callable[[dict[str, Any]], TrainableReturnData]:
        """Return a trainable for the Tuner to use."""

    # Currently cannot use TypeForm[type[TypedDict]] as it is not included in the typing spec.
    # @property
    # @abstractmethod
    # @classmethod
    # def _trainable_return_type(
    #    cls,
    # ) -> TypeForm[type[TypedDict]] | Sequence[str] | dict[str, Any] | TrainableReturnData:
    #    """Keys or a TypedDict of the return type of the trainable function."""
