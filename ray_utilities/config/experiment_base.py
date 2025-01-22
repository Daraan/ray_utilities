from __future__ import annotations

# pyright: enableExperimentalFeatures=true

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypedDict, Literal, overload

from ._typed_argument_parser import DefaultArgumentParser

__all__ = [
    "DefaultArgumentParser",
    "ExperimentSetupBase",
]

if TYPE_CHECKING:
    import argparse
    from ray.rllib.algorithms import AlgorithmConfig
    from typing_extensions import TypeForm
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec


class TrainableReturnData(TypedDict, total=False):
    pass


class ExperimentSetupBase(ABC):
    """
    Methods:
    - create_parser
    - create_config
    - trainable_from_config
    - trainable_return_type
    """

    def create_parser(self) -> argparse.ArgumentParser:
        return DefaultArgumentParser()

    @overload
    def create_config(self, args: argparse.Namespace, *, return_module_spec: Literal[False]) -> AlgorithmConfig: ...

    @overload
    def create_config(
        self, args: argparse.Namespace, *, return_module_spec: Literal[True] = True
    ) -> tuple[AlgorithmConfig, RLModuleSpec]: ...

    @abstractmethod
    def create_config(
        self, args: argparse.Namespace, *, return_module_spec: bool = True
    ) -> tuple[AlgorithmConfig, RLModuleSpec] | AlgorithmConfig: ...

    @abstractmethod
    def trainable_from_config(
        self, *, args: argparse.Namespace | DefaultArgumentParser, config: AlgorithmConfig
    ) -> Callable[[dict[str, Any]], TrainableReturnData]: ...

    # @property
    # @abstractmethod
    def _trainable_return_type(
        self,
    ) -> TypeForm[type[TypedDict]] | Sequence[str] | dict[str, Any] | TrainableReturnData:
        """Keys or a TypedDict of the return type of the trainable function."""
        ...
