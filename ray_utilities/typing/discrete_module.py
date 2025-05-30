from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from ray.rllib.core.models.tf.base import TfModel
from ray.rllib.core.models.torch.base import TorchModel  # noqa: F401  # type: ignore
from typing_extensions import Self

if TYPE_CHECKING:
    from ray.rllib.core.rl_module.rl_module import RLModuleConfig


class DiscreteModelABC(ABC):
    """
    Model that implements `create_discrete_copy` for discrete evaluation

    TODO:
        Otherwise implement a switch_mode; but might not work that well with logging.
    """

    @abstractmethod
    def create_discrete_copy(self) -> Self: ...


class _DiscreteTorchModelBase(DiscreteModelABC, TorchModel): ...


class _DiscreteTFModelBase(DiscreteModelABC, TfModel): ...


class DiscreteModuleBase(Protocol):
    CAN_USE_DISCRETE_EVAL = True

    if TYPE_CHECKING:

        def __instance_members(self):
            self.is_discrete: bool
            self.inference_only: bool
            self.config: RLModuleConfig

    @abstractmethod
    def switch_mode(self, *, discrete: bool): ...


class _TorchPPOModule(DiscreteModuleBase):
    def __instance_members(self):
        super().__instance_members()
        self.pi: _DiscreteTorchModelBase
        self.vf: _DiscreteTorchModelBase


class _TFPPOModule(DiscreteModuleBase):
    def __instance_members(self):
        super().__instance_members()
        self.pi: _DiscreteTFModelBase
        self.vf: _DiscreteTFModelBase


class DiscreteTorchPPOModule(_TorchPPOModule, DiscreteModuleBase):
    def switch_mode(self, *, discrete: bool):
        assert self.inference_only == self.config.inference_only
        if discrete and not self.is_discrete:
            self.pi = self.pi.create_discrete_copy()
            self.pi.eval()
            if not self.inference_only:  # vf is not used in inference and missing in later ray versions
                self.vf = self.vf.create_discrete_copy()
                self.vf.eval()
            self.is_discrete = True
        elif not discrete and self.is_discrete:
            self.pi = self.pi
            if not self.inference_only:
                self.vf = self.vf
            self.is_discrete = False


class DiscreteTFPPOModule(_TFPPOModule, DiscreteModuleBase):
    def switch_mode(self, *, discrete: bool):
        assert self.inference_only == self.config.inference_only
        if discrete and not self.is_discrete:
            self.pi = self.pi.create_discrete_copy()
            # self.pi.eval()
            if not self.inference_only:  # vf is not used in inference and missing in later ray versions
                self.vf = self.vf.create_discrete_copy()
                # self.vf.eval()
            self.is_discrete = True
        elif not discrete and self.is_discrete:
            self.pi = self.pi
            if not self.inference_only:
                self.vf = self.vf
            self.is_discrete = False
