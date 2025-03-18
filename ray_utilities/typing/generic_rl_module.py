from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Optional, TypeVar, Union

from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.rl_module import RLModule

if TYPE_CHECKING:
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


_ModelConfig_co = TypeVar("_ModelConfig_co", bound="None | dict | DefaultModelConfig", covariant=True)


class RLModuleWithConfig(RLModule, Generic[_ModelConfig_co]):
    """A RLModule with a config attribute."""

    def __instance_members(self):
        self.model_config: Optional[Union[dict, DefaultModelConfig]]


class CatalogWithConfig(Catalog, Generic[_ModelConfig_co]): ...
