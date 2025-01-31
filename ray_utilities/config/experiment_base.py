from __future__ import annotations

import logging

# pyright: enableExperimentalFeatures=true
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
    overload,
)

from ray import tune
from ray.rllib.core.rl_module import MultiRLModuleSpec
from tap.tap import Tap
from typing_extensions import TypeVar

from ray_utilities.callbacks import LOG_IGNORE_ARGS, remove_ignored_args

from ._typed_argument_parser import DefaultArgumentParser
from .tuner_setup import TunerSetup

if TYPE_CHECKING:
    import argparse

    import gymnasium as gym
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.utils.typing import EnvType
    # from typing_extensions import TypeForm

__all__ = [
    "DefaultArgumentParser",
    "ExperimentSetupBase",
    "NamespaceType",
    "Parser",
]

logger = logging.getLogger(__name__)

ParserType = TypeVar("ParserType", bound="DefaultArgumentParser", default="DefaultArgumentParser")
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

    default_extra_tags: ClassVar[list[str]] = [
        "dev",
        "<test>",
        "<gpu>",
        "<env_type>",
        "<agent_type>",
    ]
    """extra tags to add if """

    @property
    @abstractmethod
    def project_name(self) -> str:
        """
        Name of the project for logging. Will be used for:
            - output folder
            - wandb project
            - comet workspace
        """

    @property
    @abstractmethod
    def group_name(self) -> str:
        """
        Name of the group for logging. Will be used for:
            - wandb group
            - comet project
        """

    def __init__(self, args: Optional[Sequence[str]] = None, *, init_config: bool = True):
        self.parser: Parser[ParserType]
        self.parser = self.create_parser()
        self.args = self.parse_args(args)
        if init_config:
            self.config: _ConfigType = self.create_config()
            self.param_space = self.create_param_space()
            self.trainable = self.create_trainable()

    # region Argument Parsing

    def create_parser(self) -> Parser[ParserType]:
        self.parser = DefaultArgumentParser()
        return self.parser

    def postprocess_args(self, args: NamespaceType[ParserType]) -> NamespaceType[ParserType]:
        """
        Post-process the arguments.

        Note:
            This is not an abstract method
        """
        return args

    def args_to_dict(self, args: ParserType | argparse.Namespace) -> dict[str, Any]:
        if isinstance(args, Tap):
            return {k: getattr(args, k) for k in args.class_variables}
        return vars(args).copy()

    def get_args(self) -> NamespaceType[ParserType]:
        if not self.args:
            self.args = self.parse_args()
        return self.args

    def parse_args(self, args: Sequence[str] | None = None) -> NamespaceType[ParserType]:
        """
        Raises:
            ValueError: If parse_args is called twice without recreating the parser.
        """
        if not self.parser:
            self.parser = self.create_parser()
        parsed = self.parser.parse_args(args)
        self.args = self.postprocess_args(parsed)
        return self.args

    # endregion

    # region Tags

    def _substitute_tag(self, tag: str):
        if not tag.startswith("<"):
            return tag
        assert tag[-1] == ">", f"Invalid tag parsing format: {tag}. Must be '<argattribute>'"
        tag = tag[1:-1]
        if hasattr(self.args, tag):
            value = getattr(self.args, tag)
            if isinstance(value, bool) or value is None:
                if value:
                    return tag
                return None
            return value
        return None  # error

    def _parse_extra_tags(self, extra_tags: Sequence[str] | None = None) -> list[str]:
        if extra_tags is None:
            extra_tags = self.default_extra_tags.copy()
        else:
            extra_tags = list(extra_tags)
        for i, tag in enumerate(extra_tags):
            subst = self._substitute_tag(tag)
            if not isinstance(subst, str):
                extra_tags[i] = ""
                logger.debug("Could not find tag: %s in the ArgumentParser %s", tag, self.__class__.__name__)
                continue
            extra_tags[i] = subst
        return list(filter(None, extra_tags))

    def create_tags(self, extra_tags: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self.args, "tags"):
            logger.info("Parsed arguments have not attribute tags.")
            return self._parse_extra_tags(extra_tags)
        return [
            *self.args.tags,
            *self._parse_extra_tags(extra_tags),
        ]

    # endregion
    # region hparams

    def clean_args_to_hparams(self, args: Optional[NamespaceType[ParserType]] = None):
        args = args or self.get_args()
        upload_args = remove_ignored_args(args, remove=(*LOG_IGNORE_ARGS, "process_number"))
        return upload_args

    def create_param_space(self) -> dict[str, Any]:
        """
        Create a dict to upload as hyperparameters and pass as first argument to the trainable

        Attention:
            This function must set the `hparams` attribute
        """
        module_spec = self.get_module_spec(copy=False)
        param_space: dict[str, Any] = {
            "env": self.config.env if isinstance(self.config.env, str) else self.config.env.unwrapped.spec.id,  # pyright: ignore[reportOptionalMemberAccess]
            "algo": self.config.algo_class.__name__ if self.config.algo_class is not None else "UNDEFINED",
            "module": module_spec.module_class.__name__ if module_spec.module_class is not None else "UNDEFINED",
            "model_config": module_spec.model_config,
        }
        # WandB might not log them if they are not selected as choice
        param_space = {k: tune.choice([v]) for k, v in param_space.items()}
        # Log CLI args as hyperparameters
        param_space["cli_args"] = self.clean_args_to_hparams(self.args)
        self.param_space = param_space
        return param_space

    # region config and trainable

    @abstractmethod
    def _create_config(self) -> _ConfigType:
        """Creates the config for the experiment."""

    def create_config(self) -> _ConfigType:
        """
        Creates the config for the experiment.

        Attention:
            Do not overwrite this method. Overwrite _create_config instead.
        """
        self.config = self._create_config()
        return self.config

    @overload
    def get_module_spec(self, *, copy: Literal[False]) -> RLModuleSpec: ...

    @overload
    def get_module_spec(
        self,
        *,
        copy: Literal[True],
        env: Optional[EnvType] = None,
        spaces: Optional[dict[str, gym.Space]] = None,
        inference_only: Optional[bool] = None,
    ) -> RLModuleSpec: ...

    def get_module_spec(
        self,
        *,
        copy: bool,
        env: Optional[EnvType] = None,
        spaces: Optional[dict[str, gym.Space]] = None,
        inference_only: Optional[bool] = None,
    ) -> RLModuleSpec:
        if not self.config:
            raise ValueError("Config not defined yet, call create_config first.")
        if copy:
            return self.config.get_rl_module_spec(env, spaces, inference_only)
        if self.config._rl_module_spec is None:
            raise ValueError("ModuleSpec not defined yet, call config.rl_module first.")
        assert not isinstance(self.config._rl_module_spec, MultiRLModuleSpec)
        return self.config._rl_module_spec

    def create_config_and_module_spec(
        self,
        *,
        env: Optional[EnvType] = None,
        spaces: Optional[dict[str, gym.Space]] = None,
        inference_only: Optional[bool] = None,
    ) -> tuple[_ConfigType, RLModuleSpec]:
        """
        Creates the config and module spec for the experiment.

        Warning:
            The returned module_spec can be a copy. Modifying it will not result in a change when
            calling config.build() again.
        """
        config = self.create_config()
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

    @abstractmethod
    def create_trainable(self) -> Callable[[dict[str, Any]], TrainableReturnData]:
        """Return a trainable for the Tuner to use."""

    # endregion

    # region Tuner

    def create_tuner(self) -> tune.Tuner:
        return TunerSetup(setup=self).create_tuner()

    # endregion

    # Currently cannot use TypeForm[type[TypedDict]] as it is not included in the typing spec.
    # @property
    # @abstractmethod
    #
    # def _trainable_return_type(
    #    self,
    # ) -> TypeForm[type[TypedDict]] | Sequence[str] | dict[str, Any] | TrainableReturnData:
    #    """Keys or a TypedDict of the return type of the trainable function."""
