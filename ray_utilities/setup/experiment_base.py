from __future__ import annotations

import logging
import os
from pathlib import Path
import pickle
from abc import ABC, abstractmethod

# pyright: enableExperimentalFeatures=true
from inspect import isclass
from types import SimpleNamespace
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
    cast,
    final,
    overload,
)

import ray
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec
from tap.tap import Tap
from typing_extensions import Self, TypedDict, TypeVar

from ray_utilities.callbacks import LOG_IGNORE_ARGS, remove_ignored_args
from ray_utilities.comet import CometArchiveTracker
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN
from ray_utilities.environment import create_env
from ray_utilities.misc import get_trainable_name
from ray_utilities.setup.tuner_setup import TunerSetup
from ray_utilities.training.default_class import TrainableBase

if TYPE_CHECKING:
    import argparse

    import gymnasium as gym
    import ray.tune.search.sample  # noqa: TC004  # present at runtime from import ray.tune
    from ray.rllib.algorithms import PPO, Algorithm
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.utils.typing import EnvType

    from ray_utilities.training.default_class import TrainableBase
    from ray_utilities.typing import TrainableReturnData

    # from typing_extensions import TypeForm

__all__ = [
    "AlgorithmType_co",
    "ConfigType_co",
    "DefaultArgumentParser",
    "ExperimentSetupBase",
    "NamespaceType",
    "ParserType_co",
]

logger = logging.getLogger(__name__)

ParserType_co = TypeVar("ParserType_co", bound="DefaultArgumentParser", covariant=True, default="DefaultArgumentParser")
"""TypeVar for the ArgumentParser type of a Setup, bound and defaults to DefaultArgumentParser."""

Parser: TypeAlias = "argparse.ArgumentParser | ParserType_co"
NamespaceType: TypeAlias = "argparse.Namespace | ParserType_co"  # Generic, formerly union with , prefer duck-type

ConfigType_co = TypeVar("ConfigType_co", bound="AlgorithmConfig", covariant=True, default="AlgorithmConfig")
"""TypeVar for the AlgorithmConfig type of a Setup, e.g. PPOConfig, DQNConfig, etc; defaults to AlgorithmConfig."""

AlgorithmType_co = TypeVar("AlgorithmType_co", bound="Algorithm", covariant=True, default="PPO")
"""TypeVar for the Algorithm type of a Setup, e.g. PPO, DQN, etc; defaults to PPO."""

_MaybeNone = Any
"""Attribute might be None when trainable is not set up"""


class SetupCheckpointDict(TypedDict, Generic[ParserType_co, ConfigType_co, AlgorithmType_co]):
    """
    TypedDict for the setup checkpoint.
    Contains the args, config, param_space, and setup_class.
    """

    args: ParserType_co
    """Duck-typed SimpleNamespace"""
    param_space: dict[str, Any]
    setup_class: type[ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]]
    config: ConfigType_co | Literal[False]
    __init_config__: bool
    """If True, the config is initialized from the args, `config` is ignored and should be unset"""


class ExperimentSetupBase(ABC, Generic[ParserType_co, ConfigType_co, AlgorithmType_co]):
    """
    Base class for experiment setup, providing a framework for argument parsing, configuration,
    and trainable algorithm instantiation to be compatible with ray.tune.Tuner.

    This class is intended to be subclassed for specific experiment setups.

    Methods:
    - create_parser
    - create_config

    Attributes:
        PROJECT: A string used by `project_name` to determine the project name,
            this is `Unnamed Project` by default, if not changed a warning is logged.

    Generics:
        ParserType: Type of the ArgumentParser, e.g. DefaultArgumentParser
        ConfigType_co: Type of the AlgorithmConfig, e.g. PPOConfig
        AlgorithmType_co: Type of the Algorithm, e.g. PPO
    """

    default_extra_tags: ClassVar[list[str]] = [
        "dev",
        "<test>",
        "<gpu>",
        "<env_type>",
        "<agent_type>",
    ]
    """extra tags to add if """

    PROJECT: str = "Unnamed Project"

    config_class: type[ConfigType_co]
    algo_class: type[AlgorithmType_co]

    _retrieved_callbacks = False

    parse_known_only: ClassVar[bool] = True
    """If True does not fail on unrecognized arguments, will print a warning instead"""

    _fixed_argv: ClassVar[list[str] | None] = None
    """When using remote (no sys.args available) and checkpoints fix the args to the time of creation"""

    @property
    def project_name(self) -> str:
        """Name for the output folder, wandb project, and comet workspace."""
        if self.PROJECT == "Unnamed Project":
            logger.warning(
                "Setup class %s has no custom PROJECT attribute set to determine `project_name`.",
                self.__class__.__name__,
            )
        return "dev-workspace" if self.args.test else self.PROJECT

    @project_name.setter
    def project_name(self, value: str):
        logger.warning("Setting project name to %s. Prefer creation of a new class", value)
        self.PROJECT: str = value

    @property
    @abstractmethod
    def group_name(self) -> str:
        """
        Name of the group for logging. Will be used for:
            - wandb group
            - comet project
        """

    def __init__(
        self,
        args: Optional[Sequence[str]] = None,
        *,
        init_config: bool = True,
        init_param_space: bool = True,
        init_trainable: bool = True,
        parse_args: bool = True,
    ):
        """
        Initializes the experiment base class with optional argument parsing and setup.

        Args:
            args : Command-line arguments to parse. If None, defaults to sys.argv.
            init_config : Whether to initialize the configuration. Defaults to True.
            init_param_space : Whether to initialize the parameter space. Defaults to True.
            init_trainable : Whether to initialize the trainable component. Defaults to True.
                Note:
                    When the setup creates a trainable that is a class, the config is frozen to
                    prevent potentially unforwarded changes between setup.config the config of the
                    trainable. Use `init_trainable=False` or `unset_trainable()`, edit the config
                    and restore the trainable class with a call to `create_trainable()`.
            parse_args : Whether to parse the provided arguments. Defaults to True.

        Attributes:
            parser (Parser[ParserType_co]): The argument parser instance.

        Calls:
            - self.create_parser(): Creates and assigns the argument parser.
            - self.setup(): Performs further setup based on initialization flags.
        """
        self.parser: Parser[ParserType_co]
        self.parser = self.create_parser()
        self.setup(
            args,
            init_config=init_config,
            init_param_space=init_param_space,
            init_trainable=init_trainable,
            parse_args=parse_args,
        )

    def setup(
        self,
        args: Optional[Sequence[str]] = None,
        *,
        init_config: bool = True,
        init_param_space: bool = True,
        init_trainable: bool = True,
        parse_args: bool = True,
    ):
        if parse_args:
            self.args = self.parse_args(args or self._fixed_argv, known_only=self.parse_known_only)
        if init_config:
            self.config: ConfigType_co = self.create_config()
        if hasattr(self, "args"):
            self._set_dynamic_parameters_to_tune()
        if init_trainable:
            self.create_trainable()
        else:
            self.trainable = None
        if init_param_space:
            # relies on trainable to get its name
            self.param_space: dict[str, Any] | _MaybeNone = self.create_param_space()
        if hasattr(self, "args") and self.args.comet:
            self.comet_tracker = CometArchiveTracker()
        else:
            self.comet_tracker = None

    # region Argument Parsing

    def create_parser(self) -> Parser[ParserType_co]:
        self.parser = DefaultArgumentParser(allow_abbrev=False)
        return self.parser

    def postprocess_args(self, args: NamespaceType[ParserType_co]) -> NamespaceType[ParserType_co]:
        """
        Post-process the arguments.

        Note:
            This is not an abstract method
        """
        init_env = create_env(args.env_type)
        env_name = init_env.unwrapped.spec.id  # pyright: ignore[reportOptionalMemberAccess]
        args.env_type = env_name
        return args

    def args_to_dict(self, args: Optional[NamespaceType[ParserType_co] | dict[str, Any]] = None) -> dict[str, Any]:
        if args is None:
            args = self.args
        if isinstance(args, Tap):
            return {k: getattr(args, k) for k in args.class_variables}
        if isinstance(args, dict):
            return args.copy()
        return vars(args).copy()

    def get_args(self) -> NamespaceType[ParserType_co]:
        """Get the parsed arguments or parse them if not already done."""
        if not self.args:
            self.args = self.parse_args(known_only=self.parse_known_only)
        return self.args

    def parse_args(
        self, args: Sequence[str] | None = None, *, known_only: bool | None = None, checkpoint: Optional[str] = None
    ) -> NamespaceType[ParserType_co]:
        """
        Raises:
            ValueError: If parse_args is called twice without recreating the parser.
        """
        if known_only is None:
            known_only = self.parse_known_only
        if not self.parser:
            self.parser = self.create_parser()
        try:
            # If Tap parser or compatible
            self.parser = cast("ParserType_co", self.parser)
            parsed = self.parser.parse_args(args, known_only=known_only)
            extra_args = self.parser.extra_args
        except TypeError as e:
            if "'known_only' is an invalid invalid keyword" not in str(e):
                raise
            if known_only:
                parsed, extra_args = self.parser.parse_known_args(args)
            else:
                parsed = self.parser.parse_args(args)
                extra_args = None
        if extra_args:
            logger.warning(
                "The following arguments were not recognized by the parser: %s.",
                extra_args,
            )
        # Merge args from a checkpoint:
        checkpoint = checkpoint or parsed.from_checkpoint
        if checkpoint:
            path = Path(checkpoint)
            with open(path / "state.pkl", "rb") as f:
                state: dict[str, Any] = pickle.load(f)
            # Create a patched parser with the old values as default values
            new_default_parser = self.create_parser()
            restored_args: dict[str, Any] = vars(state["setup"]["args"])
            actions: list[argparse.Action] = new_default_parser._actions
            for action in actions:
                action.default = restored_args.get(action.dest, action.default)  # set new default values
            self.parser = new_default_parser
            parsed = self.parser.parse_args()

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
                # Info if a tag is not found, this could be due to a not provided bool argument
                # e.g. --gpu is not provided
                logger.debug(
                    "Could not find tag: %s in the ArgumentParser %s, or it is not set", tag, self.__class__.__name__
                )
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

    def clean_args_to_hparams(self, args: Optional[NamespaceType[ParserType_co]] = None) -> dict[str, Any]:
        args = args or self.get_args()
        upload_args = remove_ignored_args(args, remove=(*LOG_IGNORE_ARGS, "process_number"))
        return upload_args

    def get_trainable_name(self) -> str:
        trainable = getattr(self, "trainable", None)
        if trainable is None:
            logger.debug(
                "get_trainable_name called before trainable is set. "
                "Cannot set its name yet, relying on create_trainable to set it."
            )
            return "UNDEFINED"
        return get_trainable_name(trainable)

    def sample_params(self):
        params = self.create_param_space()
        return {k: v.sample() if isinstance(v, ray.tune.search.sample.Domain) else v for k, v in params.items()}

    def _set_dynamic_parameters_to_tune(self):
        """Call before calling `super().create_param_space()` when making use of self.args.tune"""
        if self.args.tune is False:
            self._dynamic_parameters_to_tune: list[str | Any] = []
            return
        if not hasattr(self, "_dynamic_parameters_to_tune"):
            self._dynamic_parameters_to_tune = self.args.tune.copy()

    def _check_tune_arguments_resolved(self):
        if not self.args.tune:
            return
        if not hasattr(self, "_dynamic_parameters_to_tune"):
            logger.warning("_dynamic_parameters_to_tune not set")
            return
        add_all = "all" in self._dynamic_parameters_to_tune
        if add_all:
            if len(self._dynamic_parameters_to_tune) > 1 or len(self.args.tune) > 1:
                raise ValueError("Cannot use 'all' with other tune parameters.", self._dynamic_parameters_to_tune)
            self._dynamic_parameters_to_tune.clear()
        if len(self._dynamic_parameters_to_tune) > 0:
            logger.warning(
                "Unused dynamic tuning parameters: %s "
                "Call self._set_dynamic_parameters_to_tune() and remove parameters "
                "from self._dynamic_parameters_to_tune before calling super().create_param_space().",
                self._dynamic_parameters_to_tune,
            )

    def create_param_space(self) -> dict[str, Any]:
        """
        Create a dict to upload as hyperparameters and pass as first argument to the trainable

        Attention:
            This function must set the `param_space` attribute
        """
        self._check_tune_arguments_resolved()
        module_spec = self._get_module_spec(copy=False)
        if module_spec:
            module = module_spec.module_class.__name__ if module_spec.module_class is not None else "UNDEFINED"
        else:
            module = None
        # Arguments reported on the CLI
        param_space: dict[str, Any] = {
            "env": (
                self.config.env if isinstance(self.config.env, str) else self.config.env.unwrapped.spec.id  # pyright: ignore[reportOptionalMemberAccess]
            ),  # pyright: ignore[reportOptionalMemberAccess]
            "algo": self.config.algo_class.__name__ if self.config.algo_class is not None else "UNDEFINED",
            "module": module,
            "trainable_name": self.get_trainable_name(),  # "UNDEFINED" is called before create_trainable
        }
        # If not logged in choice will not be reported in the CLI interface
        param_space = {k: tune.choice([v]) for k, v in param_space.items()}
        if self.args.seed is not None:
            param_space["env_seed"] = tune.randint(0, 2**16)
            # param_space["run_seed"] = tune.randint(0, 2**16)  # potential seed for config

        # Other args not shown in the CLI
        # NOTE: This is None when the Old API / no module_spec is used!
        param_space["model_config"] = module_spec and module_spec.model_config  # NOTE: Currently unused
        # Log CLI args as hyperparameters
        param_space["cli_args"] = self.clean_args_to_hparams(self.args)
        self.param_space = param_space
        del self._dynamic_parameters_to_tune
        return param_space

    # endregion

    # region config and trainable

    def _create_config(self):
        # Overwrite if config_from_args is not sufficient.
        return self.config_from_args(self.args)

    def _learner_config_dict_defaults(self):
        """Sets values in the learner_config_dict that are used in this packages if not already set."""
        assert self.config, "Config not defined yet, call create_config first."
        self.config.learner_config_dict.setdefault("_debug_connectors", False)
        self.config.learner_config_dict.setdefault("remove_masked_samples", False)
        self.config.learner_config_dict.setdefault("accumulate_gradients_every", 1)

    @final
    def create_config(self) -> ConfigType_co:
        """
        Creates the config for the experiment.

        Attention:
            Do not overwrite this method. Overwrite _create_config / config_from_args instead.
        """
        self.config = self._create_config()
        self._learner_config_dict_defaults()
        # classmethod, but _retrieved_callbacks might be set on instance
        self._check_callbacks_requested.__func__(self)  # pyright: ignore[reportFunctionMemberAccess]
        self._retrieved_callbacks = False  # Reset for next call
        type(self)._retrieved_callbacks = False
        return self.config

    @classmethod
    @abstractmethod
    def _config_from_args(cls, args: ParserType_co | argparse.Namespace) -> ConfigType_co:
        """
        Create an algorithm configuration; similar to `create_config` but as a `classmethod`.

        Tip:
            This method is useful if you do not have access to the setup instance, e.g.
            inside the trainable function.

        Usage:
            .. code-block:: python

                algo: AlgorithmConfig = Setup.config_from_args(args)

            The easiest way to write this method is to use:

            ```python
            config, _spec = create_algorithm_config(
                args,
                env_type=args.env_type,
                module_class=YourModuleClass,
                catalog_class=YourCatalogClass,
                model_config=args.as_dict() if hasattr(args, "as_dict") else vars(args).copy(),
                framework="torch",  # or "tf2"; "jax" not supported
                discrete_eval=False,
            )
            add_callbacks_to_config(config, cls.get_callbacks_from_args(args))
            ```
        """

    @final
    @classmethod
    def config_from_args(cls, args: NamespaceType[ParserType_co]) -> ConfigType_co:
        """
        Create an algorithm configuration; similar to `create_config` but as a `classmethod`.

        Tip:
            This method is useful if you do not have access to the setup instance, e.g.
            inside the trainable function.

        Usage:
            .. code-block:: python

                algo: AlgorithmConfig = Setup.config_from_args(args)
        """
        config = cls._config_from_args(args)
        # callbacks should be added in _config_from_args; but might be easier done here
        cls._check_callbacks_requested()
        # do not reset as we also check in create_config

        # sanity check if class aligns
        if not isinstance(config, cls.config_class):
            logger.error(
                "The class of the config returned by _config_from_args (%s) "
                "does not match the expected config class of the Setup %s.",
                type(config),
                cls.config_class,
            )
        if config.algo_class and not issubclass(config.algo_class, cls.algo_class):
            logger.error(
                "The algo_class of the config returned by _config_from_args (%s) "
                "is not subclass of the expected algo_class of the Setup %s.",
                config.algo_class,
                cls.algo_class,
            )
        elif config.algo_class is None and cls.algo_class is not None:  # pyright: ignore[reportUnnecessaryComparison]
            logger.warning(
                "The algo_class of the config returned by _config_from_args is None. "
                "This is unexpected, it should match the one defined in the Setup class (%s).",
                cls.algo_class,
            )
        return config

    @classmethod
    def algorithm_from_checkpoint(cls, path: str) -> AlgorithmType_co:
        # Algorithm.from_checkpoint is not typed as Self, but as Algorithm

        try:
            algo_class_from_config = cls.config_class().algo_class
        except Exception:
            logger.exception("Error getting algo_class from config class %s", cls.config_class)
            algo_class_from_config = cls.algo_class
        if cls.algo_class != algo_class_from_config:
            logger.error(
                "The algo_class of the config (%s) does not match the algo_class of the Setup (%s). "
                "This may lead to unexpected behavior. Using the algo_class from the Setup class.",
                cls.config_class().algo_class,
                cls.algo_class,
            )
        try:
            # Algorithm checkpoint is likely in subdir.
            return cast("AlgorithmType_co", cls.algo_class.from_checkpoint(os.path.join(path, "algorithm")))
        except ValueError:
            return cast("AlgorithmType_co", cls.algo_class.from_checkpoint(path))

    def build_algo(self) -> AlgorithmType_co:
        try:
            return self.config.build_algo()  # type: ignore[return-type]
        except AttributeError as e:
            if "build_algo" not in str(e):
                raise
            # Older API
            return self.config.build()  # type: ignore[return-type]

    @overload
    def _get_module_spec(self, *, copy: Literal[False]) -> RLModuleSpec | None: ...

    @overload
    def _get_module_spec(
        self,
        *,
        copy: Literal[True],
        env: Optional[EnvType] = None,
        spaces: Optional[dict[str, tuple[gym.Space, gym.Space]]] = None,
        inference_only: Optional[bool] = None,
    ) -> RLModuleSpec: ...

    def _get_module_spec(
        self,
        *,
        copy: bool,
        env: Optional[EnvType] = None,
        spaces: Optional[dict[str, tuple[gym.Space, gym.Space]]] = None,
        inference_only: Optional[bool] = None,
    ) -> RLModuleSpec | None:
        if not self.config:
            raise ValueError("Config not defined yet, call create_config first.")
        if copy:
            return self.config.get_rl_module_spec(env, spaces, inference_only)
        if self.config._rl_module_spec is None:
            # Or OLD API
            logger.warning("ModuleSpec not defined yet, call config.rl_module first if you use the new API")
            return None
        assert not isinstance(self.config._rl_module_spec, MultiRLModuleSpec)
        return self.config._rl_module_spec

    def create_config_and_module_spec(
        self,
        *,
        env: Optional[EnvType] = None,
        spaces: Optional[dict[str, tuple[gym.Space, gym.Space]]] = None,
        inference_only: Optional[bool] = None,
    ) -> tuple[ConfigType_co, RLModuleSpec]:
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

    @property
    def trainable_class(self) -> type[TrainableBase[ParserType_co, ConfigType_co, AlgorithmType_co]]:
        """
        Returns the trainable class for the experiment.
        This is an alias of `self.trainable` but asserts that is as class.
        """
        assert isclass(self.trainable)
        return self.trainable

    @abstractmethod
    def _create_trainable(
        self,
    ) -> (
        Callable[[dict[str, Any]], TrainableReturnData]
        | type[TrainableBase[ParserType_co, ConfigType_co, AlgorithmType_co]]
    ):
        """
        Return a trainable for the Tuner to use.

        Attention:
            When using this use the public method create_trainable instead,
            which automatically assigns the trainable to self.trainable.

        Note:
            set trainable._progress_metrics to adjust the reporter output
        """

    @final
    def create_trainable(
        self,
    ) -> (
        type[TrainableBase[ParserType_co, ConfigType_co, AlgorithmType_co]]
        | Callable[[dict[str, Any]], TrainableReturnData]
    ):
        """
        Creates the trainable for the experiment.

        Attention:
            Do not overwrite this method. Overwrite _create_trainable instead.
        """
        self.trainable: (
            Callable[[dict[str, Any]], TrainableReturnData]
            | type[TrainableBase[ParserType_co, ConfigType_co, AlgorithmType_co]]
            | _MaybeNone
        ) = self._create_trainable()
        if isclass(self.trainable):
            logger.info(
                "create_trainable returns a class '%s'. To prevent errors the config will be frozen.",
                self.trainable.__name__,
            )
            self.config.freeze()
        if hasattr(self, "param_space") and self.param_space is not None:
            self.param_space["trainable_name"] = get_trainable_name(self.trainable)

        return self.trainable

    def unset_trainable(self, *, copy_config=False):
        """
        Unsets the trainable for the experiment, this unfreezes the config
        until the next create_trainable call.

        Using the setup as a context manager is often a better alternative to this function.
        """
        self.trainable = None
        if hasattr(self, "config"):
            if copy_config:
                self.config = cast("ConfigType_co", self.config.copy(copy_frozen=False))
            else:
                self.config._is_frozen = False
                if isinstance(self.config.evaluation_config, AlgorithmConfig):
                    self.config.evaluation_config._is_frozen = False

    # endregion

    # region Tuner

    def create_tuner(self: ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]) -> tune.Tuner:
        return TunerSetup(setup=self, eval_metric=EVAL_METRIC_RETURN_MEAN, eval_metric_order="max").create_tuner()

    # endregion

    # region upload experiments

    def comet_upload_offline_experiments(self):
        """Note this does not check for args.comet"""
        if self.comet_tracker is None:
            logger.info("No comet tracker / args.comet defined. Will not upload offline experiments.")
            return
        self.comet_tracker.upload_and_move()

    def wandb_upload_offline_experiments(self):
        logger.warning("Wandb offline upload is not yet implemented.")

    def upload_offline_experiments(self):
        if self.args.comet and "upload" in self.args.comet:
            logger.info("Uploading offline experiments to Comet")
            self.comet_upload_offline_experiments()
        if self.args.wandb and "upload" in self.args.wandb:
            self.wandb_upload_offline_experiments()

    # endregion

    # region callbacks

    @classmethod
    def _check_callbacks_requested(cls):
        """
        Check if the callbacks have been requested.

        Note:
            This is only a weak check - on the class - to be compatible with
            config_from_args.
        """
        if cls._retrieved_callbacks:
            return True
        logger.warning(
            "Callbacks for the Setup class %s have not been retrieved after creating the config. "
            "It is recommended to call `get_callbacks_from_args` inside the `config_from_args` method "
            "to support potential mixins that have their own callbacks. "
            "This may result in missing callbacks in the experiment.",
            cls,
            stacklevel=3,
        )
        return False

    @classmethod
    @abstractmethod
    def _get_callbacks_from_args(cls, args: NamespaceType[ParserType_co]) -> list[type[RLlibCallback]]:
        return []  # this can be can be called; return a list

    @final
    @classmethod
    def get_callbacks_from_args(cls, args: NamespaceType[ParserType_co]) -> list[type[RLlibCallback]]:
        """
        Returns a list of callbacks to be used with the experiment.

        Attention:
            Do not overwrite this method.
            Overwrite _get_callbacks_from_args is not sufficient.
        """
        cls._retrieved_callbacks = True  # Unsafe, set on class, clear on config_from_args
        return cls._get_callbacks_from_args(args)

    def _get_callbacks(self) -> list[type[RLlibCallback]]:
        """
        Returns a list of callbacks to be used with the experiment.

        Attention:
            Callbacks should be retrieved via get_callbacks,
            which sets the flag that the callbacks have been requested
            on the respective subclass.

            Overwrite this method if _get_callbacks_from_args is not sufficient.
        """
        return self._get_callbacks_from_args(self.args)

    @final
    def get_callbacks(self) -> list[Callable]:
        """
        Returns a list of callbacks to be used with the experiment.

        Attention:
            Do not overwrite this method.
            Overwrite _get_callbacks instead if _get_callbacks_from_args is not sufficient.
        """
        self._retrieved_callbacks = True
        return self._get_callbacks()

    # endregion

    # region save and restore

    def save(self) -> SetupCheckpointDict[ParserType_co, ConfigType_co, AlgorithmType_co]:
        """
        Saves the current setup state to a dictionary.
        Class can be restored from_saved. Does not save trainable state.
        """
        data: SetupCheckpointDict[ParserType_co, ConfigType_co, AlgorithmType_co] = {
            "args": cast("ParserType_co", SimpleNamespace(**self.args_to_dict())),
            "config": self.config,
            "__init_config__": False,
            # Allows to recreate the config based on args
            "param_space": self.param_space,
            "setup_class": type(self),
        }
        return data

    @classmethod
    def from_saved(
        cls,
        data: SetupCheckpointDict[ParserType_co, ConfigType_co, AlgorithmType_co],
        *,
        load_class: bool = False,
        init_trainable: bool = True,
    ) -> Self:
        # TODO: Why not a classmethod again?
        saved_class = data.get("setup_class", cls)
        setup_class = saved_class if load_class else cls
        if saved_class is not cls:
            logger.warning(
                "This class %s is not the same as the one used to save the data %s. "
                "Will use this class %s to restore the setup. "
                "This may lead to unexpected behavior. "
                "Use `load_class=True` to load the stored type "
                "or call this method on another class to avoid this warning.",
                cls,
                saved_class,
                setup_class,
                stacklevel=2,
            )
        setup_class = cast("type[Self]", setup_class)
        new = setup_class(init_config=False, init_param_space=False, init_trainable=False, parse_args=False)
        config: ConfigType_co | Literal[False] = data.get("config", False)
        new.param_space = data["param_space"]
        if data["__init_config__"] and config:
            logger.error("Passing __init_config__=True while also passing config ignores the passed config object")
        if config:
            new.config = config
        new.args = data["args"]
        new.setup(
            None,
            parse_args=False,
            init_param_space=False,
            init_trainable=init_trainable,
            init_config=data["__init_config__"] or not bool(config),
        )
        return new

    # endregion

    @classmethod
    @final
    def typed(cls) -> type[Self]:
        """
        Dummy method that returns the class itself, but with type parameters bound.

        This is useful for type checking and IDE support when using it with
        DefaultTrainable.define(Setup.typed) or similar methods that require a class as an argument.
        """
        return cls

    # region contextmanager

    def __enter__(self) -> Self:
        """
        When used as a context manager, the config can be modified at the end the
        param_space and trainable will be created

        Usage:
            .. code-block:: python

                # less overhead when setting these two to False, otherwise some overhead
                with Setup(init_param_space=False, init_trainable=False) as setup:
                    setup.config.env_runners(num_env_runners=0)
                    setup.config.training(minibatch_size=64)

                This is roughly equivalent to:
                setup = Setup(parse_args=True, init_config=True, init_param_space=False, init_trainable=False)
                setup.config.env_runners(num_env_runners=0)
                setup.config.training(minibatch_size=64)
                setup.setup(parse_args=False, init_config=False, init_param_space=True, init_trainable=True)
        """
        self.unset_trainable()
        self.param_space = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finishes the setup and creates the trainable"""
        self.setup(
            init_config=False,
            init_param_space=True,
            init_trainable=True,
            parse_args=False,
        )

    # Currently cannot use TypeForm[type[TypedDict]] as it is not included in the typing spec.
    # @property
    # @abstractmethod
    #
    # def _trainable_return_type(
    #    self,
    # ) -> TypeForm[type[TypedDict]] | Sequence[str] | dict[str, Any] | TrainableReturnData:
    #    """Keys or a TypedDict of the return type of the trainable function."""
