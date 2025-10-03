# pyright: enableExperimentalFeatures=true
"""Base classes and utilities for Ray RLlib experiment setup and configuration.

Provides the foundational :class:`ExperimentSetupBase` class for all experiment setups
in the Ray Utilities framework. Handles the complete lifecycle of reinforcement learning
experiments from argument parsing to training management.

Key Components:
    - :class:`ExperimentSetupBase`: Abstract base class for experiment configuration
    - :class:`SetupCheckpointDict`: Type definition for experiment checkpoints
    - Utility functions for environment creation and project management

The module integrates with Ray RLlib, Ray Tune, and Comet ML for scalable RL experiments
with logging, checkpointing, and hyperparameter optimization.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from inspect import isclass
from pathlib import Path
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generator,
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
from typing_extensions import NotRequired, Self, TypedDict, TypeVar, deprecated

from ray_utilities import RUN_ID
from ray_utilities.callbacks import LOG_IGNORE_ARGS, remove_ignored_args
from ray_utilities.callbacks.algorithm.seeded_env_callback import SeedEnvsCallback
from ray_utilities.comet import CometArchiveTracker
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.config.parser.default_argument_parser import ConfigFilePreParser, SupportsMetaAnnotations
from ray_utilities.environment import create_env
from ray_utilities.misc import AutoInt, get_trainable_name
from ray_utilities.setup._experiment_uploader import ExperimentUploader
from ray_utilities.setup.tuner_setup import TunerSetup
from ray_utilities.training.default_class import TrainableBase
from ray_utilities.warn import (
    warn_about_larger_minibatch_size,
    warn_if_batch_size_not_divisible,
    warn_if_minibatch_size_not_divisible,
)

if TYPE_CHECKING:
    import argparse

    import gymnasium as gym
    import ray.tune.search.sample  # noqa: TC004  # present at runtime from import ray.tune
    from ray.rllib.algorithms import PPO, Algorithm
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.utils.typing import EnvType
    from ray.runtime_context import RuntimeContext as RayRuntimeContext
    from ray.tune.experiment import Trial as TuneTrial
    from ray.tune.result_grid import ResultGrid

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
    """Type definition for experiment setup checkpoint data.

    This :class:`typing.TypedDict` defines the structure of checkpoint data saved
    and restored by :class:`ExperimentSetupBase` instances. It ensures type safety
    and consistency when serializing experiment configurations for restoration.

    The checkpoint contains all necessary information to recreate an experiment
    setup, including parsed arguments, algorithm configuration, parameter space
    for hyperparameter tuning, and metadata about the setup class.

    Type Parameters:
        ParserType_co: Type of the argument parser used in the setup
        ConfigType_co: Type of the RLlib algorithm configuration
        AlgorithmType_co: Type of the RLlib algorithm

    See Also:
        :meth:`ExperimentSetupBase.save_checkpoint`: Method that creates these checkpoints
        :meth:`ExperimentSetupBase.restore_checkpoint`: Method that restores from these checkpoints
    """

    args: ParserType_co
    """Parsed command-line arguments. Typically a :class:`~ray_utilities.config.DefaultArgumentParser` instance."""

    param_space: dict[str, Any] | TypedDict[{"__params_not_created__": Literal[True]}]
    """Parameter space for hyperparameter tuning.

    Result of :meth:`~ExperimentSetupBase.create_param_space`. If the method was never
    called, this contains a single key ``__params_not_created__`` set to ``True``.
    """

    setup_class: type[ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]]
    """Class type of the experiment setup for proper restoration."""

    config: ConfigType_co
    """RLlib algorithm configuration instance."""

    __init_config__: bool
    """Initialization flag for configuration.

    When ``True``, the config should be initialized from the args and the stored
    ``config`` field should be ignored and remain unset.
    """

    config_overrides: dict[str, Any]
    """Hold the current dict created by updating config_overrides"""

    config_files: NotRequired[Optional[Sequence[str | os.PathLike | Path]]]
    """
    Optional list of configuration files used during argument parsing.
    Should be present if setup / parser uses config files.
    """

    trial_name_creator: NotRequired[Optional[Callable[[TuneTrial], str]]]
    """Optional trial name creator function for Ray Tune trials."""


class ExperimentSetupBase(
    ABC, ExperimentUploader[ParserType_co], Generic[ParserType_co, ConfigType_co, AlgorithmType_co]
):
    """Abstract base class for Ray RLlib experiment setup and configuration.

    This class provides a comprehensive framework for setting up reinforcement learning
    experiments with Ray RLlib and Ray Tune. It handles argument parsing, algorithm
    configuration, environment setup, callback management, and trainable instantiation.

    The class is designed to be subclassed for specific experiment types, providing
    a consistent interface for experiment configuration while allowing customization
    of algorithm types, configurations, and training behaviors.

    Key Features:
        - Type-safe configuration with generic algorithm and config types
        - Integrated argument parsing with :class:`~ray_utilities.config.DefaultArgumentParser`
        - Environment creation and configuration management
        - Callback system integration for training customization
        - Project name management with tag substitution
        - Checkpoint and restoration capabilities
        - Ray Tune compatibility for hyperparameter optimization

    Type Parameters:
        ParserType_co: Type of the argument parser, typically :class:`~ray_utilities.config.DefaultArgumentParser`
        ConfigType_co: Type of the RLlib algorithm configuration (e.g., :class:`ray.rllib.algorithms.ppo.PPOConfig`)
        AlgorithmType_co: Type of the RLlib algorithm (e.g., :class:`ray.rllib.algorithms.ppo.PPO`)

    Attributes:
        PROJECT: Base name for the project used in :attr:`project_name`. Can include
            template tags like ``<env_type>`` that are substituted with argument values.
        config_class: Class type for the RLlib algorithm configuration.
        algo_class: Class type for the RLlib algorithm.
        use_dev_project: When ``True``, uses "dev-workspace" as project name in test mode.
        parse_known_only: When ``True``, ignores unrecognized arguments instead of failing.

    Example:
        >>> class PPOExperiment(ExperimentSetupBase[DefaultArgumentParser, PPOConfig, PPO]):
        ...     PROJECT = "CartPole-<agent_type>"
        ...     config_class = PPOConfig
        ...     algo_class = PPO
        ...
        ...     def create_config(self, args):
        ...         return self.config_class().environment("CartPole-v1")

    See Also:
        :class:`~ray_utilities.setup.algorithm_setup.AlgorithmSetup`: Concrete implementation
        :class:`~ray_utilities.setup.tuner_setup.TunerSetup`: For hyperparameter tuning
        :class:`~ray_utilities.training.default_class.DefaultTrainable`: Associated trainable class
    """

    default_extra_tags: ClassVar[list[str]] = [
        "dev",
        "<test>",
        "<gpu>",
        "<env_type>",
        "<agent_type>",
        "<num_envs:num_envs_per_env_runner=#>",
    ]
    """extra tags to add"""

    PROJECT: str = "Unnamed Project"
    """Base for project_name. Can consist of tags written as <args_attribute> that are substituted"""

    use_dev_project: bool = True
    """When True the `project_name` will be "dev-workspace" in test mode"""

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
        return "dev-workspace" if self.use_dev_project and self.args.test else self._parse_project_name(self.PROJECT)

    @project_name.setter
    def project_name(self, value: str):
        logger.warning("Setting project name to %s. Prefer creation of a new class", value)
        self.PROJECT: str = value

    def _parse_project_name(self, project_name: str):
        while "<" in project_name and ">" in project_name:
            start = project_name.index("<")
            end = project_name.index(">", start)
            tag = project_name[start : end + 1]
            substituted = self._substitute_tag(tag)
            if substituted is not None:
                project_name = project_name.replace(tag, str(substituted), 1)
        return project_name

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
        config_files: Optional[Sequence[str | os.PathLike | Path]] = None,
        load_args: Optional[str | os.PathLike | Path] = None,
        init_config: bool = True,
        init_param_space: bool = True,
        init_trainable: bool = True,
        parse_args: bool = True,
        trial_name_creator: Optional[Callable[[TuneTrial], str]] = None,
        change_log_level: Optional[bool] = True,
    ):
        """
        Initializes the experiment base class with optional argument parsing and setup.

        Args:
            args : Command-line arguments to parse. If None, defaults to sys.argv.
            config_files: Additional files with command line arguments that are
                loaded during argument parsing. Arguments in the file have a lower priority than
                those provided by the command line or :meth:`patch_args`.
                See: https://github.com/swansonk14/typed-argument-parser?tab=readme-ov-file#saving-and-loading-arguments
            load_args: A json file to load command line arguments from.
                See: https://github.com/swansonk14/typed-argument-parser?tab=readme-ov-file#saving-and-loading-arguments
            init_config : Whether to initialize the configuration. Defaults to True.
            init_param_space : Whether to initialize the parameter space. Defaults to True.
            init_trainable : Whether to initialize the trainable component. Defaults to True.
            parse_args : Whether to parse the provided arguments. Defaults to True.
            trial_name_creator: trial_name_creator function for :class:`ray.tune.Trial` objects
                when using :class:`ray.tune.Tuner`.
            change_log_level: If the parser supports it change the log level of the project with
                :func:`change_log_level`, if not none sets the `_change_log_level` of the parser.

        Note:
            When the setup creates a trainable that is a class, the config is frozen to
            prevent potentially unforwarded changes between setup.config the config of the
            trainable. Use `init_trainable=False` or `unset_trainable()`, edit the config
            and restore the trainable class with a call to `create_trainable()`.

        Attributes:
            parser (Parser[ParserType_co]): The argument parser instance.

        Calls:
            - self.create_parser(): Creates and assigns the argument parser.
            - self.setup(): Performs further setup based on initialization flags.
        """
        cfg_file_parser = ConfigFilePreParser()
        cfgs_from_cli = cfg_file_parser.parse_args(args, known_only=True)
        if config_files:
            logger.info("Adding config files %s to those found in args: %s", config_files, cfgs_from_cli.config_files)
            config_files = list(config_files)
            config_files.extend(cfgs_from_cli.config_files)
        else:
            config_files = cfgs_from_cli.config_files  # pyright: ignore[reportAssignmentType]
        pre_parsed_args = cfg_file_parser.extra_args
        self._config_overrides: Optional[dict[str, Any]] = None
        self._config_files = config_files
        self._load_args = load_args
        self._tune_trial_name_creator = trial_name_creator
        self._change_log_level = change_log_level
        self.parser: Parser[ParserType_co]
        self.parser = self.create_parser(config_files)
        if load_args:
            if not hasattr(self.parser, "load"):
                raise AttributeError(f"Parser {self.parser} has no attribute 'load' to support load_args")
            # possibly add skip_unsettable=True
            self.parser.load(load_args)  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(self.parser, Tap):
                logger.info("When using parse_args with a loaded config, argument parsing will be skipped.")
                parse_args = False  # cannot parse again
        self.setup(
            pre_parsed_args,
            init_config=init_config,
            init_param_space=init_param_space,
            init_trainable=init_trainable,
            parse_args=parse_args,
        )
        ExperimentUploader.__init__(self)  # args must be setup

    def setup(
        self,
        args: Optional[list[str]] = None,
        *,
        init_config: bool = True,
        init_param_space: bool = True,
        init_trainable: bool = True,
        parse_args: bool = True,
    ):
        """Initialize the experiment setup with configuration parsing and validation.

        This method orchestrates the complete setup process, from argument parsing
        through configuration creation to trainable initialization. It's typically
        called automatically during class instantiation.

        Args:
            args: Command line arguments to parse. If None, uses the arguments
                provided during instantiation.
            init_config: Whether to create and initialize the algorithm configuration.
            init_param_space: Whether to create the hyperparameter search space
                for Ray Tune optimization.
            init_trainable: Whether to create the trainable function/class for training.
            parse_args: Whether to parse command line arguments.

        Examples:
            Manual setup with custom arguments:

            >>> setup = PPOSetup()
            >>> setup.setup(["--env", "CartPole-v1", "--lr", "0.001"])

            Setup without trainable for configuration-only usage:

            >>> setup.setup(init_trainable=False)
            >>> setup.config.lr = 0.01  # Config remains mutable
            >>> setup.create_trainable()  # Create trainable when ready

        Note:
            The configuration is automatically frozen after trainable creation
            to prevent accidental modifications during training.
        """
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
            self._unfreeze_config()
        if init_param_space:
            # relies on trainable to get its name
            self.param_space: dict[str, Any] | _MaybeNone = self.create_param_space()

    # region Argument Parsing

    def create_parser(self, config_files: Optional[Sequence[str | os.PathLike]] = None) -> Parser[ParserType_co]:
        self.parser = DefaultArgumentParser(allow_abbrev=False, config_files=config_files)
        if self._change_log_level is not None:
            self.parser._change_log_level = self._change_log_level
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
        warn_if_batch_size_not_divisible(
            batch_size=args.train_batch_size_per_learner, num_envs_per_env_runner=args.num_envs_per_env_runner
        )
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

    def _merge_args_from_checkpoint(
        self, parsed: NamespaceType[ParserType_co], checkpoint: str
    ) -> NamespaceType[ParserType_co]:
        # Merge args from a checkpoint:
        path = Path(checkpoint)
        with open(path / "state.pkl", "rb") as f:
            state: dict[str, Any] = pickle.load(f)
        # Create a patched parser with the old values as default values
        new_parser = self.create_parser()
        if hasattr(new_parser, "_change_log_level"):
            new_parser._change_log_level = False  # pyright: ignore[reportAttributeAccessIssue]
        self.parser = new_parser
        restored_args: dict[str, Any] = vars(state["setup"]["args"])
        for action in new_parser._actions:
            if isinstance(parsed, DefaultArgumentParser):
                action.default = parsed.restore_arg(
                    action.dest, restored_value=restored_args.get(action.dest, action.default)
                )
            else:
                action.default = restored_args.get(action.dest, action.default)  # set new default values
            # These are changed in process_args. Problem we do not know if we should
            # restore them their value or "auto". e.g. --iterations 10 -> need to change iterations
            if action.dest == "iterations":
                logger.debug(
                    "Resetting the parsers iterations default value to 'auto' after checkpoint restore. "
                    "It will be recreated from the total_steps argument."
                )
                action.default = "auto"
        self.parser = new_parser
        return self.parser.parse_args()

    @staticmethod
    def _remove_testing_args_from_argv(args: list[str] | None = None):
        """
        When run under test some shorthand commands might infer with the argument parser.
        Clean those away.

        Returns:
            sys.argv with --udiscovery ... -- removed if present.
        """
        args = sys.argv if args is None else args
        if "--udiscovery" in args:
            start = args.index("--udiscovery")
            if "--" in args[start:]:
                # slice args away until --
                end = start + args[start:].index("--")
            else:
                end = len(args)
            argv = args[:start] + args[end + 1 :]
            logger.info("Removing testing argument %s from command line arguments.", args[start : end + 1])
            return argv
        return sys.argv[1:] if args is sys.argv else args

    def parse_args(
        self, args: list[str] | None = None, *, known_only: bool | None = None, checkpoint: Optional[str] = None
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
            parsed = self.parser.parse_args(self._remove_testing_args_from_argv(args), known_only=known_only)
            extra_args = self.parser.extra_args
        except TypeError as e:
            if "'known_only' is an invalid invalid keyword" not in str(e):
                raise
            if known_only:
                # NOTE: In this case parsed is not self.parser!
                parsed, extra_args = self.parser.parse_known_args(args)
            else:
                parsed = self.parser.parse_args(args)
                extra_args = None
        if extra_args:
            logger.warning(
                "The following arguments were not recognized by the parser: %s.",
                extra_args,
            )
        checkpoint = checkpoint or parsed.from_checkpoint
        if checkpoint:
            parsed = self._merge_args_from_checkpoint(parsed, checkpoint)

        self.args = self.postprocess_args(parsed)
        return self.args

    # endregion

    # region Tags

    def _substitute_tag(self, tag: str):
        """Substitutes tags written with <args_attribute> with the respective attribute value."""
        if not tag.startswith("<"):
            return tag
        assert tag[-1] == ">", f"Invalid tag parsing format: {tag}. Must be '<args_attribute>'"
        tag = tag[1:-1]
        append_value = False
        if tag.endswith("=#"):
            tag = tag[:-2]
            append_value = True
        if ":" in tag:
            nickname, tag = tag.split(":")
        else:
            nickname = tag
        if hasattr(self.args, tag):
            value = getattr(self.args, tag)
            if isinstance(value, bool) or value is None:
                if not append_value:
                    if value:  # is True
                        return nickname
                    return None
            if append_value:
                if isinstance(value, float):
                    value = round(value, 2)
                return nickname + "=" + str(value)
            return value
        return None  # error

    def _parse_extra_tags(self, extra_tags: Sequence[str] | None = None) -> list[str]:
        """
        Parses tags that are stored in default_extra_tags.
        Tags that are written like <args_attribute> are substituted with the respective attribute.

        Tags ending with <args_attribute=#> are written as "args_attribute=attribue_value"
        A tag written as <nick:args_attribute> will lookup args_attribute and when found will be
        displayed as nick. However, this only applies for true boolean values and "=#" numerical values.
        """
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
        extra_tags.append(f"run_id:{RUN_ID}")
        return list(filter(None, extra_tags))

    def create_tags(self, extra_tags: Sequence[str] | None = None) -> list[str]:
        if not hasattr(self.args, "tags"):
            logger.info("Parsed arguments have not attribute tags.")
            return self._parse_extra_tags(extra_tags)
        return DefaultArgumentParser.organize_subtags(
            [
                *self.args.tags,
                *self._parse_extra_tags(extra_tags),
            ]
        )

    # endregion
    # region hparams

    def clean_args_to_hparams(self, args: Optional[NamespaceType[ParserType_co]] = None) -> dict[str, Any]:
        args = args or self.get_args()
        to_remove: list[str] = [*LOG_IGNORE_ARGS, "process_number"]
        if isinstance(args, SupportsMetaAnnotations):
            to_remove.extend(args.get_non_cli_args())
        upload_args = remove_ignored_args(args, remove=to_remove)
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
        """Sample hyperparameters from the configured parameter space.

        Generates a single set of parameter values by sampling from any tune domains
        in the parameter space. This is useful for test runs or single experiments
        without full hyperparameter optimization.

        Returns:
            Dict of sampled parameter values with tune domains resolved to concrete values.

        Examples:
            Sample parameters for a single training run:

            >>> setup = PPOSetup()
            >>> setup.add_tune_config({"lr": tune.uniform(0.001, 0.01)})
            >>> params = setup.sample_params()
            >>> print(params["lr"])  # Random value between 0.001 and 0.01

        See Also:
            :meth:`create_param_space`: Creates the parameter space
            :func:`ray.tune.search.sample.Domain.sample`: Domain sampling method
        """
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
            if module_spec.module_class is not None:
                module = module_spec.module_class.__name__
            else:
                module = f"RLModule({self.args.agent_type})"
        else:
            module = None
        # Arguments reported on the CLI
        param_space: dict[str, Any] = {
            "env": (
                self.config.env if isinstance(self.config.env, str) else self.config.env.unwrapped.spec.id  # pyright: ignore[reportOptionalMemberAccess]
            ),  # pyright: ignore[reportOptionalMemberAccess]
            "algo": self.config.algo_class.__name__ if self.config.algo_class is not None else "UNDEFINED",
            "module": module,
            "setup_cls": self.__class__.__name__,
            "trainable_name": self.get_trainable_name(),  # "UNDEFINED" is called before create_trainable
        }
        # If not logged in choice will not be reported in the CLI interface
        # Comment out to not display in CLI
        # param_space = {k: tune.choice([v]) for k, v in param_space.items()}
        if self.args.env_seeding_strategy == "same":
            # Fixed or same random selected seed
            # NOTE: This might not be used by create_algorithm_config.
            # Rather is, make_seeded_env_callback(args["seed"])
            param_space["env_seed"] = self.args.seed  # if self.args.seed is not None else random.randint(0, 2**16)
        elif self.args.env_seeding_strategy == "constant":
            param_space["env_seed"] = SeedEnvsCallback.env_seed  # use as constant value
        elif self.args.env_seeding_strategy == "random":
            param_space["env_seed"] = None
        else:  # "sequential", sample from distribution
            param_space["env_seed"] = tune.randint(0, 2**16)
            # param_space["run_seed"] = tune.randint(0, 2**16)  # potential seed for config

        # Other args not shown in the CLI
        # Log CLI args as hyperparameters
        param_space["cli_args"] = self.clean_args_to_hparams(self.args)
        param_space["run_id"] = RUN_ID
        param_space["experiment_id"] = RUN_ID
        param_space["experiment_name"] = self.project_name
        param_space["experiment_group"] = self.group_name
        self.param_space = param_space
        del self._dynamic_parameters_to_tune
        if self._config_files:
            # Store config files for trial to sync when using a scheduler or remote node.
            param_space["_config_files"] = self._config_files
        return param_space

    # endregion

    # region config and trainable

    def config_overrides(self, *, update=False, **kwargs) -> dict[str, Any]:
        """
        Override the algorithm configuration with the given keyword arguments.
        Or just get the current overrides when no kwargs are given.

        Args:
            update: If True, updates the existing config overrides instead of replacing them.
            kwargs: The keyword arguments to override in the algorithm configuration.

        Returns:
            The current configuration overrides.

        Raises:
            ValueError: If the config is already frozen. That is, after `create_trainable`
                has already be called.
        """
        if not kwargs:
            return self._config_overrides or {}
        if hasattr(self, "config") and self.config._is_frozen:
            raise ValueError(
                "Cannot override algorithm configuration as the config is already frozen. "
                "Use this function in a `with setup` block or call `unset_trainable()` first."
            )
        overrides = self.config_class.overrides(**kwargs)
        if not self._config_overrides or not update:
            self._config_overrides = overrides
        else:
            self._config_overrides.update(overrides)
        return self._config_overrides

    def _unfreeze_config(self_or_config: Self | AlgorithmConfig):  # pyright: ignore[reportSelfClsParameterName] # noqa: N805
        """
        Unfreeze the configuration to allow further modifications.
        This is useful when you want to change the configuration after it has been frozen.
        """
        if isinstance(self_or_config, AlgorithmConfig):
            config = self_or_config
        else:
            if not hasattr(self_or_config, "config") or not self_or_config.config:
                return
            config = self_or_config.config
        config._is_frozen = False
        if isinstance(config.evaluation_config, AlgorithmConfig):
            config.evaluation_config._is_frozen = False

    def _create_config(self, base: Optional[ConfigType_co] = None) -> ConfigType_co:
        # Overwrite if config_from_args is not sufficient.
        return self.config_from_args(self.args, base=base)

    def _learner_config_dict_defaults(self):
        """Sets values in the learner_config_dict that are used in this packages if not already set."""
        assert self.config, "Config not defined yet, call create_config first."
        self.config.learner_config_dict.setdefault("_debug_connectors", False)
        self.config.learner_config_dict.setdefault("remove_masked_samples", False)
        self.config.learner_config_dict.setdefault("accumulate_gradients_every", 1)

    @final
    def create_config(self, base: Optional[ConfigType_co] = None) -> ConfigType_co:
        """
        Creates the config for the experiment.

        Attention:
            Do not overwrite this method. Overwrite _create_config / config_from_args instead.

        Args:
            base: Optional config to update based on args, instead of creating a new one.
        """
        self.config = self._create_config(base=base)
        overrides = self.config_overrides()
        if hasattr(self.config, "_restored_overrides"):
            old_overwrites: dict[str, Any] = self.config._restored_overrides  # pyright: ignore[reportAttributeAccessIssue]
            different_keys = set(old_overwrites) - set(overrides)
            if different_keys:
                msg = (
                    f"The config has been restored from a checkpoint with config overrides {old_overwrites}, "
                    f"but the current config does not set the same keys ({overrides}). "
                    "Restoring the config is ambiguous - priority of (new) config_from_args vs. (old) config_overrides."
                    " Old overwrites will not be enforced and might be overwritten. Verify that the config is correct!"
                )
                warnings.warn(msg, UserWarning, stacklevel=1)
                logger.warning(msg)
            # always restore values will be added to the overrides

            # to make overrides have a higher priority than args:
            if isinstance(self.parser, SupportsMetaAnnotations):
                overrides = {
                    k: v for k, v in old_overwrites.items() if k in self.parser.get_to_restore_values()
                } | overrides
                self.config_overrides(**overrides)
        # classmethod, but _retrieved_callbacks might be set on instance
        if overrides:
            self.config.update_from_dict(overrides)
        self._learner_config_dict_defaults()
        self.config.freeze()
        self._check_callbacks_requested.__func__(self)  # pyright: ignore[reportFunctionMemberAccess]
        self._retrieved_callbacks = False  # Reset for next call
        type(self)._retrieved_callbacks = False
        return self.config

    @classmethod
    @abstractmethod
    def _config_from_args(
        cls, args: ParserType_co | argparse.Namespace, base: Optional[ConfigType_co] = None
    ) -> ConfigType_co:
        """
        Create an algorithm configuration; similar to `create_config` but as a `classmethod`.

        Tip:
            This method is useful if you do not have access to the setup instance, e.g.
            inside the trainable function.

        Args:
            args: The parsed arguments from the command line.
            base: Optional a config to update based on the args.

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

    @classmethod
    def _config_from_checkpoint(cls, path: str) -> tuple[ConfigType_co, bool, dict[str, Any]]:
        """
        Load the config from a checkpoint.

        Args:
            path: The path to the checkpoint file.
        """
        # likely need cloudpickle
        with open(Path(path) / "state.pkl", "rb") as f:
            state: dict[str, Any] = pickle.load(f)
        setup_state: SetupCheckpointDict = state["setup"]
        config = setup_state["config"]
        init_config = setup_state["__init_config__"]
        overrides = setup_state["config_overrides"]
        cls._unfreeze_config(config)
        return (
            config,  # pyright: ignore[reportReturnType], cannot guarantee AlgorithmType
            init_config,
            overrides,
        )

    @final
    @classmethod
    def config_from_args(
        cls, args: NamespaceType[ParserType_co], base: Optional[ConfigType_co] = None
    ) -> ConfigType_co:
        """
        Create an algorithm configuration; similar to `create_config` but as a `classmethod`.

        Tip:
            This method is useful if you do not have access to the setup instance, e.g.
            inside the trainable function.

        Args:
            args: The parsed arguments from the command line.
            base: Optional a config to update based on the args.

        Usage:
            .. code-block:: python

                algo: AlgorithmConfig = Setup.config_from_args(args)
        """
        overrides = None
        if args.from_checkpoint:
            # init_config tells setup if init_config should be called; here we are passed that
            loaded_config, init_config, overrides = cls._config_from_checkpoint(args.from_checkpoint)
            if not init_config:
                logger.debug("Not updating config from checkpoint, init_config is False.")
            else:
                if loaded_config and base:
                    raise ValueError("Cannot load a config from checkpoint and update a base at the same time.")
                if not loaded_config:
                    logger.info("from_checkpoint is set, but no config found in the checkpoint.")
                base = base or loaded_config
            if base:
                base._restored_overrides = overrides

        config = cls._config_from_args(args, base=base)

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
    def algorithm_from_checkpoint(
        cls, path: str, *, config: Optional[ConfigType_co] = None, **kwargs
    ) -> AlgorithmType_co:
        """
        Load an algorithm from a checkpoint.

        Args:
            path: The path to the checkpoint directory.
            ignore_config: Whether to ignore the Setup's config and restore the config from the
                checkpoint or overwrite it with the config obtained by `create_config`.
                Defaults to False.
        """
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
        if config is not None:
            warn_if_batch_size_not_divisible(
                batch_size=config.train_batch_size_per_learner, num_envs_per_env_runner=config.num_envs_per_env_runner
            )
            if config.minibatch_size is not None and config.minibatch_size > config.train_batch_size_per_learner:
                warn_about_larger_minibatch_size(
                    minibatch_size=config.minibatch_size,
                    train_batch_size_per_learner=config.train_batch_size_per_learner,
                    note_adjustment=True,
                )
                config.minibatch_size = config.train_batch_size_per_learner
            warn_if_minibatch_size_not_divisible(
                minibatch_size=config.minibatch_size,
                num_envs_per_env_runner=config.num_envs_per_env_runner,
            )
            kwargs = {"config": config, **kwargs}
        try:
            # Algorithm checkpoint is likely in subdir.
            return cast("AlgorithmType_co", cls.algo_class.from_checkpoint(os.path.join(path, "algorithm"), **kwargs))
        except ValueError:
            return cast("AlgorithmType_co", cls.algo_class.from_checkpoint(path, **kwargs))

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

    @deprecated("Use get_rl_module_spec on the config instead")
    def create_config_and_module_spec(
        self,
        base: Optional[ConfigType_co] = None,
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

        Args:
            base: Optional config to update instead of creating a new one.
            env: Optional environment to use for the module spec.
            spaces: Optional spaces to use for the module spec.
            inference_only: If True, the module spec will be created for inference only.
        """
        config = self.create_config(base=base)
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
        logger.debug(
            "create_trainable called. To prevent errors the config will be frozen.",
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
                self._unfreeze_config()

    # endregion

    # region Tuner

    def _tuner_add_iteration_stopper(self) -> bool | None:
        """
        Determine whether the Tuner should add a maximum iteration stopper.

        For example, when tuning the batch_size or if other factors are implemented that
        cause the max iterations to be dynamic for different trainables, the Tuner should
        not add an Iteration stopper.

        Return:
            False: Tell the tuner to not add an iteration stopper.
            True: If the Tuner should add it, if appropriate
            None: Decide in Tuner.

        See Also:
            :class:`MaximumResultIterationStopper`

        The default variant of this version checks if
        `iterations` or `train_batch_size_per_learner` is in :attr:`args` otherwise returns None
        """
        if not self.args.tune:
            return None
        tune_keys = set(self.args.tune or [])
        if isinstance(self.args.iterations, int) and not isinstance(self.args.iterations, AutoInt):
            logger.info("args.iterations is a non AutoInt integer, telling the Tuner to add a stopper.")
            return True  # If the iteration is set manually do add a stopper
        return False if len({"iterations", "batch_size", "train_batch_size_per_learner"} & tune_keys) > 0 else None

    def create_tuner(self: ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]) -> tune.Tuner:
        """Create a Ray Tune tuner for hyperparameter optimization.

        Creates a tuner with sensible defaults for reinforcement learning experiments,
        including episode return maximization and standard stopping criteria.

        Returns:
            Configured Ray Tune tuner ready for hyperparameter optimization.

        Examples:
            Basic tuning setup:

            >>> setup = PPOSetup()
            >>> setup.add_tune_config({"lr": tune.grid_search([0.001, 0.01])})
            >>> tuner = setup.create_tuner()
            >>> results = tuner.fit()

        See Also:
            :class:`TunerSetup`: Advanced tuner configuration options
            :class:`ray.tune.Tuner`: Underlying Ray Tune tuner class
        """
        return TunerSetup(
            setup=self,
            eval_metric=self.args.metric,
            eval_metric_order=self.args.mode,
            add_iteration_stopper=self._tuner_add_iteration_stopper(),
            trial_name_creator=self._tune_trial_name_creator,
        ).create_tuner()

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

    def get_state(self) -> SetupCheckpointDict[ParserType_co, ConfigType_co, AlgorithmType_co]:
        """
        Saves the current setup state to a dictionary.
        Class can be restored from_saved. Does not save trainable state.
        """
        data: SetupCheckpointDict[ParserType_co, ConfigType_co, AlgorithmType_co] = {
            "args": cast("ParserType_co", SimpleNamespace(**self.args_to_dict())),
            "config": self.config,
            "config_overrides": self.config_overrides(),
            "config_files": self._config_files,
            "__init_config__": True,
            # Allows to recreate the config based on args
            "param_space": getattr(self, "param_space", {"__params_not_created__": True}),
            "setup_class": type(self),
            "trial_name_creator": self._tune_trial_name_creator,
        }
        return data

    @classmethod
    def from_saved(
        cls,
        data: SetupCheckpointDict[ParserType_co, ConfigType_co, AlgorithmType_co],
        *,
        load_class: bool = False,
        init_trainable: bool = True,
        init_config: Optional[bool] = None,
        load_config_files: bool = True,
    ) -> Self:
        """
        Args:
            init_config: If True, the config will be re-initialized from args after loading the state.
                If False, the config from the state will be used. If None (default), the config will be
                re-initialized if the stored __init_config__ is True in the state or if no config was saved.
            load_config_files: By default updates to config files are respected and loaded,
                this might change the loaded config. Set to False to not load the config files
                and just use the args/config from the state.
        """
        # TODO: Why not a classmethod again?
        unchecked_keys = set(data.keys())
        saved_class = data.get("setup_class", cls)
        unchecked_keys.discard("setup_class")
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

        # Handle config file restoration
        config_files_to_use = None

        if load_config_files and (original_config_files := data.get("config_files")):
            not_existing_files = [file for file in map(Path, original_config_files) if not file.exists()]
            if not_existing_files:
                synced_files = []
                try:
                    # sync missing config files without the need of the Syncing callback.
                    ctx: RayRuntimeContext = ray.get_runtime_context()

                    # WARNING: The following accesses Ray's internal runtime context and worker state,
                    # which may change between Ray versions. This code is known to work with Ray >=2.49.0.
                    # If you upgrade Ray and encounter errors here, check for changes in Ray's internal API.
                    try:
                        actor: TrainableBase | Any = ctx.worker.actors[ctx.worker.actor_id]
                        storage = getattr(actor, "_storage", None)
                        if TYPE_CHECKING:
                            storage = actor._storage
                        if storage:
                            for file in map(Path, not_existing_files):
                                if file.is_absolute():
                                    dest = file.relative_to(Path(os.environ["TUNE_ORIG_WORKING_DIR"])).as_posix()
                                    source = file.as_posix()
                                else:
                                    dest = file.as_posix()
                                    source = (Path(os.environ["TUNE_ORIG_WORKING_DIR"]) / file).as_posix()
                                storage.syncer.sync_down(remote_dir=source, local_dir=dest)
                                # FIXME: Is one wait enough when there are multiple files?
                                try:
                                    storage.syncer.wait()
                                except FileNotFoundError as e:
                                    logger.error("Could not sync missing config files: %s", e)
                                else:
                                    synced_files.append(dest)
                        else:
                            logger.error(
                                "Config files %s do not exist and cannot be restored because no storage is configured.",
                                not_existing_files,
                            )
                            # TODO: possibly create empty files so that parser does not fail
                    except (AttributeError, KeyError) as e:
                        logger.error(
                            "Failed to access Ray's internal worker state to restore config files. "
                            "This may be due to a Ray version incompatibility. Error: %s",
                            e,
                        )
                    except Exception:
                        logger.exception(
                            "Unexpected error while restoring missing config files %s.", not_existing_files
                        )
                        # TODO: possibly create empty files so that parser does not fail
                except Exception:
                    logger.exception("Could not restore missing config files %s.", not_existing_files)
                config_files_to_use = [
                    file for file in map(Path, original_config_files) if file not in not_existing_files
                ]
                config_files_to_use.extend(synced_files)
            else:
                config_files_to_use = original_config_files

        new = setup_class(
            init_config=False,
            init_param_space=False,
            init_trainable=False,
            parse_args=False,
            trial_name_creator=data.get("trial_name_creator"),
            config_files=config_files_to_use,
        )
        unchecked_keys.discard("trial_name_creator")
        unchecked_keys.discard("config_files")

        new.config_overrides(**data.get("config_overrides", {}))
        config: ConfigType_co | Literal[False] = data.get("config", False)
        unchecked_keys.discard("config_overrides")
        unchecked_keys.discard("config")

        new.param_space = data["param_space"]
        if init_config is None and data["__init_config__"] and config:
            logger.warning(
                "Having __init_config__=True in the state while also restoring a config ignores "
                "the restored config object. You can control the behavior and disable this warning "
                "by setting init_config=True/False in the :meth:`from_saved` method. "
                "Or, by removing/changing the keys of the state dict before calling this function."
            )
        unchecked_keys.remove("param_space")

        init_config = data["__init_config__"] or not bool(config) if init_config is None else init_config
        if init_config:
            logger.info("Re-initializing config from args after state was loaded", stacklevel=2)
        else:
            logger.info(
                "Not re-initializing config from args after state was loaded. "
                "Using the config from the state. "
                "This may lead to unexpected behavior if the args do not match the config.",
                stacklevel=2,
            )
        unchecked_keys.remove("__init_config__")

        if config:
            new.config = config
        new.args = data["args"]
        unchecked_keys.discard("args")
        new.setup(
            None,
            parse_args=False,
            init_param_space=False,
            init_trainable=init_trainable,
            init_config=init_config,
        )
        if new.args.comet and not new.comet_tracker:
            new.comet_tracker = CometArchiveTracker()
            logger.info("CometArchiveTracker setup")
        if unchecked_keys:  # possibly a subclass has more keys
            logger.info(
                "Some keys in the state were not used: %s.",
                unchecked_keys,
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

    @contextmanager
    def open_config(self) -> Generator[ConfigType_co, Any, None]:
        """
        Contextmanager that unfreezes the setups config for editing.

        Updates the config_overrides with the changes made within.
        """
        _was_frozen = self.config._is_frozen
        config_before = self.config.to_dict()
        self._unfreeze_config()
        yield self.config
        config_after = self.config.to_dict()
        diff = {k: v for k, v in config_after.items() if config_before.get(k) != v}
        if diff:
            self._config_overrides = self.config_overrides(update=True, **diff)
        if _was_frozen:
            self.config.freeze()

    def __enter__(self) -> Self:
        """
        When used as a context manager, the config can be modified at the end the
        param_space and trainable will be created

        Usage:
            .. code-block:: python

                # less overhead when setting these two to False, otherwise some overhead
                with Setup(init_param_space=False, init_trainable=False) as setup:
                    setup.config_overrides(
                        num_env_runners=0,
                        minibatch_size=64,
                    )

                This is roughly equivalent to:
                setup = Setup(parse_args=True, init_config=True, init_param_space=False, init_trainable=False)
                setup.config.env_runners(num_env_runners=0)
                setup.config.training(minibatch_size=64)
                setup.setup(parse_args=False, init_config=False, init_param_space=True, init_trainable=True)
        """
        self.unset_trainable()
        self.param_space = None
        self.__open_config = self.open_config()
        self.__open_config.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finishes the setup and creates the trainable"""
        self.__open_config.__exit__(exc_type, exc_value, traceback)
        del self.__open_config
        if self._config_overrides:
            self._unfreeze_config()
            self.config.update_from_dict(self._config_overrides)
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
