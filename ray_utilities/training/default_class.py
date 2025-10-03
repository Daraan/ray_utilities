"""Default trainable classes for Ray RLlib experiments with checkpointing and progress tracking.

This module provides base classes and utilities for creating trainable algorithms
that integrate with Ray Tune. It includes support for checkpointing, progress bars,
git metadata tracking, and experiment setup management.

The main classes provide a framework for running reinforcement learning experiments
with standardized logging, restoration, and state management capabilities.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import pathlib
import pickle
import sys
from abc import ABCMeta
from copy import copy
from inspect import isclass
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, ClassVar, Collection, Generic, Optional, TypedDict, TypeVar, cast, overload

import git
import pyarrow.fs
import ray
from ray import get_runtime_context, tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.callbacks.utils import make_callback
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_EVAL_ENV_RUNNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_METRICS_LOGGER,
)
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.tune.result import SHOULD_CHECKPOINT
from typing_extensions import Self, TypeAliasType

from ray_utilities.callbacks.progress_bar import restore_pbar, save_pbar_state, update_pbar
from ray_utilities.callbacks.tuner.metric_checkpointer import TUNE_RESULT_IS_A_COPY
from ray_utilities.config.parser.default_argument_parser import LOG_STATS, LogStatsChoices
from ray_utilities.constants import FORK_FROM, PERTURBED_HPARAMS
from ray_utilities.misc import AutoInt, is_pbar
from ray_utilities.nice_logger import set_project_log_level
from ray_utilities.training.functional import training_step
from ray_utilities.training.helpers import (
    create_running_reward_updater,
    episode_iterator,
    get_current_step,
    get_total_steps,
    patch_model_config,
    setup_trainable,
    sync_env_runner_states_after_reload,
)
from ray_utilities.typing.trainable_return import RewardUpdaters
from ray_utilities.warn import (
    warn_about_larger_minibatch_size,
    warn_if_batch_size_not_divisible,
    warn_if_minibatch_size_not_divisible,
)

if TYPE_CHECKING:
    import git.types  # noqa: TC004  # false positive
    from ray.experimental import tqdm_ray
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.utils.typing import StateDict
    from ray.runtime_context import RuntimeContext
    from tqdm import tqdm
    from typing_extensions import NotRequired

    from ray_utilities.callbacks.progress_bar import RangeState, RayTqdmState, TqdmState
    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase, SetupCheckpointDict
    from ray_utilities.typing.metrics import AutoExtendedLogMetricsDict, LogMetricsDict


_logger = logging.getLogger(__name__)

_ParserTypeInner = TypeVar("_ParserTypeInner", bound="DefaultArgumentParser")
_ConfigTypeInner = TypeVar("_ConfigTypeInner", bound="AlgorithmConfig")
_AlgorithmTypeInner = TypeVar("_AlgorithmTypeInner", bound="Algorithm")

_ParserType = TypeVar("_ParserType", bound="DefaultArgumentParser")
_ConfigType = TypeVar("_ConfigType", bound="AlgorithmConfig")
_AlgorithmType = TypeVar("_AlgorithmType", bound="Algorithm")

_ExperimentSetup = TypeAliasType(
    "_ExperimentSetup",
    "type[ExperimentSetupBase[_ParserType, _ConfigType, _AlgorithmType]] | ExperimentSetupBase[_ParserType, _ConfigType, _AlgorithmType]",  # noqa: E501
    type_params=(_ParserType, _ConfigType, _AlgorithmType),
)

_UNKNOWN_GIT_SHA = "unknown"


def _validate_algorithm_config_afterward(func):
    """Decorator to validate the algorithm config after a method call.

    This decorator ensures that the algorithm configuration remains valid
    after method execution, which is important for reloaded algorithms
    that might have inconsistent state that could fail validation tests.

    Args:
        func: The method to decorate. Should be a method of a class that
            has an ``algorithm_config`` attribute.

    Returns:
        The decorated function that validates the algorithm config after execution.

    Example:
        >>> class MyTrainable:
        ...     @_validate_algorithm_config_afterward
        ...     def setup_method(self):
        ...         # Method implementation
        ...         pass
    """

    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.algorithm_config.validate()
        return result

    return wrapper


class TrainableStateDict(TypedDict):
    """State dictionary structure returned by :meth:`TrainableBase.get_state`.

    This TypedDict defines the complete state structure that can be saved
    and restored for a trainable algorithm, including the algorithm itself,
    its configuration, progress tracking information, and metadata.
    """

    trainable: StateDict
    """The state obtained by :meth:`ray.tune.Trainable.get_state`."""

    algorithm: NotRequired[StateDict]  # component; can be ignored
    """Algorithm state (optional since algorithm may handle its own checkpointing)."""
    algorithm_config: StateDict
    """Serialized algorithm configuration."""
    algorithm_overrides: Optional[dict[str, Any]]
    """Configuration overrides applied to the algorithm."""
    iteration: int
    """Current training iteration number."""
    pbar_state: RayTqdmState | TqdmState | RangeState
    """Progress bar state for restoration."""

    reward_updaters: dict[str, list[float]]
    """Historical reward values for running average calculations."""

    setup: SetupCheckpointDict[Any, Any, Any]
    """Experiment setup checkpoint data."""

    current_step: int
    """Current environment step count."""

    git_sha: NotRequired[str]
    """SHA hash of the current git commit for reproducibility."""


class PartialTrainableStateDict(TypedDict, total=False):
    """Partial state dictionary with all fields optional.

    This TypedDict variant of :class:`TrainableStateDict` makes all fields
    optional, which is useful for incremental state updates or when only
    a subset of the state needs to be specified.

    All attributes have the same meaning as in :class:`TrainableStateDict`,
    but are optional and may not be present in the dictionary.
    """

    trainable: StateDict
    """The state obtained by :meth:`ray.tune.Trainable.get_state`."""

    algorithm: StateDict
    """Algorithm component state."""
    algorithm_config: StateDict
    """Serialized algorithm configuration."""
    iteration: int
    """Current training iteration number."""
    pbar_state: RayTqdmState | TqdmState | RangeState
    """Progress bar state for restoration."""

    reward_updaters: dict[str, list[float]]
    """Historical reward values for running average calculations."""

    setup: SetupCheckpointDict[Any, Any, Any]
    """Experiment setup checkpoint data."""


class TrainableBase(Checkpointable, tune.Trainable, Generic[_ParserType, _ConfigType, _AlgorithmType]):
    """Base class for Ray Tune trainable algorithms with comprehensive state management.

    This class extends both :class:`ray.rllib.utils.checkpoints.Checkpointable` and
    :class:`ray.tune.Trainable` to provide a complete framework for running
    reinforcement learning experiments with Ray Tune. It includes support for:

    - Automatic checkpointing and restoration with progress tracking
    - Git metadata tracking for experiment reproducibility
    - Progress bar integration with tqdm/tqdm_ray
    - Experiment setup management with typed argument parsing
    - Reward tracking and running averages

    Type Parameters::

        ``_ParserType``: Type of the argument parser (extends Tap)
        ``_ConfigType``: Type of the algorithm configuration
        ``_AlgorithmType``: Type of the RL algorithm

    Checkpoint and Restoration Flow::

        1. **Saving**: :meth:`save_to_path` → :meth:`get_state` → :meth:`get_metadata`
        2. **Loading**: :meth:`from_checkpoint` → :meth:`restore_from_path` → :meth:`set_state`

    Key Methods::

        - **Checkpointable Interface:**
            - :meth:`get_state`: Returns complete state dictionary
            - :meth:`set_state`: Restores state from dictionary
            - :meth:`get_metadata`: Returns experiment metadata with git info
            - :meth:`get_ctor_args_and_kwargs`: Returns constructor arguments

        - **Trainable Interface:**
            - :meth:`setup`: Initialize the algorithm and experiment
            - :meth:`step`: Perform one training step (abstract)
            - :meth:`save_checkpoint`: Save state to checkpoint
            - :meth:`load_checkpoint`: Load state from checkpoint

    Example:
        >>> class MyTrainable(TrainableBase[MyParser, AlgorithmConfig, PPO]):
        ...     def step(self):
        ...         return self.algorithm.train()
        >>>
        >>> DefinedTrainable = MyTrainable.define(MySetup)
        >>> trainable = DefinedTrainable()
    """

    setup_class: _ExperimentSetup[_ParserType, _ConfigType, _AlgorithmType]
    """Experiment setup class that defines the configuration and algorithm.

    This class attribute must be set via the :meth:`define` class method
    before the trainable can be instantiated. It provides the blueprint
    for creating and configuring the RL algorithm.
    """
    discrete_eval: bool = False
    """Whether to use discrete evaluation episodes instead of continuous evaluation."""
    use_pbar: bool = True
    """Whether to display progress bars during training using tqdm."""

    _git_repo_sha: str = _UNKNOWN_GIT_SHA
    """SHA hash of the current git commit, set during class definition for reproducibility."""

    _log_level: str | int | None = None

    cls_model_config: ClassVar[Optional[dict[str, Any] | DefaultModelConfig]] = None

    @classmethod
    def define(
        cls,
        setup_cls: _ExperimentSetup[_ParserTypeInner, _ConfigTypeInner, _AlgorithmTypeInner],
        *,
        discrete_eval: bool = False,
        use_pbar: bool = True,
        fix_argv: bool = False,
        model_config: Optional[dict[str, Any] | DefaultModelConfig] = None,
        log_level: Optional[int | str] = None,
    ) -> type[DefaultTrainable[_ParserTypeInner, _ConfigTypeInner, _AlgorithmTypeInner]]:
        """Create a trainable subclass with the specified experiment setup.

        This class method creates a new trainable class that is bound to a specific
        experiment setup. The resulting class can be instantiated and used with
        Ray Tune for training. It also captures git metadata for reproducibility.

        Args:
            setup_cls: The experiment setup class that defines how to create
                and configure the RL algorithm. Must be a subclass of the
                experiment setup base class.
            model_config: Optional dictionary of model configuration overrides to apply
                for this trainable class.
            discrete_eval: Whether to use discrete evaluation episodes instead
                of continuous evaluation. Defaults to False.
            use_pbar: Whether to display progress bars during training.
                Defaults to True.
            fix_argv: Whether to fix the current :data:`sys.argv` to the setup class.
                When True, the current command-line arguments are saved and used
                for initialization in remote contexts where the original argv
                is not available. Defaults to False.


        Returns:
            A new trainable class with the setup class bound and git metadata
            captured. The returned class can be instantiated for training.

        Raises:
            UserWarning: If the git repository has uncommitted changes when
                capturing the commit SHA.

        Example:
            >>> from my_experiments import MySetup
            >>> MyTrainable = TrainableBase.define(MySetup, discrete_eval=True, use_pbar=True)
            >>> # Now MyTrainable can be used with Ray Tune
            >>> tune.run(MyTrainable, config={...})

        Note:
            The ``fix_argv`` parameter is particularly useful in distributed
            training scenarios where the trainable is instantiated on remote
            workers that don't have access to the original command-line arguments.
        """
        # Avoid undefined variable error in class body
        discrete_eval_ = discrete_eval
        use_pbar_ = use_pbar
        # Fix current cli args to the trainable - necessary for remote
        if fix_argv:
            if isclass(setup_cls):
                setup_cls = type(setup_cls.__name__ + "FixedArgv", (setup_cls,), {"_fixed_argv": sys.argv})
            else:
                _logger.error("fix_argv is currently only supported when passing a setup class.")
        # Get git metadata now, as when on remote we are in a temp dir
        try:
            repo = git.Repo(search_parent_directories=True)
        except (git.InvalidGitRepositoryError, Exception) as e:
            _logger.warning("Could not get git commit SHA: %s", e)
            cls._git_repo_sha = _UNKNOWN_GIT_SHA
        else:
            cls._git_repo_sha: str = cast("git.types.AnyGitObject", repo.head.object).hexsha
            if repo.is_dirty(untracked_files=False):
                _logger.warning("Saving commit sha as metadata, but repository has uncommitted changes.")
            _logger.info("Current commit sha is %s", cls._git_repo_sha)

        class DefinedTrainable(
            cls,
            metaclass=_TrainableSubclassMeta,
            base=cls,
        ):
            # FIXME: Somehow these attributes are possibly not carried over into tune properly
            _git_repo_sha = cls._git_repo_sha
            setup_class = setup_cls
            discrete_eval = discrete_eval_
            use_pbar = use_pbar_
            cls_model_config = model_config
            _log_level = log_level

        DefinedTrainable.__name__ = "Defined" + cls.__name__

        assert not TYPE_CHECKING or issubclass(DefinedTrainable, TrainableBase)
        assert DefinedTrainable._base_cls is cls

        return DefinedTrainable  # type: ignore[return-value]

    # region Trainable setup
    @override(tune.Trainable)
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        algorithm_overrides: Optional[dict[str, Any]] = None,
        model_config: Optional[dict[str, Any] | DefaultModelConfig] = None,
        **kwargs,
    ):
        """
        Args:
            config: Configuration dictionary for the trainable.

                Special behavior:

                - Keys that match attributes on the setup class' parsed
                  arguments namespace (e.g., :class:`DefaultArgumentParser`) override those
                  attributes.
                - The special key "model_config" extends the model_config of
                  the created Algorithm.

            algorithm_overrides: Overrides applied to the
                algorithm configuration before building it.

            model_config: Model
                configuration to extend the algorithm's model configuration.

            **kwargs: Forwarded to superclass constructors (e.g., storage for
                tune.Trainable).
        """
        # Change log level:
        # When remote set log level or project here
        if ray.is_initialized():
            run_context: RuntimeContext = get_runtime_context()
            if run_context.get_actor_name() is not None:  # we are remote
                log_level = (
                    config.get("log_level", config.get("cli_args", {}).get("log_level", self._log_level))
                    if config
                    else self._log_level
                )
                if log_level is not None:
                    set_project_log_level(logging.getLogger(__name__.split(".")[0]), log_level)

        self._algorithm = None
        self._algorithm_overrides = algorithm_overrides
        self._model_config = model_config if model_config is not None else self.cls_model_config
        if self._algorithm_overrides and self.setup_class._fixed_argv:
            _logger.warning(
                "Using a Trainable with fixed argv on the setup_class and algorithm_overrides, "
                "might result in unexpected values after a restore. Test carefully."
            )
            # NOTE: Use get_ctor_args_and_kwargs to include the overwrites on a reload
        if config and PERTURBED_HPARAMS in config:
            _logger.info("Received perturbed config: %s", config[PERTURBED_HPARAMS])
            self._perturbed_config: Optional[dict[str, Any]] = config[PERTURBED_HPARAMS]
        else:
            self._perturbed_config = None
        """When initialized with a perturbed config, this holds the original config."""

        self._current_step: int = 0
        """The current env steps sampled by the trainable, updated by step()."""

        super().__init__(config or {}, **kwargs)  # calls setup
        # TODO: do not create loggers, if any are created
        self.config: dict[str, Any]
        """Not the AlgorithmConfig, config passed by the tuner"""

        self._setup: ExperimentSetupBase[_ParserType, _ConfigType, _AlgorithmType]
        """
        The setup that was used to initially create this trainable.

        Attention:
            When restoring from a checkpoint, this reflects the *inital* setup, not the current one.
            Config and args hold by this object might differ from the current setup.
        """

        self._algo_config: _ConfigType
        """Config used during setup, available even when algorithm is None during checkpoint loading."""

    @property
    def algorithm(self) -> _AlgorithmType:
        # self._algorithm should only be used inside setup and load_checkpoint
        # to take care of a potential None algorithm
        assert self._algorithm
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: _AlgorithmType) -> None:
        self._algorithm = value

    @override(tune.Trainable)
    def setup(self, config: dict[str, Any], *, algorithm_overrides: Optional[dict[str, Any]] = None) -> None:
        """Initialize the trainable with Ray Tune configuration and setup experiment components.

        This method is called automatically by Ray Tune during trainable instantiation.
        It initializes the RL algorithm, progress tracking, experiment setup, and reward
        tracking components based on the provided configuration.

        Args:
            config: Ray Tune configuration dictionary containing hyperparameters and
                experiment settings. This includes both standard RLlib configuration
                and Ray Utilities specific parameters.
            algorithm_overrides: Optional algorithm configuration overrides to apply
                during setup. These take precedence over setup class defaults.

        Sets:
            - algorithm: The Ray RLlib algorithm instance for training
            - _pbar: Progress bar for training iteration tracking
            - _iteration: Current training iteration counter
            - _setup: Experiment setup instance with parsed configuration
            - _reward_updaters: List of reward tracking utilities

        Raises:
            ValueError: If setup_class is not defined. Use :meth:`define` to create
                a properly configured trainable class.

        Note:
            This method should not be called directly. Instead, Ray Tune calls it
            automatically during trainable initialization with the configured
            parameter space.

        Examples:
            Proper usage through Ray Tune:

            >>> TrainableCls = DefaultTrainable.define(PPOSetup)
            >>> tuner = tune.Tuner(TrainableCls, param_space={"lr": 0.001})
            >>> # setup() is called automatically during tuner.fit()
        """
        # NOTE: Called during __init__
        # Setup algo, config, args, etc.
        if not hasattr(self, "setup_class"):
            raise ValueError(
                f"setup_class is not set on {self}. "
                f"Use TrainableCls = {self.__class__.__name__}.define(setup_class) to set it.",
            )
        if algorithm_overrides is not None and self._algorithm_overrides is not None:
            if algorithm_overrides != self._algorithm_overrides:
                _logger.warning(
                    "Both `algorithm_overrides` and `self._algorithm_overrides` (during __init__) are set. "
                    "Consider passing only one. Overwriting self._algorithm_overrides with new value."
                )
            else:
                _logger.error(
                    "Both `algorithm_overrides` and `self._algorithm_overrides` (during __init__) "
                    "and do not match. Overwriting self._algorithm_overrides with new value from setup(). \n %s != %s "
                )
            self._algorithm_overrides = algorithm_overrides
        assert self.setup_class.parse_known_only is True
        _logger.debug("Setting up %s with config: %s", self.__class__.__name__, config)

        if isclass(self.setup_class):
            self._setup = self.setup_class(
                init_trainable=False, init_param_space=False
            )  # XXX # FIXME # correct args; might not work when used with Tuner
        else:
            self._setup = self.setup_class
        # TODO: Possible unset setup._config to not confuse configs (or remove setup totally?)
        # use args = config["cli_args"] # XXX

        # _logger.debug("Sys argv during Trainable.setup(): %s", sys.argv)
        _logger.info(
            "args %s are:\n %s",
            "(in config)" if "cli_args" in config else "(on setup)",
            pformat(
                config.get(
                    "cli_args",
                    {
                        k: v
                        for k, v in (
                            self._setup.args.as_dict()
                            if hasattr(self._setup.args, "as_dict")
                            else vars(self._setup.args)
                        ).items()
                        if not callable(v)
                    },
                )
            ),
        )
        # NOTE: args is a dict, self._setup.args a Namespace | Tap
        self._reward_updaters: RewardUpdaters
        # if FORK_FROM we assume we load the checkpoint later by an outside call.
        load_from_checkpoint = "cli_args" in config and config["cli_args"].get("from_checkpoint")
        load_algorithm = load_from_checkpoint or FORK_FROM in config
        self._algorithm: _AlgorithmType | None
        args, _algo_config, algorithm, self._reward_updaters = setup_trainable(
            hparams=config,
            setup=self._setup,
            setup_class=self.setup_class if isclass(self.setup_class) else None,
            config_overrides=self._algorithm_overrides,
            model_config=self._model_config,
            create_algo=not load_algorithm,  # Avoid creating unnecessary intermediate algorithm
        )
        # Store the config for access when algorithm might be None during checkpoint loading
        self._algo_config = _algo_config
        self._param_overrides: dict[str, Any] = args.get("__overwritten_keys__", {})
        """Changed parameters via the hparams argument, e.g. passed by the tuner. See also: --tune"""
        self._pbar: tqdm | range | tqdm_ray.tqdm = episode_iterator(args, config, use_pbar=self.use_pbar)
        self._iteration: int = 0
        self.log_stats: LogStatsChoices = args[LOG_STATS]
        # After components have been setup up load checkpoint if requested
        # When restore is called by a Tuner, setup was called a while ago
        # keep args ONLY to calculate steps and iterations
        self._args_during_setup = args
        if load_algorithm:
            self._algo_class: type[_AlgorithmType] | None = _algo_config.algo_class
            # store args to be latter used in restore call
            if not load_from_checkpoint:
                # reload controlled by the outside, restore called, e.g. because of FORK_FROM
                return
            _logger.info("At end of setup(), loading from checkpoint: %s", config["cli_args"]["from_checkpoint"])
            # calls restore from path; from_checkpoint could also be a dict here
            # algorithm is None when create_algo=False, will be set in load_checkpoint
            self.load_checkpoint(config["cli_args"]["from_checkpoint"])
            assert self._algorithm is not None
        else:
            assert algorithm is not None
            self._algorithm = algorithm
        assert self.algorithm.config
        self._calculate_steps_and_iterations(args)  # also called in load_checkpoint

    def _calculate_steps_and_iterations(self, args: dict[str, Any]):
        """After the setup / load_checkpoint recalculate the total_steps & iterations until the goal."""
        # Use the config from setup_trainable to calculate total steps
        # which handles both algorithm and non-algorithm cases
        assert self.algorithm
        assert self.algorithm.metrics
        assert self.algorithm.config
        current_step = get_current_step(self.algorithm.metrics)
        steps_to_new_goal = args["total_steps"] - current_step
        iterations_to_new_goal = (
            steps_to_new_goal // self.algorithm.config.train_batch_size_per_learner + 1
            if steps_to_new_goal % self.algorithm.config.train_batch_size_per_learner != 0
            else steps_to_new_goal // self.algorithm.config.train_batch_size_per_learner
        )
        if isinstance(args["iterations"], AutoInt):
            # if dynamic buffer to not recalculate total_steps
            args["iterations"] = "auto" if args["dynamic_buffer"] else iterations_to_new_goal
            # Get before changing iterations again:
            steps_with_current_batch_size = get_total_steps(args, self.algorithm.config)
            args["iterations"] = iterations_to_new_goal
        else:
            steps_with_current_batch_size = get_total_steps(args, self.algorithm.config)

        # iterations was passed by user; do not overwrite
        # adjust total_steps instead
        if steps_with_current_batch_size is not None and not args["use_exact_total_steps"]:
            total_steps = steps_with_current_batch_size + current_step
        else:
            total_steps = args.get("total_steps", None)
        self._total_steps = {
            "total_steps": total_steps,
            "iterations": "auto",
        }

    @property
    def algorithm_config(self) -> _ConfigType:
        """
        Config of the algorithm used.

        Note:
            This is a copy of the setup's config which might has been further modified.
        """
        if self._algorithm is not None:
            return self._algorithm.config  # pyright: ignore[reportReturnType]
        # During checkpoint loading, algorithm might be None, use stored config
        return self._algo_config  # pyright: ignore[reportReturnType]

    @algorithm_config.setter
    def algorithm_config(self, value: _ConfigType):
        # Does not update env_runners and learner !
        if self._algorithm is not None:
            self._algorithm.config = value
        else:
            # During checkpoint loading, algorithm might be None, update stored config
            self._algo_config = value

    @property
    def trainable_config(self) -> dict[str, Any]:
        return self.config

    @trainable_config.setter
    def trainable_config(self, value: dict[str, Any]):
        self.config = value

    @override(tune.Trainable)
    def reset_config(self, new_config):  # pyright: ignore[reportIncompatibleMethodOverride] # currently not correctly typed in ray
        # Return True if the config was reset, False otherwise
        # This will be called when tune.TuneConfig(reuse_actors=True) is used
        # TODO
        super().reset_config(new_config)
        self.setup(new_config)
        return True

    @override(tune.Trainable)
    def cleanup(self):
        # call stop to fully free resources
        super().cleanup()
        if self._algorithm is not None:
            self._algorithm.cleanup()
        if is_pbar(self._pbar):
            self._pbar.close()

    # endregion Trainable setup

    # region checkpointing

    def _rebuild_algorithm_if_necessary(self, new_algo_config: AlgorithmConfig) -> bool | None:
        """Check if env runners need to be recreated based on state or algorithm changes.

        Args:
            new_algo_config: The new algorithm config being restored.

        Todo:
            Does not check if learners need to be recreated.
            Assumes num_learners does not change.
        """
        if self._algorithm is None:
            return None
        call_setup_again = False
        env_runners_need_update = (
            self.algorithm.env_runner_group
            and new_algo_config.num_env_runners != self.algorithm.env_runner_group.num_remote_env_runners()
        )
        eval_config = (
            new_algo_config.get_evaluation_config_object() if not new_algo_config.in_evaluation else new_algo_config
        )
        eval_env_runners_need_update = (
            eval_config
            and self.algorithm._should_create_evaluation_env_runners(eval_config)  # if deactivated dont care
            and self.algorithm.eval_env_runner_group
            # if new_algo_config is evaluation config cannot get another evaluation config
            and eval_config.num_env_runners != self.algorithm.eval_env_runner_group.num_remote_env_runners()
        )
        if env_runners_need_update:
            if self.algorithm.env_runner_group:
                # end old
                self.algorithm.env_runner_group.stop()
                delattr(self.algorithm, "env_runner_group")

            self.algorithm.env_runner_group = EnvRunnerGroup(
                env_creator=self.algorithm.env_creator,
                validate_env=self.algorithm.validate_env,
                default_policy_class=self.algorithm.get_default_policy_class(new_algo_config),
                config=new_algo_config,
                # New API stack: User decides whether to create local env runner.
                # Old API stack: Always create local EnvRunner.
                local_env_runner=(
                    True
                    if not new_algo_config.enable_env_runner_and_connector_v2
                    else bool(new_algo_config.create_local_env_runner)
                ),
                logdir=self.algorithm.logdir,
                tune_trial_id=self.algorithm.trial_id,
            )
            # FIXME: These are not equal if num_env_per_env_runner changes
            # assert self.algorithm.spaces == self.algorithm.eval_env_runner_group.get_spaces()
        if eval_env_runners_need_update:
            assert eval_config is not None
            _, env_creator = self.algorithm._get_env_id_and_creator(eval_config.env, eval_config)

            # Create a separate evaluation worker set for evaluation.
            # If evaluation_num_env_runners=0, use the evaluation set's local
            # worker for evaluation, otherwise, use its remote workers
            # (parallelized evaluation).
            self.algorithm.eval_env_runner_group = EnvRunnerGroup(
                env_creator=env_creator,
                validate_env=None,
                default_policy_class=self.algorithm.get_default_policy_class(eval_config),
                config=eval_config,
                logdir=self.algorithm.logdir,
                tune_trial_id=self.algorithm.trial_id,
                # New API stack: User decides whether to create local env runner.
                # Old API stack: Always create local EnvRunner.
                local_env_runner=(
                    True
                    if not eval_config.enable_env_runner_and_connector_v2
                    else bool(eval_config.create_local_env_runner)
                ),
                # offset for the placement group
                pg_offset=new_algo_config.num_env_runners or 0,
            )
            # FIXME
            # assert self.algorithm.spaces == self.algorithm.eval_env_runner_group.get_spaces()
        return False

    # region Trainable checkpoints

    @override(tune.Trainable)
    def save_checkpoint(self, checkpoint_dir: str) -> dict[str, Any]:
        # A returned dict will be serialized
        # can return dict_or_path
        # NOTE: Do not rely on absolute paths in the implementation of
        state = self.get_state()
        # save in subdir
        assert isinstance(state, dict)
        if self._storage:
            # Assume we are used with a Tuner and StorageContext handles checkpoints.
            # NOTE: This is a fixed path as relative paths are not well supported by restore
            # which just passes a temp dir here.

            # Checkpoint index is updated after this function returns
            self._storage.current_checkpoint_index += 1
            algorithm_checkpoint_dir = (Path(self._storage.checkpoint_fs_path) / "algorithm").as_posix()
            self._storage.current_checkpoint_index -= 1
        else:  # Assume checkpoint_dir is a temporary path
            algorithm_checkpoint_dir = (Path(checkpoint_dir) / "algorithm").as_posix()

        self.save_to_path(
            (Path(checkpoint_dir)).absolute().as_posix(), state=cast("dict[str, Any]", state)
        )  # saves components
        save = {
            "state": state,  # contains most information
            "algorithm_checkpoint_dir": algorithm_checkpoint_dir,
        }
        return save

    @override(tune.Trainable)
    def load_checkpoint(self, checkpoint: Optional[dict | str], *, ignore_setup: bool = False, **kwargs) -> None:
        # NOTE: from_checkpoint is a classmethod, this isn't
        # set pbar
        # set weights
        # set iterations
        # set reward_updaters
        # config comes from new setup
        setup_config = self._setup.config.copy(copy_frozen=False)
        if self._model_config is not None:
            patch_model_config(setup_config, self._model_config)
        if PERTURBED_HPARAMS in self.config:
            perturbed: dict[str, Any] = {k: self.config[k] for k in self.config[PERTURBED_HPARAMS]}
            # assert perturbed == self.config[PERTURBED_HPARAMS]
            setup_config = setup_config.update_from_dict(perturbed)
            # Remove __perturbed__ from config so that a future checkpoint hparams does not see them as highest priority
            self._perturbed_config: Optional[dict[str, Any]] = self.config.pop(PERTURBED_HPARAMS)
        else:
            perturbed = {}
            self._perturbed_config = None
        algo_kwargs: dict[str, Any] = (
            {**kwargs}
            if ignore_setup  # NOTE: also ignores overrides on self
            else {"config": setup_config, **kwargs}
        )
        if perturbed and ignore_setup:
            _logger.warning("Using a perturbed keys and ignore_setup likely ignores the perturbed arguments.")

        if "config" in algo_kwargs and (self._algorithm_overrides or self._param_overrides or perturbed):
            config_overrides = (self._algorithm_overrides or {}) | self._param_overrides | perturbed
            algo_kwargs["config"] = (
                cast("AlgorithmConfig", algo_kwargs["config"])
                .copy(copy_frozen=False)
                .update_from_dict(config_overrides)
            )
            # Fix batch_size < minibatch_size:
            if (
                algo_kwargs["config"].minibatch_size is not None
                and algo_kwargs["config"].train_batch_size_per_learner < algo_kwargs["config"].minibatch_size
            ):
                warn_about_larger_minibatch_size(
                    minibatch_size=algo_kwargs["config"].minibatch_size,
                    train_batch_size_per_learner=algo_kwargs["config"].train_batch_size_per_learner,
                    note_adjustment=True,
                )
                config_overrides["minibatch_size"] = algo_kwargs["config"].train_batch_size_per_learner
                algo_kwargs["config"].minibatch_size = algo_kwargs["config"].train_batch_size_per_learner
            algo_kwargs["config"].freeze()
        else:
            config_overrides = perturbed or {}
        setup_config.freeze()
        overrides_at_start = self._algorithm_overrides or {}
        if isinstance(checkpoint, dict):
            keys_to_process = set(checkpoint.keys())  # Sanity check if processed all keys

            # from_checkpoint calls restore_from_path which calls set_state
            # if checkpoint["algorithm_checkpoint_dir"] is a tempdir (e.g. from tune, this is wrong)
            # However, self.set_state should have take care of algorithm already even if checkpoint dir is missing
            if os.path.exists(checkpoint["algorithm_checkpoint_dir"]):
                if self._algorithm is not None:
                    self._algorithm.stop()  # free resources first
                    algo_class = type(self._algorithm)
                    delattr(self, "_algorithm")
                else:
                    algo_class = self._algo_class
                    assert algo_class is not None
                # Does not call on_checkpoint_loaded callback
                self._algorithm = algo_class.from_checkpoint(
                    Path(checkpoint["algorithm_checkpoint_dir"]).absolute().as_posix(), **algo_kwargs
                )  # pyright: ignore[reportAttributeAccessIssue]
            elif "algorithm_state" in checkpoint:
                _logger.error(
                    "Algorithm checkpoint directory %s does not exist, will restore from state",
                    checkpoint["algorithm_checkpoint_dir"],
                )
                if self._algorithm is None:
                    _logger.critical(
                        "Cannot restore algorithm from state because algorithm was not created. "
                        "This should not happen when loading from checkpoint with create_algo=False."
                    )
                    raise RuntimeError("Algorithm is None but trying to restore from algorithm_state")
                if self._algorithm_overrides:
                    checkpoint["algorithm_state"]["config"] = checkpoint["algorithm_state"]["config"].update_from_dict(
                        self._algorithm_overrides
                    )
                self.algorithm.set_state(checkpoint["algorithm_state"])
                keys_to_process.remove("algorithm_state")
            else:
                _logger.critical(
                    "Algorithm checkpoint directory %s does not exist, (possibly temporary path was saved) "
                    "and no state provided. Cannot restore algorithm.",
                    checkpoint["algorithm_checkpoint_dir"],
                )
                raise FileNotFoundError(None, "algorithm_checkpoint_dir", checkpoint["algorithm_checkpoint_dir"])
            keys_to_process.remove("algorithm_checkpoint_dir")
            # can add algorithm_state to check correctness
            if "algorithm" not in checkpoint["state"]:
                _logger.debug("No algorithm state found in checkpoint.")
            self.set_state(checkpoint["state"])
            keys_to_process.remove("state")
            # TODO: sync env runner states?
            # FIXME: reloaded might not have same amount of env_runners if created with a different amount

            assert len(keys_to_process) == 0, f"Not all keys were processed during load_checkpoint: {keys_to_process}"
        elif checkpoint is not None:
            components = {c[0] for c in self.get_checkpointable_components()}
            components.discard("algorithm")
            # Restore from path does not account for new algorithm_config; so this merely sets the state
            # use from_checkpoint to do that
            self.restore_from_path(checkpoint, **algo_kwargs)
            # Restored overrides:
            if "config" in algo_kwargs and (
                (self._algorithm_overrides and overrides_at_start != self._algorithm_overrides) or perturbed
            ):
                # Overrides at start should have higher priority
                algo_kwargs["config"] = (
                    algo_kwargs["config"]
                    .copy(copy_frozen=False)
                    # Restored < algorithm_overrides < hparams < perturbed
                    .update_from_dict((self._algorithm_overrides or {}) | config_overrides)
                )
                # Fix minibatch size < batch_size if reloaded bad value
                if (
                    algo_kwargs["config"].minibatch_size is not None
                    and algo_kwargs["config"].train_batch_size_per_learner < algo_kwargs["config"].minibatch_size
                ):
                    warn_about_larger_minibatch_size(
                        minibatch_size=algo_kwargs["config"].minibatch_size,
                        train_batch_size_per_learner=algo_kwargs["config"].train_batch_size_per_learner,
                        note_adjustment=True,
                    )
                    config_overrides["minibatch_size"] = algo_kwargs["config"].train_batch_size_per_learner
                    algo_kwargs["config"].minibatch_size = algo_kwargs["config"].train_batch_size_per_learner
                algo_kwargs["config"].freeze()
            # return
            # for component in components:
            #    self.restore_from_path(checkpoint, component=component, **algo_kwargs)
            # free resources first
            if self._algorithm is not None:
                self._algorithm.stop()  # free resources first
            assert self._algorithm or self._algo_class
            self._algorithm = cast(
                "_AlgorithmType",
                (self._algorithm or self._algo_class).from_checkpoint(  # pyright: ignore[reportOptionalMemberAccess]
                    (Path(checkpoint) / "algorithm").as_posix(),
                    **algo_kwargs,
                ),
            )
            sync_env_runner_states_after_reload(self.algorithm)
        else:
            raise ValueError(f"Checkpoint must be a dict or a path. Not {type(checkpoint)}")
        if perturbed:  # XXX: check that perturbed has highest priority and updated the config
            for k, v in perturbed.items():
                assert getattr(self.algorithm_config, k) == v, "Expected perturbed key '%s' to be %s, but got %s" % (
                    k,
                    v,
                    getattr(self.algorithm_config, k),
                )
        warn_if_batch_size_not_divisible(
            batch_size=self.algorithm_config.train_batch_size_per_learner,
            num_envs_per_env_runner=self.algorithm_config.num_envs_per_env_runner,
        )
        warn_if_minibatch_size_not_divisible(
            minibatch_size=self.algorithm_config.minibatch_size,
            num_envs_per_env_runner=self.algorithm_config.num_envs_per_env_runner,
        )
        # callbacks are not called by the above methods.
        make_callback(
            "on_checkpoint_loaded",
            # ray has a wrong type signature here, accepting only list
            callbacks_objects=self.algorithm.callbacks,  # pyright: ignore[reportArgumentType]
            callbacks_functions=self.algorithm.config.callbacks_on_checkpoint_loaded,  # pyright: ignore[reportArgumentType,reportOptionalMemberAccess]
            kwargs={"algorithm": self.algorithm, "metrics_logger": self.algorithm.metrics},
        )
        self._calculate_steps_and_iterations(self._args_during_setup)

    # endregion

    # region Checkpointable methods

    @overload
    def get_state(
        self,
        components: None = None,
        *,
        not_components: None = None,
        **kwargs,
    ) -> TrainableStateDict: ...

    @overload
    def get_state(
        self,
        components: Optional[str | Collection[str]] = None,
        *,
        not_components: Optional[str | Collection[str]] = None,
        **kwargs,
    ) -> PartialTrainableStateDict | TrainableStateDict: ...

    @override(Checkpointable)
    @override(tune.Trainable)
    def get_state(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        components: Optional[str | Collection[str]] = None,
        *,
        not_components: Optional[str | Collection[str]] = None,
        **kwargs,  # noqa: ARG002
    ) -> TrainableStateDict | PartialTrainableStateDict:
        """Returns the implementing class's current state as a dict.

        The returned dict must only contain msgpack-serializable data if you want to
        use the `AlgorithmConfig._msgpack_checkpoints` option. Consider returning your
        non msgpack-serializable data from the `Checkpointable.get_ctor_args_and_kwargs`
        method, instead.

        Args:
            components: An optional collection of string keys to be included in the
                returned state. This might be useful, if getting certain components
                of the state is expensive (e.g. reading/compiling the weights of a large
                NN) and at the same time, these components are not required by the
                caller.
            not_components: An optional list of string keys to be excluded in the
                returned state, even if the same string is part of `components`.
                This is useful to get the complete state of the class, except
                one or a few components.
            kwargs: Forward-compatibility kwargs.

        Returns:
            The current state of the implementing class (or only the `components`
            specified, w/o those in `not_components`).
        """
        if components is not None and not isinstance(components, str):
            components = copy(components)
        if not_components is not None and not isinstance(not_components, str):
            not_components = copy(not_components)
        trainable_state = super(Checkpointable, self).get_state()  # cheap call
        algorithm_config_state = (
            self.algorithm_config.get_state()
            if self._check_component("algorithm_config", components, not_components)
            else None
        )
        setup_state = self._setup.get_state() if self._check_component("setup", components, not_components) else None
        reward_updaters_state = (
            {k: v.keywords["reward_array"] for k, v in self._reward_updaters.items()}
            if self._check_component("reward_updaters", components, not_components)
            else None
        )
        pbar_state = (
            save_pbar_state(self._pbar, self._iteration)
            if self._check_component("pbar", components, not_components)
            else None
        )
        if self._check_component("algorithm", components, not_components):
            algo_state = self.algorithm.get_state(
                self._get_subcomponents("algorithm", components),
                not_components=force_list(self._get_subcomponents("algorithm", not_components)),
            )
            algo_state["algorithm_class"] = type(self.algorithm)  # NOTE: Not msgpack-serializable
            algo_state["config"] = algorithm_config_state
        else:
            algo_state = {}
        # for integrity of the TrainableStateDict, remove None case:
        if TYPE_CHECKING:
            assert setup_state
            assert algorithm_config_state
            assert reward_updaters_state
            assert pbar_state
        state: TrainableStateDict = {
            "trainable": trainable_state,
            "algorithm": algo_state,  # might be removed by save_to_path
            "algorithm_config": algorithm_config_state,
            "algorithm_overrides": (
                self._algorithm_overrides.to_dict()
                if isinstance(self._algorithm_overrides, AlgorithmConfig)
                else self._algorithm_overrides
            ),
            "iteration": self._iteration,
            "pbar_state": pbar_state,
            "reward_updaters": reward_updaters_state,
            "setup": setup_state,
            "current_step": self._current_step,
            "git_sha": self._git_repo_sha,
        }
        # Current step is
        # state["trainable"]["last_result"]["current_step"]
        # Filter out components not in the components list
        if components is not None or not_components is not None:
            return cast(
                "PartialTrainableStateDict",
                {k: v for k, v in state.items() if self._check_component(k, components, not_components)},
            )
        return state

    # @_validate_algorithm_config_afterward
    @override(Checkpointable)
    def set_state(self, state: StateDict | TrainableStateDict) -> None:
        """Sets the implementing class' state to the given state dict.

        If component keys are missing in `state`, these components of the implementing
        class will not be updated/set.

        Args:
            state: The state dict to restore the state from. Maps component keys
                to the corresponding subcomponent's own state.
        """
        # NOTE: When coming from restore_from_path, the components have already be restored
        # are those states possibly more correct?
        keys_to_process = set(state.keys())
        assert state["trainable"]["iteration"] == state["iteration"]
        try:
            super(Checkpointable, self).set_state(state.get("trainable", {}))  # pyright: ignore
        except AttributeError:
            # Currently no set_state method
            trainable_state = state["trainable"]
            self._iteration = trainable_state["iteration"]
            self._timesteps_total = trainable_state["timesteps_total"]
            self._time_total = trainable_state["time_total"]
            self._episodes_total = trainable_state["episodes_total"]
            self._last_result = trainable_state["last_result"]
            if ray.__version__ != trainable_state.get("ray_version", ""):
                _logger.info(
                    "Checkpoint was created with a different Ray version: %s != %s",
                    trainable_state["ray_version"],
                    ray.__version__,
                )
        keys_to_process.remove("trainable")
        self._current_step = int(state["current_step"])
        keys_to_process.remove("current_step")

        self._iteration = state["iteration"]
        keys_to_process.remove("iteration")
        # Setup
        # NOTE: setup.config can differ from new_algo_config when algorithm_overrides is used!
        # self._setup.config = new_algo_config  # TODO: Possible unset setup._config to not confuse configs
        # also _setup.args does not respect current CLI args!
        self._setup = self._setup.from_saved(state["setup"], init_trainable=False)
        keys_to_process.remove("setup")

        # Algorithm - steps are very likely skipped as it is a checkpointable component and was not pickled
        # Get algorithm state; fallback to only config (which however might not do anything)
        if "algorithm" in state:
            if self._algorithm is None:
                _logger.warning("Cannot set algorithm state as algorithm is None.")
            else:
                if self.algorithm.metrics and COMPONENT_METRICS_LOGGER in state["algorithm"]:
                    assert self.algorithm.metrics
                    self.algorithm.metrics.reset()
                for component in COMPONENT_ENV_RUNNER, COMPONENT_EVAL_ENV_RUNNER, COMPONENT_LEARNER_GROUP:
                    if component not in state["algorithm"]:
                        _logger.warning("Restoring algorithm without %s component in state.", component)
                self.algorithm.set_state(state["algorithm"])  # if this is in config might not be respected
        keys_to_process.discard("algorithm")

        # region Algorithm config
        algorithm_overrides = state.get("algorithm_overrides", None)
        algorithm_state_dict = state["algorithm_config"]
        if algorithm_overrides or self._perturbed_config:
            # What to do with old overwrites?
            if self._algorithm_overrides is None:
                self._algorithm_overrides = algorithm_overrides
            else:
                _logger.info(
                    "Not setting _algorithm_overrides to %s as it would overwrite present values %s. "
                    "Use _algorithm_overrides=None first to load them on set_state; use an empty dict "
                    "if you do not want to restore them.",
                    algorithm_overrides,
                    self._algorithm_overrides,
                )
            # NOTE: This is a state dict keys differ from a "from_dict"
            algorithm_state_dict: dict[str, Any] = algorithm_state_dict | (
                (self._algorithm_overrides or {}) | (self._perturbed_config or {})
            )
        # Fix private key with public property when using from_state, do not overwrite property
        if "train_batch_size_per_learner" in algorithm_state_dict:
            algorithm_state_dict["_train_batch_size_per_learner"] = algorithm_state_dict.pop(
                "train_batch_size_per_learner"
            )
        for k in algorithm_state_dict.keys():
            ATTR_NOT_FOUND = object()
            cls_attr = getattr(algorithm_state_dict["class"], k, ATTR_NOT_FOUND)
            if cls_attr is ATTR_NOT_FOUND:  # instance attribute
                continue
            if isinstance(cls_attr, property):
                _logger.error(
                    "%s is a property of class %s. State contains key overwriting this property. "
                    "This can lead to unexpected behavior.",
                    k,
                    algorithm_state_dict["class"],
                )
        # state["algorithm_config"] contains "class" to restore the correct config class
        new_algo_config = AlgorithmConfig.from_state(algorithm_state_dict)
        if type(new_algo_config) is not type(self.algorithm_config):
            _logger.warning(
                "Restored config class %s differs from expected class %s", type(new_algo_config), type(self.config)
            )
        new_algo_config = cast("_ConfigType", new_algo_config)
        num_envs_before = self.algorithm_config.num_envs_per_env_runner
        num_envs_new = new_algo_config.num_envs_per_env_runner
        recreate_envs = num_envs_before != num_envs_new
        if recreate_envs:
            _logger.info(
                "Amount of vectorized envs changed from %d to %d. Need to recreate envs, this is expensive.",
                num_envs_before,
                num_envs_new,
            )
        did_reset = self._algorithm is not None and self._algorithm.reset_config(
            algorithm_state_dict
        )  # likely does nothing
        if not did_reset:
            # NOTE: does not SYNC config if env_runners / learners not in components we do that below
            # NOTE: evaluation_config might also not be set!
            self.algorithm_config = new_algo_config

        # Update env_runners after restore
        # check if config has been restored correctly - TODO: Remove after more testing
        if self._algorithm:
            from ray_utilities.testing_utils import TestHelpers

            config1_dict = TestHelpers.filter_incompatible_remote_config(self.algorithm_config.to_dict())
            config2_dict = TestHelpers.filter_incompatible_remote_config(self.algorithm.env_runner.config.to_dict())
            if self._algorithm.env_runner and (config1_dict != config2_dict):
                _logger.info(  # Sync below will make configs match
                    "Updating env_runner config after restore, did not match after set_state",
                )
                self._algorithm.env_runner.config = self.algorithm_config.copy(copy_frozen=True)
                if self._algorithm.learner_group is not None and self._algorithm.learner_group.is_local:
                    self._algorithm.learner_group._learner.config = self.algorithm_config.copy(copy_frozen=True)  # pyright: ignore[reportOptionalMemberAccess]
        if self._algorithm is not None:  # Otherwise algorithm will be created later
            self._rebuild_algorithm_if_necessary(new_algo_config)
            sync_env_runner_states_after_reload(self.algorithm)  # NEW, Test, sync states here
        if False and self._algorithm.env_runner_group:
            # TODO: remove block
            # Passing config here likely has no effect at all; possibly sync metrics with custom function
            # Does not sync config!, recreate env_runner_group or force sync. Best via reference
            self.algorithm.env_runner_group.sync_env_runner_states(config=self.algorithm_config)
            if recreate_envs and self.algorithm.env_runner and self.algorithm_config.num_env_runners == 0:
                self.algorithm.env_runner.make_env()
            if (self.algorithm_config.num_env_runners or 0) > 0:
                remote_config_ref = ray.put(self.algorithm_config)
                self.algorithm.env_runner_group._remote_config_obj_ref = remote_config_ref
                self.algorithm.env_runner_group._remote_config = self.algorithm_config.copy(copy_frozen=True)

                def set_env_runner_config(
                    r: EnvRunner, remote_config_ref=remote_config_ref, recreate_envs=recreate_envs
                ):
                    r.config = ray.get(remote_config_ref)
                    if recreate_envs:
                        r.make_env()

                self.algorithm.env_runner_group.foreach_env_runner(set_env_runner_config, local_env_runner=False)
        keys_to_process.remove("algorithm_config")
        keys_to_process.remove("algorithm_overrides")
        # endregion

        self._pbar = restore_pbar(state["pbar_state"])
        if is_pbar(self._pbar):
            self._pbar.set_description("Loading checkpoint... (pbar)")
        keys_to_process.remove("pbar_state")

        assert RewardUpdaters.__required_keys__ <= state["reward_updaters"].keys(), (
            "Reward updaters state does not contain all required keys: "
            f"{state['reward_updaters'].keys()} vs {RewardUpdaters.__required_keys__}"
        )
        self._reward_updaters = cast(
            "RewardUpdaters", {k: create_running_reward_updater(v) for k, v in state["reward_updaters"].items()}
        )
        keys_to_process.remove("reward_updaters")
        if self._git_repo_sha == _UNKNOWN_GIT_SHA:
            self._git_repo_sha = state.get("git_sha", _UNKNOWN_GIT_SHA)
            if self._git_repo_sha != _UNKNOWN_GIT_SHA:
                self._git_repo_sha += "_restored"  # unsure what the actual git commit is
        elif self._git_repo_sha != state.get("git_sha", _UNKNOWN_GIT_SHA):
            _logger.info(
                "Git repo sha has changed %s vs %s",
                self._git_repo_sha,
                state.get("git_sha", _UNKNOWN_GIT_SHA),
            )
        keys_to_process.discard("git_sha")

        if len(keys_to_process) > 0:
            _logger.warning(
                "The following keys were not processed during set_state: %s",
                ", ".join(keys_to_process),
            )

    @override(Checkpointable)
    def get_checkpointable_components(self) -> list[tuple[str, Checkpointable]]:
        components = super().get_checkpointable_components()
        if self._algorithm is not None:  # pyright: ignore[reportUnnecessaryComparison]
            components.append(("algorithm", self.algorithm))
        return components

    @override(Checkpointable)
    def get_ctor_args_and_kwargs(self) -> tuple[tuple, dict[str, Any]]:
        """Returns the args/kwargs used to create `self` from its constructor.

        Returns:
            A tuple of the args (as a tuple) and kwargs (as a Dict[str, Any]) used to
            construct `self` from its class constructor.
        """
        config = self.config.copy()
        kwargs = {"config": config, "algorithm_overrides": self._algorithm_overrides}  # possibly add setup_class
        args = ()
        return args, kwargs

    @override(Checkpointable)
    def get_metadata(self) -> dict:
        """Returns JSON writable metadata further describing the implementing class.

        Note that this metadata is NOT part of any state and is thus NOT needed to
        restore the state of a Checkpointable instance from a directory. Rather, the
        metadata will be written into `self.METADATA_FILE_NAME` when calling
        `self.save_to_path()` for the user's convenience.

        Returns:
            dict: JSON-encodable metadata information.

            Default contents::

                {
                    "class_and_ctor_args_file": self.CLASS_AND_CTOR_ARGS_FILE_NAME,
                    "state_file": self.STATE_FILE_NAME,
                    "ray_version": ray.__version__,
                    "ray_commit": ray.__commit__,
                    "repo_sha": self._git_repo_sha,
                }

        """
        metadata = super().get_metadata()
        metadata["ray_utilities_version"] = importlib.metadata.version("ray_utilities")
        if self._git_repo_sha == _UNKNOWN_GIT_SHA:
            try:
                repo = git.Repo(search_parent_directories=True)
            except (git.InvalidGitRepositoryError, Exception):
                # For some reason defined value was not kept :/
                _logger.warning(
                    "_git_repo_sha was not set and current workdir is not a repository. "
                    "Checking: os.environ['TUNE_ORIG_WORKING_DIR']"
                )
                try:
                    repo = git.Repo(os.environ["TUNE_ORIG_WORKING_DIR"], search_parent_directories=True)
                except KeyError as e:
                    _logger.error("KeyError %s not set, cannot find git repo for metadata", e)
                except (git.InvalidGitRepositoryError, Exception) as e:
                    _logger.error("Failed to get git Repo for metadata: %s", e)
            else:
                self._git_repo_sha = cast("git.types.AnyGitObject", repo.head.object).hexsha
        metadata["repo_sha"] = self._git_repo_sha
        return metadata

    # endregion checkpoints

    def step(self) -> LogMetricsDict:
        # Update self._current_step in child class
        raise NotImplementedError("Subclasses must implement the `step` method.")

    if TYPE_CHECKING:  # update signature

        def train(self) -> AutoExtendedLogMetricsDict:  # pyright: ignore[reportIncompatibleMethodOverride]
            return super().train()  # pyright: ignore[reportReturnType]

    def __del__(self):
        # Cleanup the pbar if it is still open
        try:
            if is_pbar(self._pbar):
                self._pbar.close()
        except:  # noqa: E722
            pass
        try:
            self.cleanup()
        except:  # noqa: E722
            pass
        try:
            self.stop()
        except:  # noqa: E722
            pass

    # if TYPE_CHECKING:  # want to return -> Self

    @classmethod
    def from_checkpoint(
        cls,
        path: str | pathlib.Path,
        filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        **kwargs,
    ) -> Self:
        # check if pickle file with type, args and kwargs can be found - ray fails silently
        # Duplication of `from_checkpoint` code:
        # ----- TODO possibly remove after testing ------
        # We need a string path for the `PyArrow` filesystem.
        path = path if isinstance(path, str) else path.as_posix()

        # If no filesystem is passed in create one.
        if path and not filesystem:
            # Note the path needs to be a path that is relative to the
            # filesystem (e.g. `gs://tmp/...` -> `tmp/...`).
            filesystem, path = pyarrow.fs.FileSystem.from_uri(path)
        # Only here convert to a `Path` instance b/c otherwise
        # cloud path gets broken (i.e. 'gs://' -> 'gs:/').
        path = pathlib.Path(path)

        # Get the class constructor to call and its args/kwargs.
        # Try reading the pickle file first, ray fails silently in case of an error.
        try:
            assert filesystem is not None
            with filesystem.open_input_stream((path / cls.CLASS_AND_CTOR_ARGS_FILE_NAME).as_posix()) as f:
                ctor_info = pickle.load(f)
            _ctor = ctor_info["class"]
            _ctor_args = force_list(ctor_info["ctor_args_and_kwargs"][0])
            _ctor_kwargs = ctor_info["ctor_args_and_kwargs"][1]
        except Exception:
            _logger.exception(
                "Failed to load class and ctor args from checkpoint at %s:",
                path,
            )
        # -----
        # from_checkpoint -> restore_from_path first restores subcomponents then calls set_state
        restored = super().from_checkpoint(path, filesystem=filesystem, **kwargs)
        restored = cast("Self", restored)
        # Restore algorithm metric states; see my PR https://github.com/ray-project/ray/pull/54148/
        # sync_env_runner_states_after_reload(restored.algorithm)
        # callbacks are not called by the above methods.
        make_callback(
            "on_checkpoint_loaded",
            # ray has a wrong type signature here, accepting only list
            callbacks_objects=restored.algorithm.callbacks,  # pyright: ignore[reportArgumentType]
            callbacks_functions=restored.algorithm.config.callbacks_on_checkpoint_loaded,  # pyright: ignore[reportArgumentType,reportOptionalMemberAccess]
            kwargs={"algorithm": restored.algorithm, "metrics_logger": restored.algorithm.metrics},
        )
        return restored


if TYPE_CHECKING:
    TrainableBase()  # check ABC


class _TrainableSubclassMeta(ABCMeta):
    """
    When restoring the locally defined trainable,
    rllib performs a subclass check, that fails without a custom hook.

    issubclass will be True if both classes are subclasses of TrainableBase class
    and the setup classes are subclasses of each other

    Because of https://github.com/python/cpython/issues/13671 do not use `__subclasshook__`
    and do not use issubclass(subclass, cls._base_cls) can cause recursion because of ABCMeta.
    """

    _base_cls: type[TrainableBase[Any, Any, Any]]
    setup_class: _ExperimentSetup[Any, Any, Any]

    def __new__(cls, name, bases, namespace, base: type[TrainableBase[Any, Any, Any]] = TrainableBase):
        namespace["_base_cls"] = base
        try:
            namespace["_git_repo_sha"] = next(
                sha for b in bases if (sha := getattr(b, "_git_repo_sha", _UNKNOWN_GIT_SHA)) and sha != _UNKNOWN_GIT_SHA
            )
        except StopIteration:
            pass
        return super().__new__(cls, name, bases, namespace)

    def __subclasscheck__(cls, subclass: type[TrainableBase[Any, Any, Any] | Any]):
        if cls._base_cls not in subclass.mro():
            return False
        # Check that the setup class is also a subclass relationship
        if hasattr(subclass, "setup_class") and issubclass(
            (
                subclass.setup_class if isclass(subclass.setup_class) else type(subclass.setup_class)  # pyright: ignore[reportGeneralTypeIssues]
            ),
            cls.setup_class if isclass(cls.setup_class) else type(cls.setup_class),
        ):
            return True
        return False


class DefaultTrainable(TrainableBase[_ParserType, _ConfigType, _AlgorithmType]):
    """Default trainable implementation for Ray RLlib algorithms with Ray Tune integration.

    This is the primary trainable class for running Ray RLlib experiments within the
    Ray Utilities framework. It provides automatic training loop management, checkpointing,
    and evaluation integration with sensible defaults for most RL use cases.

    **Key Features:**

    - **Automatic Training Loop**: Handles the complete RL training cycle including
      evaluation episodes and metrics collection
    - **Smart Checkpointing**: Supports both iteration and step-based checkpointing
      with automatic frequency management
    - **Evaluation Integration**: Seamlessly integrates with discrete evaluation
      utilities for consistent episode-based evaluation
    - **Metrics Management**: Automatic reward tracking and statistical aggregation

    **Usage Patterns:**

    The class is typically used in one of two ways:

    1. **With Setup Classes** (Recommended):

       >>> trainable = DefaultTrainable.define(PPOSetup)
       >>> tuner = tune.Tuner(trainable, param_space={"lr": tune.grid_search([0.001, 0.01])})

    2. **Direct Instantiation** (Advanced):

       >>> trainable = DefaultTrainable()
       >>> trainable.setup(config={"env": "CartPole-v1", "lr": 0.001})

    **Training Process:**

    Each training step involves:

    1. Execute training iteration via :func:`~ray_utilities.training.functional.training_step`
    2. Run evaluation episodes if configured
    3. Update progress tracking and metrics
    4. Handle automatic checkpointing based on configured frequency

    **Configuration:**

    The trainable respects standard Ray Tune configuration along with Ray Utilities
    specific settings for evaluation, checkpointing, and progress tracking.

    Attributes:
        _last_checkpoint_iteration: Tracks the last checkpoint iteration for frequency control
        _last_checkpoint_step: Tracks the last checkpoint step for step-based checkpointing

    See Also:
        :class:`TrainableBase`: Base class with comprehensive state management
        :func:`~ray_utilities.training.functional.training_step`: Core training function
        :class:`~ray_utilities.setup.ExperimentSetupBase`: Setup framework for configuration
    """

    _last_checkpoint_iteration = -1
    _last_checkpoint_step = -1

    def step(self) -> LogMetricsDict:  # iteratively
        result, metrics, rewards = training_step(
            self.algorithm,
            reward_updaters=self._reward_updaters,
            discrete_eval=self.discrete_eval,
            disable_report=True,
            log_stats=self.log_stats,
        )
        self._current_step = get_current_step(result)
        if (
            "cli_args" in self.config
            and self._current_step
            > self.config["cli_args"].get("total_steps", float("inf"))
            + self.algorithm_config.train_batch_size_per_learner
        ):
            _logger.info(
                "Current step %s exceeds total steps. Expecting the trainable to have stopped.", self._current_step
            )
        # HACK: as long as tune does not allow custom result checkpointing use this
        # see for example: https://github.com/ray-project/ray/pull/55527
        if (
            TUNE_RESULT_IS_A_COPY
            and self._setup.args.checkpoint_frequency_unit == "steps"  # type: ignore
            and self._setup.args.checkpoint_frequency  # type: ignore
            and (_steps_since_last_checkpoint := self._current_step - self._last_checkpoint_step)
            >= self._setup.args.checkpoint_frequency
        ):
            _logger.info(
                "Creating checkpoint at step %s as last checkpoint was at step %s, difference %s >= %s (frequency)",
                self._current_step,
                self._last_checkpoint_step if self._last_checkpoint_step >= 0 else "Never",
                _steps_since_last_checkpoint,
                self._setup.args.checkpoint_frequency,
            )
            self._last_checkpoint_iteration = self._iteration  # iteration might be off by 1 as set after return
            self._last_checkpoint_step = self._current_step
            metrics[SHOULD_CHECKPOINT] = result[SHOULD_CHECKPOINT] = True
        # Update progress bar
        if is_pbar(self._pbar):
            update_pbar(
                self._pbar,
                rewards=rewards,
                metrics=metrics,
                result=result,
                current_step=self._current_step,
                total_steps=get_total_steps(self._total_steps, self.algorithm_config),
            )
            self._pbar.update()
        return metrics


if TYPE_CHECKING:  # check ABC
    DefaultTrainable()
