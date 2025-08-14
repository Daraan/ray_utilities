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
from typing import TYPE_CHECKING, Any, Collection, Generic, Optional, TypedDict, TypeVar, cast, overload

import pyarrow.fs
import ray
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_EVAL_ENV_RUNNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_METRICS_LOGGER,
)
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.tune.result import SHOULD_CHECKPOINT
from typing_extensions import Self, TypeAliasType

from ray_utilities.callbacks.progress_bar import restore_pbar, save_pbar_state, update_pbar
from ray_utilities.callbacks.tuner.metric_checkpointer import TUNE_RESULT_IS_A_COPY
from ray_utilities.config.typed_argument_parser import LOG_STATS, LogStatsChoices
from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
from ray_utilities.misc import is_pbar
from ray_utilities.training.functional import training_step
from ray_utilities.training.helpers import (
    create_running_reward_updater,
    episode_iterator,
    get_current_step,
    get_total_steps,
    setup_trainable,
    sync_env_runner_states_after_reload,
)
from ray_utilities.typing.trainable_return import RewardUpdaters

if TYPE_CHECKING:
    from ray.experimental import tqdm_ray
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.utils.typing import StateDict
    from tqdm import tqdm
    from typing_extensions import NotRequired

    from ray_utilities.callbacks.progress_bar import RangeState, RayTqdmState, TqdmState
    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase, SetupCheckpointDict
    from ray_utilities.typing import LogMetricsDict


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


def _validate_algorithm_config_afterward(func):
    """
    Decorator to validate the algorithm config after the function is called.
    This fixes some values on reloaded algorithms that can fail tests.
    """

    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.algorithm_config.validate()
        return result

    return wrapper


class TrainableStateDict(TypedDict):
    """Returned by `TrainableBase.get_state()`."""

    trainable: StateDict
    """The state obtained by tune.Trainable.get_state()."""

    algorithm: NotRequired[StateDict]  # component; can be ignored
    """Als Algorithm is a Checkpointable its state might not be saved here"""
    algorithm_config: StateDict
    algorithm_overrides: Optional[dict[str, Any]]
    iteration: int
    pbar_state: RayTqdmState | TqdmState | RangeState

    reward_updaters: dict[str, list[float]]

    setup: SetupCheckpointDict[Any, Any, Any]

    current_step: int


class PartialTrainableStateDict(TypedDict, total=False):
    """Returned by `TrainableBase.get_state()`."""

    trainable: StateDict
    """The state obtained by tune.Trainable.get_state()."""

    algorithm: StateDict
    algorithm_config: StateDict
    iteration: int
    pbar_state: RayTqdmState | TqdmState | RangeState

    reward_updaters: dict[str, list[float]]

    setup: SetupCheckpointDict[Any, Any, Any]


class TrainableBase(Checkpointable, tune.Trainable, Generic[_ParserType, _ConfigType, _AlgorithmType]):
    """
    Loading logic:
        - (classmethod) from_checkpoint -> restore_from_path -> set_state
            looks for loads class_and_ctor_args.pkl -> class, args&kwargs

    Methods:

        - Checkpointable methods:
            save_to_path()  # available in super
                calls: get_metadata(), pickles type(self) and ctor_args_and_kwargs, get_state
            restore_from_path()  # available in super, calls set_state and iterates subcomponents
            from_checkpoint()  # available in super, calls restore_from_path
            get_state()
            set_state()
            get_ctor_args_and_kwargs()
            get_metadata()
            get_checkpointable_components() # available in super, extend

        - Trainable methods:
            setup()
            step()   # Keep abstract
            save_checkpoint()  # we call save_to_path
            load_checkpoint() -> restore_from_path (if path) or ... (when dict)
            reset_config()
            cleanup()
            save()  # available; calls save_checkpoint -> save_to_path
            restore()  # available; calls load_checkpoint
    """

    setup_class: _ExperimentSetup[_ParserType, _ConfigType, _AlgorithmType]
    """
    Defines the setup class to use for this trainable, needs a call to `define` to create a subclass.
    with this value set.
    """
    discrete_eval: bool = False
    use_pbar: bool = True

    @classmethod
    def define(
        cls,
        setup_cls: _ExperimentSetup[_ParserTypeInner, _ConfigTypeInner, _AlgorithmTypeInner],
        *,
        discrete_eval: bool = False,
        use_pbar: bool = True,
        fix_argv: bool = False,
    ) -> type[DefaultTrainable[_ParserTypeInner, _ConfigTypeInner, _AlgorithmTypeInner]]:
        """
        This creates a subclass with ``setup_class`` set to the given class.

        Args:
            fix_argv: If True, the current sys.argv will be fixed to the setup_class args.
                When instantiated - and not other args are explicitly provided during __init__ -
                these saved args are used to initialize the setup_class.
                **disregarding the current sys.argv**, this is necessary in remote contexts where
                the initial sys.argv is not available.
        """
        # Avoid undefined variable error in class body
        discrete_eval_ = discrete_eval
        use_pbar_ = use_pbar
        # Fix current cli args to the trainable - necessary for remote
        if fix_argv:
            if isclass(setup_cls):
                setup_cls = type(setup_cls.__name__ + "FixedArgv", (setup_cls,), {"_fixed_argv": sys.argv})

        class DefinedTrainable(
            cls,
            metaclass=_TrainableSubclassMeta,
            base=cls,
        ):
            setup_class = setup_cls
            discrete_eval = discrete_eval_
            use_pbar = use_pbar_

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
        **kwargs,
    ):
        self._algorithm_overrides = algorithm_overrides
        if self._algorithm_overrides and self.setup_class._fixed_argv:
            _logger.warning(
                "Using a Trainable with fixed argv on the setup_class and algorithm_overrides, "
                "might result in unexpected values after a restore. Test carefully."
            )
            # NOTE: Use get_ctor_args_and_kwargs to include the overwrites on a reload
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

        self._current_step: int = 0
        """The current env steps sampled by the trainable, updated by step()."""

    @override(tune.Trainable)
    def setup(self, config: dict[str, Any], *, algorithm_overrides: Optional[dict[str, Any]] = None) -> None:
        """
        Sets:
            - algorithm
            - _pbar
            - _iteration
            - _setup
            - _reward_updaters
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
            self._setup = self.setup_class()  # XXX # FIXME # correct args; might not work when used with Tuner
        else:
            self._setup = self.setup_class
        # TODO: Possible unset setup._config to not confuse configs (or remove setup totally?)
        # use args = config["cli_args"] # XXX
        import sys
        from pprint import pformat

        print("Sys argv during Trainable.setup()", sys.argv)
        print(
            "args",
            "(in config)" if "cli_args" in config else "(on setup)",
            "are:\n",
            pformat(config.get("cli_args", self._setup.args)),
        )
        # NOTE: args is a dict, self._setup.args a Namespace | Tap
        self._reward_updaters: RewardUpdaters
        args, _algo_config, self.algorithm, self._reward_updaters = setup_trainable(
            hparams=config,
            setup=self._setup,
            setup_class=self.setup_class if isclass(self.setup_class) else None,
            config_overrides=self._algorithm_overrides,
        )
        self._param_overrides: dict[str, Any] = args.get("__overwritten_keys__", {})
        """Changed parameters via the hparams argument, e.g. passed by the tuner. See also: --tune"""
        self._pbar: tqdm | range | tqdm_ray.tqdm = episode_iterator(args, config, use_pbar=self.use_pbar)
        self._iteration: int = 0
        self.log_stats: LogStatsChoices = args[LOG_STATS]
        assert self.algorithm.config
        # calculate total steps once
        # After components have been setup up load checkpoint if requested
        current_step = 0
        if "cli_args" in config and config["cli_args"].get("from_checkpoint"):
            _logger.info("At end of setup(), loading from checkpoint: %s", config["cli_args"]["from_checkpoint"])
            # calls restore from path; from_checkpoint could also be a dict here
            self.load_checkpoint(config["cli_args"]["from_checkpoint"])
            if self.algorithm.metrics:
                current_step = self.algorithm.metrics.peek(
                    (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), default=None
                )
                if current_step is None:
                    current_step = self.algorithm.metrics.peek(
                        (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=None
                    )
                if current_step is None:
                    _logger.warning(
                        "No current step found in restored algorithm metrics to re-calculate total_steps, using 0. "
                    )

        # Resulting steps are divisible by current train_batch_size_per_learner
        # Does not allow for current_steps (divisible by old batch size) + current batch_size
        args["iterations"] -= self.algorithm._iteration
        total_steps = get_total_steps(args, self.algorithm.config)
        if total_steps is not None:
            # on reload, old batch_size might not be divisible by new batch size
            # account for past iterations with different batch size
            total_steps += current_step
        self._total_steps = {
            "total_steps": total_steps,
            "iterations": "auto",
        }
        args["iterations"] += self.algorithm._iteration

    @property
    def algorithm_config(self) -> _ConfigType:
        """
        Config of the algorithm used.
        Note:
            This is a copy of the setup's config which might has been further modified.
        """
        return self.algorithm.config  # pyright: ignore[reportReturnType]

    @algorithm_config.setter
    def algorithm_config(self, value: _ConfigType):
        self.algorithm.config = value

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
        self.algorithm.cleanup()
        if is_pbar(self._pbar):
            self._pbar.close()

    # endregion Trainable setup

    # region checkpointing

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
        algo_kwargs: dict[str, Any] = (
            {**kwargs}
            if ignore_setup  # NOTE: also ignores overrides on self
            else {"config": self._setup.config, **kwargs}
        )

        if "config" in algo_kwargs and (self._algorithm_overrides or self._param_overrides):
            config_overrides = (self._algorithm_overrides or {}) | self._param_overrides
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
                _logger.warning(
                    "minibatch_size %d is larger than train_batch_size_per_learner %d, this can result in an error. "
                    "Reducing the minibatch_size to the train_batch_size_per_learner.",
                    algo_kwargs["config"].minibatch_size,
                    algo_kwargs["config"].train_batch_size_per_learner,
                )
                config_overrides["minibatch_size"] = algo_kwargs["config"].train_batch_size_per_learner
                algo_kwargs["config"].minibatch_size = algo_kwargs["config"].train_batch_size_per_learner
            algo_kwargs["config"].freeze()
        else:
            config_overrides = {}
        overrides_at_start = self._algorithm_overrides or {}
        if isinstance(checkpoint, dict):
            keys_to_process = set(checkpoint.keys())  # Sanity check if processed all keys

            # from_checkpoint calls restore_from_path which calls set state
            # if checkpoint["algorithm_checkpoint_dir"] is a tempdir (e.g. from tune, this is wrong)
            # However, self.set_state should have take care of algorithm already even if checkpoint dir is missing
            if os.path.exists(checkpoint["algorithm_checkpoint_dir"]):
                self.algorithm.stop()  # free resources first
                self.algorithm = self.algorithm.from_checkpoint(
                    Path(checkpoint["algorithm_checkpoint_dir"]).absolute().as_posix(), **algo_kwargs
                )
            elif "algorithm_state" in checkpoint:
                _logger.error(
                    "Algorithm checkpoint directory %s does not exist, will restore from state",
                    checkpoint["algorithm_checkpoint_dir"],
                )
                if self._algorithm_overrides:
                    checkpoint["algorithm_state"]["config"] = checkpoint["algorithm_state"]["config"].update_from_dict(
                        self._algorithm_overrides
                    )
                self.algorithm.set_state(checkpoint["algorithm_state"])
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

            assert len(keys_to_process) == 0, f"Not all keys were processed during load_checkpoint: {keys_to_process}"
        elif checkpoint is not None:
            components = {c[0] for c in self.get_checkpointable_components()}
            components.discard("algorithm")
            # Restore from path does not account for new algorithm_config; so this merely sets the state
            # use from_checkpoint to do that
            self.restore_from_path(checkpoint, **algo_kwargs)
            # Restored overrides:
            if (
                "config" in algo_kwargs
                and self._algorithm_overrides
                and overrides_at_start != self._algorithm_overrides
            ):
                # Overrides at start should have higher priority
                algo_kwargs["config"] = (
                    algo_kwargs["config"]
                    .copy(copy_frozen=False)
                    # Restored < algorithm_overrides < hparams
                    .update_from_dict(self._algorithm_overrides | config_overrides)
                )
                # Fix minibatch size < batch_size if reloaded bad value
                if (
                    algo_kwargs["config"].minibatch_size is not None
                    and algo_kwargs["config"].train_batch_size_per_learner < algo_kwargs["config"].minibatch_size
                ):
                    _logger.warning(
                        "minibatch_size %d is larger than train_batch_size_per_learner %d, this can result in an error."
                        " Reducing the minibatch_size to the train_batch_size_per_learner.",
                        algo_kwargs["config"].minibatch_size,
                        algo_kwargs["config"].train_batch_size_per_learner,
                    )
                    config_overrides["minibatch_size"] = algo_kwargs["config"].train_batch_size_per_learner
                    algo_kwargs["config"].minibatch_size = algo_kwargs["config"].train_batch_size_per_learner
                algo_kwargs["config"].freeze()
            # return
            # for component in components:
            #    self.restore_from_path(checkpoint, component=component, **algo_kwargs)
            # free resources first
            self.algorithm.stop()
            self.algorithm = self.algorithm.from_checkpoint((Path(checkpoint) / "algorithm").as_posix(), **algo_kwargs)
            sync_env_runner_states_after_reload(self.algorithm)
        else:
            raise ValueError(f"Checkpoint must be a dict or a path. Not {type(checkpoint)}")

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
        self._current_step = state["current_step"]
        keys_to_process.remove("current_step")

        self._iteration = state["iteration"]
        keys_to_process.remove("iteration")

        # state["algorithm_config"] contains "class" to restore the correct config class
        new_algo_config = AlgorithmConfig.from_state(state["algorithm_config"])
        if type(new_algo_config) is not type(self.algorithm_config):
            _logger.warning(
                "Restored config class %s differs from expected class %s", type(new_algo_config), type(self.config)
            )
        new_algo_config = cast("_ConfigType", new_algo_config)
        did_reset = self.algorithm.reset_config(state["algorithm_config"])  # likely does nothing
        if not did_reset:
            self.algorithm_config = (
                new_algo_config  # NOTE: does not SYNC config if env_runners / learners not in components
            )
            # NOTE: evaluation_config might also not be set!
        keys_to_process.remove("algorithm_config")

        # Setup
        # NOTE: setup.config can differ from new_algo_config when algorithm_overrides is used!
        # self._setup.config = new_algo_config  # TODO: Possible unset setup._config to not confuse configs
        # also _setup.args does not respect current CLI args!
        self._setup = self._setup.from_saved(state["setup"], init_trainable=False)
        keys_to_process.remove("setup")

        algorithm_overrides = state.get("algorithm_overrides", None)
        algorithm_config = state["algorithm_config"]
        if algorithm_overrides:
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
            algorithm_config = algorithm_config.copy() | self._algorithm_overrides
        keys_to_process.remove("algorithm_overrides")

        # algorithm might not be in state as it is a checkpointable component and was not pickled
        if "algorithm" in state:
            if self.algorithm.metrics and COMPONENT_METRICS_LOGGER in state["algorithm"]:
                self.algorithm.metrics.reset()
            for component in COMPONENT_ENV_RUNNER, COMPONENT_EVAL_ENV_RUNNER, COMPONENT_LEARNER_GROUP:
                if component not in state["algorithm"]:
                    _logger.warning("Restoring algorithm without %s component in state.", component)
        # Get algorithm state; fallback to only config (which however might not do anything)
        # NOTE: This sync env_runner -> eval_env_runner which causes wrong env_steps_sampled metric
        # TODO: # XXX as algorithm is not in state and restored via components, state might not be correct
        algo_state = state.get("algorithm")
        if algo_state:
            if COMPONENT_METRICS_LOGGER in algo_state:
                assert self.algorithm.metrics
                self.algorithm.metrics.reset()
            self.algorithm.set_state(algo_state)
        # Update env_runners after restore
        # check if config has been restored correctly - TODO: Remove after more testing
        from ray_utilities.testing_utils import TestHelpers

        config1_dict = TestHelpers.filter_incompatible_remote_config(self.algorithm_config.to_dict())
        config2_dict = TestHelpers.filter_incompatible_remote_config(self.algorithm.env_runner.config.to_dict())
        if self.algorithm.env_runner and (config1_dict != config2_dict):
            _logger.info(  # Sync below will make configs match
                "Updating env_runner config after restore, did not match after set_state",
            )
            self.algorithm.env_runner.config = self.algorithm_config.copy(copy_frozen=True)
        if self.algorithm.env_runner_group:
            # TODO: Passing config here likely has no effect at all; possibly sync metrics with custom function
            # Does not sync config!, recreate env_runner_group or force sync. Best via reference
            self.algorithm.env_runner_group.sync_env_runner_states(config=self.algorithm_config)
            if (self.algorithm_config.num_env_runners or 0) > 0:
                remote_config_ref = ray.put(self.algorithm_config)
                self.algorithm.env_runner_group._remote_config_obj_ref = remote_config_ref
                self.algorithm.env_runner_group._remote_config = self.algorithm_config.copy(copy_frozen=True)

                def set_env_runner_config(r: EnvRunner, remote_config_ref=remote_config_ref):
                    r.config = ray.get(remote_config_ref)

                self.algorithm.env_runner_group.foreach_env_runner(set_env_runner_config, local_env_runner=False)
        keys_to_process.discard("algorithm")

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

        if len(keys_to_process) > 0:
            _logger.warning(
                "The following keys were not processed during set_state: %s",
                ", ".join(keys_to_process),
            )

    @override(Checkpointable)
    def get_checkpointable_components(self) -> list[tuple[str, Checkpointable]]:
        components = super().get_checkpointable_components()
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
            A JSON-encodable dict of metadata information.

            By default:
                {
                    "class_and_ctor_args_file": self.CLASS_AND_CTOR_ARGS_FILE_NAME,
                    "state_file": self.STATE_FILE_NAME,
                    "ray_version": ray.__version__,
                    "ray_commit": ray.__commit__,
                }
        """
        metadata = super().get_metadata()
        metadata["ray_utilities_version"] = importlib.metadata.version("ray_utilities")
        try:
            import git  # noqa: PLC0415

            metadata["repo_sha"] = git.Repo(search_parent_directories=True).head.object.hexsha
        except Exception as e:  # noqa: BLE001
            _logger.warning("Could not get git commit SHA: %s", e)
            metadata["repo_sha"] = "unknown"
        return metadata

    # endregion checkpoints

    def step(self) -> LogMetricsDict:
        # Update self._current_step in child class
        raise NotImplementedError("Subclasses must implement the `step` method.")

    def __del__(self):
        # Cleanup the pbar if it is still open
        try:
            if is_pbar(self._pbar):
                self._pbar.close()
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
        sync_env_runner_states_after_reload(restored.algorithm)
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
    """Default trainable for ray.tune based on RLlib algorithms."""

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
