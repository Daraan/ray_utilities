from __future__ import annotations

from abc import ABCMeta
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
from copy import copy
from typing import TYPE_CHECKING, Any, Collection, Generic, Optional, TypedDict, TypeVar, cast
from inspect import isclass

import ray
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_EVAL_ENV_RUNNER,
    COMPONENT_LEARNER_GROUP,
)
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import Checkpointable
from typing_extensions import TypeAliasType

from ray_utilities.callbacks.progress_bar import restore_pbar, save_pbar_state, update_pbar
from ray_utilities.config.typed_argument_parser import LOG_STATS, LogStatsChoices
from ray_utilities.misc import is_pbar
from ray_utilities.training.functional import training_step
from ray_utilities.training.helpers import (
    create_running_reward_updater,
    episode_iterator,
    get_current_step,
    get_total_steps,
    setup_trainable,
)
from ray_utilities.typing.trainable_return import RewardUpdaters

if TYPE_CHECKING:
    from ray.experimental import tqdm_ray
    from ray.rllib.algorithms import Algorithm
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
    # TODO: What with own state, e.g. hparams passed?, not contained in get_state

    algorithm: NotRequired[StateDict]  # component; can be ignored
    algorithm_config: StateDict
    iteration: int
    pbar_state: RayTqdmState | TqdmState | RangeState

    reward_updaters: dict[str, list[float]]

    setup: SetupCheckpointDict[Any, Any, Any]


class TrainableBase(Checkpointable, tune.Trainable, Generic[_ParserType, _ConfigType, _AlgorithmType]):
    """
    Methods:

        - Checkpointable methods:
            save_to_path()  # available in super
                calls: get_metadata(), pickles type(self) and ctor_args_and_kwargs, get_state
            restore_from_path()  # available in super, calls set_state and iterates subcomponents
            from_checkpoint()  # available in super, restore_from_path
            get_state()
            set_state()
            get_ctor_args_and_kwargs()
            get_metadata()
            get_checkpointable_components() # available in super, extend

        - Trainable methods:
            setup()
            step()   # Keep abstract
            save_checkpoint()
            load_checkpoint()
            reset_config()
            cleanup()
            save()  # available; calls save_checkpoint
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
        # TODO: Allow instance of setup here as well
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

        assert issubclass(DefinedTrainable, TrainableBase)
        assert DefinedTrainable._base_cls is cls

        return DefinedTrainable

    # region Trainable setup

    @override(tune.Trainable)
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        overwrite_algorithm: Optional[AlgorithmConfig | dict[str, Any]] = None,
        **kwargs,
    ):
        self._overwrite_algorithm = overwrite_algorithm
        if self._overwrite_algorithm and self.setup_class._fixed_argv:
            _logger.warning(
                "Using a Trainable with fixed argv on the setup_class and overwrite_algorithm, "
                "might result in unexpected values after a restore. Test carefully."
            )
            # NOTE: Use get_ctor_args_and_kwargs to include the overwrites on a reload
        super().__init__(config or {}, **kwargs)  # calls setup
        # TODO: do not create loggers
        self.config: dict[str, Any]
        """Not the AlgorithmConfig, config passed by the tuner"""

    @override(tune.Trainable)
    def setup(
        self, config: dict[str, Any], *, overwrite_algorithm: Optional[AlgorithmConfig | dict[str, Any]] = None
    ) -> None:
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
                f"setup_class is not set on {self}. Use TrainableCls = {self.__class__.__name__}.define(setup_class) to set it.",
            )
        if overwrite_algorithm is not None and self._overwrite_algorithm is not None:
            if overwrite_algorithm != self._overwrite_algorithm:
                _logger.warning(
                    "Both `overwrite_algorithm` and `self._overwrite_algorithm` (during __init__) are set. "
                    "Consider passing only one. Overwriting self._overwrite_algorithm with new value."
                )
            else:
                _logger.error(
                    "Both `overwrite_algorithm` and `self._overwrite_algorithm` (during __init__) "
                    "and do not match. Overwriting self._overwrite_algorithm with new value from setup(). \n %s != %s "
                )
            self._overwrite_algorithm = overwrite_algorithm
        assert self.setup_class.parse_known_only is True
        _logger.debug("Setting up %s with config: %s", self.__class__.__name__, config)
        if "cli_args" in config and config["cli_args"].get("from_checkpoint") not in (None, ""):
            # calls restore from path; from_checkpoint could also be a dict here
            self.load_checkpoint(config["cli_args"]["from_checkpoint"])
            return
        if isclass(self.setup_class):
            self._setup = self.setup_class()  # XXX # FIXME # correct args; might not work when used with Tuner
        else:
            self._setup = self.setup_class
        # TODO: Possible unset setup._config to not confuse configs (or remove setup totally?)
        # use args = config["cli_args"] # XXX
        import sys

        print("Sys argv during Trainable.setup()", sys.argv)
        print(self._setup.args)
        # NOTE: args is a dict, self._setup.args a Namespace | Tap
        self._reward_updaters: RewardUpdaters
        args, _algo_config, self.algorithm, self._reward_updaters = setup_trainable(
            hparams=config, setup=self._setup, setup_class=self.setup_class, overwrite_config=self._overwrite_algorithm
        )
        self._pbar: tqdm | range | tqdm_ray.tqdm = episode_iterator(args, config, use_pbar=self.use_pbar)
        self._iteration: int = 0
        self.log_stats: LogStatsChoices = args[LOG_STATS]
        assert self.algorithm.config
        # calculate total steps once
        self._total_steps = {"total_steps": get_total_steps(args, self.algorithm.config), "iterations": "auto"}

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
        super().cleanup()
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
        state = self.get_state()  # TODO: check components
        # save in subdir
        algo_save_dir = (Path(checkpoint_dir) / "algorithm").as_posix()
        self.save_to_path(checkpoint_dir)  # saves components
        save = {
            "state": state,  # contains most information
            "algorithm_checkpoint_dir": algo_save_dir,
        }
        return save

    @override(tune.Trainable)
    def load_checkpoint(self, checkpoint: Optional[dict] | str):
        # NOTE: from_checkpoint is a classmethod, this isn't
        # set pbar
        # set weights
        # set iterations
        # set reward_updaters
        if isinstance(checkpoint, dict):
            # TODO
            keys_to_process = set(checkpoint.keys())  # Sanity check if processed all keys

            self.set_state(checkpoint["state"])
            keys_to_process.remove("state")

            # from_checkpoint calls restore_from_path which calls set state
            # if checkpoint["algorithm_checkpoint_dir"] is a tempdir (e.g. from tune, this is wrong)
            if not os.path.exists(checkpoint["algorithm_checkpoint_dir"]):
                if "algorithm_state" in checkpoint:
                    _logger.error(
                        "Algorithm checkpoint directory %s does not exist, will restore from state",
                        checkpoint["algorithm_checkpoint_dir"],
                    )
                    self.algorithm.set_state(checkpoint["algorithm_state"])
                else:
                    _logger.critical(
                        "Algorithm checkpoint directory %s does not exist, (possibly temporary path was saved) and no state provided. "
                        "Restored algorithm might be in an unexpected state.",
                        checkpoint["algorithm_checkpoint_dir"],
                    )
            else:
                self.algorithm = self.algorithm.from_checkpoint(checkpoint["algorithm_checkpoint_dir"])
            # Is set_state even needed?
            # Test loaded algo state:
            loaded_algo_state = self.algorithm.get_state(
                components=self._get_subcomponents("algorithm", None),
                not_components=force_list(self._get_subcomponents("algorithm", None)),
            )
            keys_to_process.remove("algorithm_checkpoint_dir")
            if False and checkpoint["state"]["algorithm"]:  # can add algorithm_state to check correctness
                if checkpoint["state"]["algorithm"] != loaded_algo_state:
                    _logger.error(
                        "Algorithm state in checkpoint differs from current algorithm state. "
                        "This may lead to unexpected behavior."
                    )
                    import unittest

                    tester = unittest.TestCase()
                    tester.maxDiff = 340_000  # Limit the max diff length
                    for key in loaded_algo_state.keys():
                        if isinstance(loaded_algo_state[key], dict) and isinstance(
                            checkpoint["state"]["algorithm"][key], dict
                        ):  # Check if both are dicts
                            try:
                                tester.assertDictEqual(
                                    checkpoint["state"]["algorithm"][key],
                                    loaded_algo_state[key],
                                    "Algorithm state in checkpoint differs from current algorithm state.",
                                )
                            except ValueError:
                                _logger.error("Cannot compare dicts for key %s", key)
                        else:
                            tester.assertEqual(
                                checkpoint["state"]["algorithm"][key],
                                loaded_algo_state[key],
                                "Algorithm state in checkpoint differs from current algorithm state.",
                            )
                    self.algorithm.set_state(checkpoint.get("algorithm_state", {}))  # TODO remove if tests pass
            else:
                # loaded from checkpoint
                _logger.debug("No algorithm state found in checkpoint.")

            assert len(keys_to_process) == 0, f"Not all keys were processed during load_checkpoint: {keys_to_process}"
        elif checkpoint is not None:
            self.restore_from_path(checkpoint)
        else:
            raise ValueError(f"Checkpoint must be a dict or a path. Not {type(checkpoint)}")

    # endregion

    # region Checkpointable methods

    @override(Checkpointable)
    @override(tune.Trainable)
    def get_state(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        components: Optional[str | Collection[str]] = None,
        *,
        not_components: Optional[str | Collection[str]] = None,
        **kwargs,  # noqa: ARG002
    ) -> TrainableStateDict | Any:
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
        trainable_state = super(Checkpointable, self).get_state()
        algorithm_config_state = self.algorithm_config.get_state()
        state: TrainableStateDict = {
            "trainable": trainable_state,
            "algorithm_config": algorithm_config_state,
            "iteration": self._iteration,
            "pbar_state": save_pbar_state(self._pbar, self._iteration),
            "reward_updaters": {k: v.keywords["reward_array"] for k, v in self._reward_updaters.items()},
            "setup": self._setup.save(),
        }
        if (not_components and "algorithm" not in not_components) or components is None or "algorithm" in components:
            if components and not isinstance(components, str):
                components = copy(components)
            algo_state = self.algorithm.get_state(
                self._get_subcomponents("algorithm", components),
                not_components=force_list(self._get_subcomponents("algorithm", not_components)),
            )  # TODO: check components
            algo_state["algorithm_class"] = type(self.algorithm)  # NOTE: Not msgpack-serializable
            algo_state["config"] = algorithm_config_state
            state["algorithm"] = algo_state
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
        keys_to_process = set(state.keys())
        try:
            super(Checkpointable, self).set_state(state.get("trainable", {}))  # pyright: ignore
        except AttributeError:
            # Currently no set_state method
            trainable_state = state["trainable"]
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
        self._setup = self._setup.from_saved(state["setup"], init_trainable=False)
        # NOTE: setup.config can differ from new_algo_config when overwrite_algorithm is used!
        # self._setup.config = new_algo_config  # TODO: Possible unset setup._config to not confuse configs
        keys_to_process.remove("setup")

        if "algorithm" in state:
            for component in COMPONENT_ENV_RUNNER, COMPONENT_EVAL_ENV_RUNNER, COMPONENT_LEARNER_GROUP:
                if component not in state["algorithm"]:
                    _logger.warning("Restoring algorithm without %s component in state.", component)
        # NOTE: config very likely not used in set_state
        self.algorithm.set_state(state.get("algorithm", {"config": state["algorithm_config"]}))
        # Update env_runners after restore
        if self.algorithm.env_runner and self.algorithm_config != self.algorithm.env_runner.config:
            _logger.debug("Updating env_runner config after restore, old did not match")
            self.algorithm.env_runner.config = self.algorithm_config.copy(copy_frozen=True)
        if self.algorithm.env_runner_group:
            self.algorithm.env_runner_group.sync_env_runner_states(config=self.algorithm_config)
        keys_to_process.remove("algorithm")

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
        kwargs = {"config": config, "overwrite_algorithm": self._overwrite_algorithm}  # possibly add setup_class
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
        raise NotImplementedError("Subclasses must implement the `step` method.")

    def __del__(self):
        # Cleanup the pbar if it is still open
        try:
            if is_pbar(self._pbar):
                self._pbar.close()
        except:  # noqa: E722
            pass


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

    def step(self) -> LogMetricsDict:  # iteratively
        result, metrics, rewards = training_step(
            self.algorithm,
            reward_updaters=self._reward_updaters,
            discrete_eval=self.discrete_eval,
            disable_report=True,
            log_stats=self.log_stats,
        )
        # Update progress bar
        if is_pbar(self._pbar):
            update_pbar(
                self._pbar,
                rewards=rewards,
                metrics=metrics,
                result=result,
                current_step=get_current_step(result),
                total_steps=get_total_steps(self._total_steps, self.algorithm_config),
            )
            self._pbar.update()
        return metrics


if TYPE_CHECKING:  # check ABC
    DefaultTrainable()
