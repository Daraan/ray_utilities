from __future__ import annotations

import importlib.metadata
import logging
import sys
from copy import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Collection, Generic, Optional, TypedDict, TypeVar, cast

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

from ray_utilities.callbacks.progress_bar import restore_pbar, save_pbar_state, update_pbar
from ray_utilities.config.typed_argument_parser import LOG_STATS
from ray_utilities.misc import is_pbar
from ray_utilities.training.helpers import (
    episode_iterator,
    get_current_step,
    get_total_steps,
    setup_trainable,
    training_step,
)

if TYPE_CHECKING:
    from ray.experimental import tqdm_ray
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.utils.typing import StateDict
    from tqdm import tqdm
    from typing_extensions import NotRequired

    from ray_utilities.callbacks.progress_bar import RangeState, RayTqdmState, TqdmState
    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase
    from ray_utilities.typing import LogMetricsDict

_logger = logging.getLogger(__name__)

_ParserTypeInner = TypeVar("_ParserTypeInner", bound="DefaultArgumentParser")
_ConfigTypeInner = TypeVar("_ConfigTypeInner", bound="AlgorithmConfig")
_AlgorithmTypeInner = TypeVar("_AlgorithmTypeInner", bound="Algorithm")

_ParserType = TypeVar("_ParserType", bound="DefaultArgumentParser")
_ConfigType = TypeVar("_ConfigType", bound="AlgorithmConfig")
_AlgorithmType = TypeVar("_AlgorithmType", bound="Algorithm")


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

    reward_updaters: dict[str, list[int]]


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

    setup_class: type[ExperimentSetupBase[_ParserType, _ConfigType, _AlgorithmType]]
    """
    Defines the setup class to use for this trainable, needs a call to `define` to create a subclass.
    with this value set.
    """
    discrete_eval: bool = False
    use_pbar: bool = True

    @classmethod
    def define(
        cls,
        setup_cls: type[ExperimentSetupBase[_ParserTypeInner, _ConfigTypeInner, _AlgorithmTypeInner]],
        *,
        discrete_eval: bool = False,
        use_pbar: bool = True,
        fix_argv: bool = True,
    ) -> type[DefaultTrainable[_ParserTypeInner, _ConfigTypeInner, _AlgorithmTypeInner]]:
        """This creates a subclass with ``setup_class`` set to the given class."""
        # Avoid undefined variable error in class body
        discrete_eval_ = discrete_eval
        use_pbar_ = use_pbar
        # Fix current cli args to the trainable - necessary for remote
        if fix_argv:
            setup_cls = type(setup_cls.__name__, (setup_cls,), {"_fixed_argv": sys.argv})

        if TYPE_CHECKING:

            class DefinedDefaultTrainable(DefaultTrainable[Any, Any, Any]):
                setup_class = setup_cls
                discrete_eval = discrete_eval_
                use_pbar = use_pbar_

        else:

            class DefinedDefaultTrainable(DefaultTrainable[_ParserTypeInner, _ConfigTypeInner, _AlgorithmTypeInner]):
                setup_class = setup_cls
                discrete_eval = discrete_eval_
                use_pbar = use_pbar_

        return DefinedDefaultTrainable

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
        super().__init__(config or {}, **kwargs)  # calls setup
        self.config: dict[str, Any]
        """Not the AlgorithmConfig, config passed by the tuner"""

    @override(tune.Trainable)
    def setup(
        self, config: dict[str, Any], *, overwrite_algorithm: Optional[AlgorithmConfig | dict[str, Any]] = None
    ) -> None:  # called once
        # NOTE: Called during __init__
        # Setup algo, config, args, etc.
        if not hasattr(self, "setup_class"):
            raise ValueError(
                f"setup_class is not set on {self}. Use TrainableCls = {self.__class__}.define(setup_class) to set it.",
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
        self._setup = self.setup_class()  # XXX # FIXME # correct args; might not work when used with Tuner
        # TODO: Possible unset setup._config to not confuse configs (or remove setup totally?)
        # use args = config["cli_args"] # XXX
        import sys

        print("Sys argv during Trainable.setup()", sys.argv)
        print(self._setup.args)
        self.args, algo_config, algo, self._reward_updaters = setup_trainable(
            hparams=config, setup=self._setup, setup_class=self.setup_class, overwrite_config=self._overwrite_algorithm
        )
        self.algorithm: _AlgorithmType = algo
        self._pbar: tqdm | range | tqdm_ray.tqdm = episode_iterator(self.args, config, use_pbar=self.use_pbar)
        self._iteration: int = 0

    @property
    def algorithm_config(self) -> _ConfigType:
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
    def reset_config(self, new_config):
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
        self.algorithm.save_checkpoint(checkpoint_dir)
        save = {
            "pbar": self._pbar,  # NOTE: if range does not hold iteration
            "state": state,
            "algorithm_checkpoint_dir": checkpoint_dir,
            "iterations": self._iteration,
            "reward_updaters": self._reward_updaters,
            "setup": self._setup.save(),
            "args": SimpleNamespace(**self._setup.args_to_dict(self.args)),
        }
        # TODO: save & pickle to checkpoint_dir, possibly subclass Checkpointable as well
        # Call save_to_path here?
        return save

    @override(tune.Trainable)
    def load_checkpoint(self, checkpoint: Optional[dict] | str):
        # NOTE:from_checkpoint is a classmethod, this isn't
        # set pbar
        # set weights
        # set iterations
        # set reward_updaters
        _logger.warning("Loading checkpoint: %s", checkpoint)
        if isinstance(checkpoint, dict):
            # TODO
            self._pbar = checkpoint.get("pbar", self._pbar)
            # from_checkpoint calls restore_from_path which calls set state
            self.algorithm = self.algorithm.from_checkpoint(checkpoint.get("algorithm_checkpoint_dir", ""))  # pyright: ignore[reportAttributeAccessIssue]
            # Is set_state even needed?
            # Test loaded algo state:
            loaded_algo_state = self.algorithm.get_state(
                components=self._get_subcomponents("algorithm", None),
                not_components=force_list(self._get_subcomponents("algorithm", None)),
            )
            if checkpoint.get("algorithm_state"):
                if checkpoint["algorithm_state"] != loaded_algo_state:
                    _logger.error(
                        "Algorithm state in checkpoint differs from current algorithm state. "
                        "This may lead to unexpected behavior."
                    )
                    import unittest

                    unittest.TestCase().assertDictEqual(
                        checkpoint["algorithm_state"],
                        loaded_algo_state,
                        "Algorithm state in checkpoint differs from current algorithm state.",
                    )
                    self.algorithm.set_state(checkpoint.get("algorithm_state", {}))  # TODO remove if tests pass
            else:
                _logger.debug("No algorithm state found in checkpoint.")
            self._iteration = checkpoint.get("iterations", self._iteration)
            self._reward_updaters = checkpoint.get("reward_updaters", self._reward_updaters)
            self._setup = self._setup.from_saved(checkpoint.get("setup", self._setup))
            # self.config = self..config  # this is algorithm.config
            if is_pbar(self._pbar):
                self._pbar.set_description("Loading checkpoint... (pbar)")
        elif checkpoint is not None:
            self.restore_from_path(checkpoint)
        else:
            raise ValueError("Checkpoint must be a dict or a path.")

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
            # "reward_updaters": {k: cast("partial", v).keywords["reward_array"] for k, v in self._reward_updaters.items()},
            "reward_updaters": self._reward_updaters,
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

    @_validate_algorithm_config_afterward
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
        new_config = AlgorithmConfig.from_state(state["algorithm_config"])
        if type(new_config) is not type(self.algorithm_config):
            _logger.warning(
                "Restored config class %s differs from expected class %s", type(new_config), type(self.config)
            )
        new_config = cast("_ConfigType", new_config)
        did_reset = self.algorithm.reset_config(state["algorithm_config"])  # likely does nothing
        if not did_reset:
            self.algorithm_config = new_config  # NOTE: does not SYNC config if env_runners / learners not in components
            # NOTE: evaluation_config might also not be set!
        self._setup.config = new_config  # TODO: Possible unset setup._config to not confuse configs
        keys_to_process.remove("algorithm_config")

        for component in COMPONENT_ENV_RUNNER, COMPONENT_EVAL_ENV_RUNNER, COMPONENT_LEARNER_GROUP:
            if component not in state:
                _logger.warning("Restoring algorithm without %s component in state.", component)
        # NOTE: config very likely not used in set_state
        self.algorithm.set_state(state.get("algorithm", {"config": state["algorithm_config"]}))
        keys_to_process.remove("algorithm")

        self._pbar = restore_pbar(state["pbar_state"])
        keys_to_process.remove("pbar_state")

        self._reward_updaters = state["reward_updaters"]
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


if TYPE_CHECKING:
    TrainableBase()  # check ABC


class DefaultTrainable(TrainableBase[_ParserType, _ConfigType, _AlgorithmType]):
    """Default trainable for RLlib algorithms.

    This class is used to create a trainable that can be used with ray tune.
    It is a wrapper around the `default_trainable` function.
    """

    def step(self) -> LogMetricsDict:  # iteratively
        result, metrics, rewards = training_step(
            self.algorithm,
            reward_updaters=self._reward_updaters,
            discrete_eval=self.discrete_eval,
            disable_report=True,
            log_stats=self.args[LOG_STATS],
        )
        # Update progress bar
        if is_pbar(self._pbar):
            update_pbar(
                self._pbar,
                rewards=rewards,
                metrics=metrics,
                result=result,
                current_step=get_current_step(result),
                total_steps=get_total_steps(self.args, self.algorithm_config),
            )
            self._pbar.update()
        return metrics


if TYPE_CHECKING:  # check ABC
    DefaultTrainable()
