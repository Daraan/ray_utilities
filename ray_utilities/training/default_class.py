from __future__ import annotations

import abc
import importlib.metadata
import logging
import sys
from copy import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Collection, Generic, Optional, TypedDict, TypeVar

import ray
from ray import tune
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_EVAL_ENV_RUNNER,
    COMPONENT_LEARNER_GROUP,
)
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
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
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


class TrainableStateDict(TypedDict):
    """Returned by `TrainableBase.get_state()`."""

    trainable: StateDict
    """The state obtained by tune.Trainable.get_state()."""
    algorithm: NotRequired[StateDict] # component; can be ignored
    config: StateDict
    iteration: int
    pbar_state: RayTqdmState | TqdmState | RangeState


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
        cls, setup_cls: type[ExperimentSetupBase[_ParserTypeInner, _ConfigTypeInner, _AlgorithmTypeInner]],
        *, discrete_eval: bool = False, use_pbar: bool = True, fix_argv: bool = True
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
    def setup(self, config: dict[str, Any]) -> None:   # called once
        # NOTE: Called during __init__
        # Setup algo, config, args, etc.
        if not hasattr(self, "setup_class"):
            raise ValueError(
                f"setup_class is not set on {self}. "
                f"Use TrainableCls = {self.__class__}.define(setup_class) to set it.",
            )
        self._setup = self.setup_class()  # XXX # FIXME # likely does not work
        # use args = config["cli_args"]
        import sys
        print("Sys argv:", sys.argv)
        print(self._setup.args)
        self.args, algo_config, algo, self._reward_updaters = setup_trainable(
            hparams=config,
            setup=self._setup,
            setup_class=self.setup_class,
        )
        self.algorithm: _AlgorithmType = algo
        self.config: _ConfigType = algo_config
        self._pbar: tqdm | range | tqdm_ray.tqdm = episode_iterator(self.args, config, use_pbar=self.use_pbar)
        self._iteration: int = 0

    @override(tune.Trainable)
    def reset_config(self, new_config):
        # Return True if the config was reset, False otherwise
        # This will be called when tune.TuneConfig(reuse_actors=True) is used
        # TODO
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
            loaded_algo_state = self.algorithm.get_state()
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
            self._iteration = checkpoint.get("iterations", self._iteration)
            self._reward_updaters = checkpoint.get("reward_updaters", self._reward_updaters)
            self._setup = self._setup.from_saved(checkpoint.get("setup", self._setup))
            self.config = self._setup.config
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
    def get_state( # pyright: ignore[reportIncompatibleMethodOverride]
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
        config_state = self.config.get_state()
        state = {
            "trainable": trainable_state,
            "config": config_state,
            "iteration": self._iteration,
            "pbar_state": save_pbar_state(self._pbar, self._iteration),
        }
        if (not_components and "algorithm" not in not_components) or components is None or "algorithm" in components:
            if components and not isinstance(components, str):
                components = copy(components)
            algo_state = self.algorithm.get_state(components, not_components=not_components)  # TODO: check components
            algo_state["algorithm_class"] = type(self.algorithm)  # NOTE: Not msgpack-serializable
            algo_state["config"] = config_state
            state["algorithm"] = algo_state
        return state

    @override(Checkpointable)
    def set_state(self, state: StateDict | TrainableStateDict) -> None:
        """Sets the implementing class' state to the given state dict.

        If component keys are missing in `state`, these components of the implementing
        class will not be updated/set.

        Args:
            state: The state dict to restore the state from. Maps component keys
                to the corresponding subcomponent's own state.
        """
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
        self._iteration = state["iteration"]
        self.config = self.config.from_state(state["config"])  # pyright: ignore[reportAttributeAccessIssue]
        did_reset = self.algorithm.reset_config(self.config.to_dict())  # likely does nothing
        if not did_reset:
            self.algorithm.config = self.config # NOTE: does not SYNC config if env_runners / learners not in components
            # NOTE: evaluation_config might also not be set!
        for component in COMPONENT_ENV_RUNNER, COMPONENT_EVAL_ENV_RUNNER, COMPONENT_LEARNER_GROUP:
            if component not in state:
                _logger.warning("Restoring algorithm without %s component.", component)
        # NOTE: config very likely not used in set_state
        self.algorithm.set_state(state.get("algorithm", {"config": state["config"]}))
        self._pbar = restore_pbar(state["pbar_state"])

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
        kwargs = {"config": config}  # possibly add setup_class
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
                total_steps=get_total_steps(self.args, self.config),
            )
            self._pbar.update()
        return metrics


if TYPE_CHECKING:  # check ABC
    DefaultTrainable()
