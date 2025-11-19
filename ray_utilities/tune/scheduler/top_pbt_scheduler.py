"""Advanced Population Based Training scheduler for Ray Tune hyperparameter optimization.

This module provides :class:`ReTuneScheduler`, an enhanced version of Ray Tune's
PopulationBasedTraining that supports grid search mutations, flexible quantile
fractions, and improved trial management for reinforcement learning experiments.

Key Components:
    - :class:`ReTuneScheduler`: Enhanced PBT scheduler with advanced features
    - :func:`_grid_search_sample_function`: Grid search sampling utilities
    - Integration with Ray Tune's trial and search algorithm framework

The scheduler extends the standard PBT approach with support for deterministic
grid search mutations and more flexible population management strategies.
"""

from __future__ import annotations

# pyright: enableExperimentalFeatures=true
import itertools
import logging
import math
import random
import shutil
from collections.abc import Iterable
from functools import partial
from itertools import cycle
from pathlib import Path
from time import time as get_time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Container,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
    overload,
)

import tree
from ray.train._internal.session import _FutureTrainingResult, _TrainingResult
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.result import TIME_TOTAL_S, TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.schedulers.pbt import PopulationBasedTraining, _fill_config
from ray.tune.utils import flatten_dict
from typing_extensions import Sentinel

from ray_utilities.constants import FORK_FROM, PERTURBED_HPARAMS, get_run_id
from ray_utilities.misc import (
    build_nested_dict,
    deep_freeze,
    flatten_mapping_with_path,
    get_current_step,
    get_value_by_path,
    make_experiment_key,
    make_fork_from_csv_header,
    make_fork_from_csv_line,
    warn_if_slow,
)
from ray_utilities.tune.scheduler.run_slow_trials_first_mixin import RunSlowTrialsFirstMixin
from ray_utilities.typing import ForkFromData, Forktime, ForktimeTuple

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from frozendict import frozendict
    from ray.tune.execution.tune_controller import TuneController
    from ray.tune.schedulers.pbt import _PBTTrialState
    from ray.tune.search.sample import Domain

    from ray_utilities.typing.algorithm_return import AlgorithmReturnData
    from ray_utilities.typing.metrics import FlatLogMetricsDict


logger = logging.getLogger(__name__)
_T = TypeVar("_T")


MAX_SKIP_LIST_LENGTH = 10000


if TYPE_CHECKING:
    from typing import type_check_only

    @type_check_only
    class _PBTTrialState2(_PBTTrialState):
        def __init__(self, trial: Trial):
            super().__init__(trial)

            # NOTE: These are type-check ONLY. Need to be set on_trial_add
            self.last_training_iteration: int  # not present before on_trial_add
            """The training iteration at which the last result was reported."""

            self.current_env_steps: int | None
            """The amount of (exact) env steps sampled"""

            self.last_update_timestamp: float = get_time()

            self.total_time_spent: float = 0.0


def _grid_search_sample_function(grid_search_space: Iterable[_T], *, repeat=True) -> Callable[[], _T]:
    """Create a function for sampling from a grid search space.

    Returns a parameterless function that yields grid search values either cyclically
    (with repetition) or once through the space (without repetition).

    Args:
        grid_search_space: Iterable containing the values to sample from.
        repeat: If True, cycle through values infinitely. If False, each value
            is returned once and then StopIteration is raised.

    Returns:
        A parameterless function that returns the next grid search sample.
        When repeat=False, the function raises StopIteration after all values
        have been returned once.

    Example:
        >>> sampler = _grid_search_sample_function([1, 2, 3], repeat=True)
        >>> sampler()  # Returns 1
        >>> sampler()  # Returns 2
        >>> sampler()  # Returns 3
        >>> sampler()  # Returns 1 (cycles back)
    """
    if repeat:
        cycler = cycle(grid_search_space)

        def cyclic_grid_iterator():
            return next(cycler)

        return cyclic_grid_iterator
    grid_search_space = list(grid_search_space)

    def grid_iterator():
        try:
            return grid_search_space.pop(0)
        except IndexError as e:
            raise StopIteration from e

    return grid_iterator


def _debug_dump_new_config(new_config: dict, mutation_keys: list[str]):
    logger.info("New config after perturbation %s", new_config)
    new_config[PERTURBED_HPARAMS] = {k: new_config[k] for k in mutation_keys}
    return new_config


def _insert_perturbed_hparams(new_config: dict, mutation_keys: list[str]) -> dict:
    new_config[PERTURBED_HPARAMS] = {k: new_config[k] for k in mutation_keys}
    return new_config


def wrap_custom_perturbed_hparams(func: Callable[[dict], dict], mutation_keys) -> Callable[[dict], dict]:
    def wrapper(new_config: dict) -> dict:
        return _insert_perturbed_hparams(func(new_config), mutation_keys)

    return wrapper


def _dummy_pass_through(new_config: dict) -> dict:
    return new_config


class KeepMutation(Generic[_T]):
    # need to be serializable, use just object at runtime
    _NOT_SET = Sentinel("_NOT_SET") if TYPE_CHECKING else object()
    NOT_FOUND = Sentinel("NOT_FOUND") if TYPE_CHECKING else object()

    def __init__(self, value: "_T | _NOT_SET" = _NOT_SET):
        self.value = value

    def set_value(self, new_value: _T):
        assert new_value not in (KeepMutation._NOT_SET, KeepMutation.NOT_FOUND), new_value
        self.value = new_value

    def __call__(self) -> _T:
        if self.value is KeepMutation._NOT_SET:
            raise ValueError("KeepMutation value not set. call set_value first.")
        return cast("_T", self.value)

    @staticmethod
    def get_config_value(config: dict[str, Any], path: tuple[str, ...]) -> _T | NOT_FOUND:
        """Given a config dict and a path (tuple of keys), return the value at that path."""
        current = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return KeepMutation.NOT_FOUND
            current = current[key]
        if isinstance(current, dict):
            logger.warning("KeepMutation path %s points to a dict, expected a value.", path)
        return current  # pyright: ignore[reportReturnType]


class _PerturbationSeed(int):
    """Represents a seed value added during environment reseeding for perturbation tracking."""


class _ReseedEnv:
    """
    Wraps a configuration dictionary to support environment reseeding during Population Based Training (PBT) perturbations.

    This class optionally applies a wrapping function to the config and, if an additional seed is specified,
    appends a `_PerturbationSeed` to the `env_seed` entry in the config. This enables tracking and reproducibility
    of environment changes caused by PBT perturbations.

    Args:
        wrap (Optional[Callable[[dict], dict]]): Optional function to apply to the config before reseeding.
        add_seed (Optional[int]): If provided, this integer is added as a `_PerturbationSeed` to the `env_seed`.
    """

    def __init__(
        self,
        wrap: Optional[Callable[[dict], dict]] = None,
        add_seed: Optional[int] = None,
        *,
        initial_seeds: Optional[dict[Trial | None, int | None]],
        next_trial: Optional[Trial] = None,
    ):
        self.wrap = wrap
        self.add_seed = add_seed
        self._trial_initial_seeds: dict[Trial | None, int | None] = initial_seeds or {}
        self._trial_initial_seeds[None] = None
        self._next_trial: Trial | None = next_trial

    def get_next_trial(self) -> Trial | None:
        """Get/Set the next trial to know which initial seed to use. After getting the next trial is cleared."""
        next_trial = self._next_trial
        self._next_trial = None
        return next_trial

    def set_next_trial(self, trial: Trial):
        self._next_trial = trial

    def __call__(self, config: dict) -> dict:
        config = self.wrap(config) if self.wrap else config
        if self.add_seed is None or "env_seed" not in config or config["env_seed"] is None:
            return config
        trial = self.get_next_trial()
        if trial is not None:
            base_seed = self._trial_initial_seeds.get(trial, config["env_seed"])
        else:
            base_seed = config["env_seed"]
        if isinstance(base_seed, int):
            config["env_seed"] = (base_seed, _PerturbationSeed(self.add_seed))
            return config
        # clean auto int from sequence
        env_seed: Sequence[int] = config["env_seed"]

        # Remove any perturbation seeds from last time, time attr should not influence
        cleaned_seed = self.remove_perturbation_seed(env_seed)
        config["env_seed"] = (*cleaned_seed, _PerturbationSeed(self.add_seed))
        return config

    @overload
    @staticmethod
    def remove_perturbation_seed(
        obj: Iterable[int | _PerturbationSeed | Iterable[object]],
    ) -> tuple[int | tuple, ...]: ...

    @overload
    @staticmethod
    def remove_perturbation_seed(obj: object) -> object | None: ...

    @staticmethod
    def remove_perturbation_seed(
        obj: Iterable[int | _PerturbationSeed | Iterable[object]] | object,
    ) -> tuple[int | tuple, ...] | object | None:
        if isinstance(obj, _PerturbationSeed):
            return None  # Remove this element
        if isinstance(obj, (tuple, list, Iterable)):
            # Recursively process sequences, filter out None
            cleaned = tuple(_ReseedEnv.remove_perturbation_seed(s) for s in obj)
            return tuple(x for x in cleaned if x is not None)
        return obj


class CyclicMutation(Generic[_T]):
    def __init__(self, values: Iterable[_T], skip: Optional[Container[_T]] = None, *, disable_skip: bool = False):
        self._cycler = cycle(values)
        self._values = list(values)
        self.skip = skip
        self.disable_skip = disable_skip
        self._warn_possible_infinite_loop()

    def update_values(self, new_values: Iterable[_T]):
        self._values = list(new_values)
        self._cycler = cycle(self._values)
        self._warn_possible_infinite_loop()

    def update_skip(self, new_values: Container[_T] | None):
        self.skip = new_values
        self._warn_possible_infinite_loop()

    def _warn_possible_infinite_loop(self) -> bool | None:
        try:
            if (
                self.skip and not self.disable_skip and len(self.skip) >= len(self._values)  # pyright: ignore[reportArgumentType]
            ):
                logger.warning(
                    "CyclicMutation skip list %s has length >= values %s. "
                    "This may cause an infinite loop when sampling.",
                    self.skip,
                    self._values,
                )
                return True
        except TypeError:
            # self.skip is only a container, not sized
            return None
        return False

    def __call__(self) -> _T:
        v = next(self._cycler)
        if self.disable_skip or self.skip is None:
            return v
        i = 0
        while self.skip and v in self.skip:
            v = next(self._cycler)
            i += 1
            if i > MAX_SKIP_LIST_LENGTH:
                loop = self._warn_possible_infinite_loop()
                if loop or loop is None:
                    raise RuntimeError(
                        "CyclicMutation appears to be stuck in an infinite loop due to skip list. "
                        "Increase MAX_SKIP_LIST_LENGTH."
                    )
            elif i % 1000 == 0:
                loop = self._warn_possible_infinite_loop()
                logger.warning(
                    "CyclicMutation still searching for non-skipped value after %s attempts. Is infinite loop: %s",
                    i,
                    loop or ("unknown (not Sized)" if loop is None else "no"),
                )
        return v


SAVE_ALL_CHECKPOINTS = False


class TopPBTTrialScheduler(RunSlowTrialsFirstMixin, PopulationBasedTraining):
    """Enhanced Population Based Training scheduler with grid search and flexible quantiles.

    This scheduler extends Ray Tune's PopulationBasedTraining with support for grid search
    mutations, and improved trial management for reinforcement
    learning experiments.
    The most prominent change is that all trials outside of the top quantile are exploited.
    That is is changes, for a quantile fraction of 0.1, all 90% of trials are exploited,
    instead of only the lowest 10% and keeping the other 80% as is.

    Key enhancements over standard PBT:
        - Grid search mutations for deterministic hyperparameter exploration
        - Custom exploration functions with mutation tracking
        - Enhanced logging and debugging capabilities

    The scheduler maintains compatibility with the standard PBT interface while providing
    additional flexibility for advanced hyperparameter optimization strategies.

    Args:
        time_attr: Attribute to use for time progression tracking.
        metric: Metric name to optimize (e.g., "episode_reward_mean").
        mode: Optimization mode, either "max" or "min".
        perturbation_interval: Number of time units between perturbations.
        burn_in_period: Time units before perturbations begin.
        hyperparam_mutations: Dictionary mapping hyperparameter names to mutation
            specifications. Supports grid_search definitions for deterministic sampling.
        quantile_fraction: Fraction of population to consider as "top performers".
        resample_probability: Probability of resampling parameters during perturbation.
        perturbation_factors: Tuple of (lower, upper) factors for parameter perturbation.
        custom_explore_fn: Optional custom function for exploration logic.
        log_config: Whether to log configuration changes.
        require_attrs: Whether to require time_attr and metric in results.
        synch: Whether to use synchronous perturbation.
        reseed: When trials are perturbed the config key "env_seed" is updated to (original_seed, current_step).
            Otherwise when using seeded environments they likely start at the same first observation
            the trial has seen before. Does nothing if there is no "env_seed" in the config.

    Example:
        >>> scheduler = ReTuneScheduler(
        ...     metric="episode_reward_mean",
        ...     mode="max",
        ...     perturbation_interval=50000,
        ...     hyperparam_mutations={
        ...         "lr": {"grid_search": [1e-4, 5e-4, 1e-3]},
        ...         "batch_size": {"grid_search": [64, 128, 256]},
        ...     },
        ...     quantile_fraction=0.1,  # Keep and exploit the top 10% of trials
        ... )

    Note:
        - Grid search spaces in ``hyperparam_mutations`` are automatically converted to
          sampling functions that cycle through the provided values.
        - When the time attr is the default ``"current_step"`` the ``perturbation_interval`` should be divisible by all
          batch_size that appear in the search space to not overstep perturbation points.

    See Also:
        :class:`ray.tune.schedulers.pbt.PopulationBasedTraining`: Base PBT scheduler
        :func:`_grid_search_sample_function`: Grid search sampling utilities
    """

    additional_config_keys = (
        FORK_FROM,
        "_top_pbt_is_in_upper_quantile",
        "_top_pbt_perturbed",
    )
    """Keys inserted into the config of trials to track PBT state."""

    def __init__(
        self,
        *,
        time_attr: str = "current_step",
        metric: str | None = None,
        mode: str | None = "max",
        perturbation_interval: float = 100_000,
        burn_in_period: float = 0,
        hyperparam_mutations: Optional[
            MutableMapping[str, dict[str, Any] | list | tuple | Callable[..., Any] | Domain]
        ] = None,
        # Use only very best trial # TODO: Should probably use more but double trials.
        quantile_fraction: float = 0.1,  # 0.25,  # 0 for no exploit -> no top trials, 0.99 for only exploit top trial
        resample_probability: float = 1.0,  # Always resample, assume grid_search in hyperparam_mutations # TODO: alt use custom_explore_fn with new value as input
        perturbation_factors: Tuple[float, float] = (0.8, 1.2),
        custom_explore_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        log_config: bool = True,
        require_attrs: bool = True,
        synch: bool = False,
        reseed: bool = True,
        num_samples: int = 1,
        prune_late_trials: bool = False,
    ):
        # if not hyperparam_mutations and custom_explore_fn is None:
        #    # Use a dummy function to log the perturbed hyperparams
        #    custom_explore_fn = _dummy_pass_through
        if custom_explore_fn is None and hyperparam_mutations:  # Otherwise use a wrapper
            custom_explore_fn = partial(_debug_dump_new_config, mutation_keys=list(hyperparam_mutations.keys()))
        elif custom_explore_fn is not None and hyperparam_mutations:
            custom_explore_fn = wrap_custom_perturbed_hparams(
                custom_explore_fn, mutation_keys=list(hyperparam_mutations.keys())
            )
        self._trial_initial_seeds: dict[Trial | None, int | None] = {None: None}
        self._reseed = reseed
        if reseed:
            custom_explore_fn = _ReseedEnv(wrap=custom_explore_fn, initial_seeds=self._trial_initial_seeds)
        self._trial_state: dict[Trial, _PBTTrialState2]  # pyright: ignore[reportIncompatibleVariableOverride]
        if hyperparam_mutations:  # either hyperparam_mutations or custom_explore_fn must be passed
            for k, v in hyperparam_mutations.items():
                if isinstance(v, dict) and "grid_search" in v:
                    hyperparam_mutations[k] = _grid_search_sample_function(v["grid_search"])
        super().__init__(
            time_attr,
            metric,
            mode,
            perturbation_interval,
            burn_in_period,
            hyperparam_mutations,  # pyright: ignore[reportArgumentType]
            quantile_fraction,
            resample_probability,
            perturbation_factors,
            custom_explore_fn,  # only used on explore (see _exploit function, get_new_config)
            log_config,
            require_attrs,
            synch,
        )
        # Store assignments for exploit distribution
        self._exploit_assignments = {}
        self._current_assignments: dict[Trial, Trial] | None = None
        self.current_trial_keys: dict[Trial, str] = {}
        """Currently assigned fork ids for each trial."""
        self._fork_ids: dict[tuple[Trial, tuple[Trial, int] | None], tuple[str, str] | str] = {}
        """
        Lookup for fork ids based on key=(trial, (parent_trial, parent_training_iteration) | None).
        The second the second element of the key is None for the initial trial (no fork).
        Key maps to (fork_id, parent_fork_id).
        """

        self._fork_time_data: dict[
            tuple[Trial, tuple[Trial, int] | None],
            dict[Literal["child", "parent"], tuple[ForktimeTuple, ForktimeTuple]],
        ] = {}
        """
        Lookup for fork time data based on key=(trial, (parent_trial, parent_training_iteration) | None).
        The second the second element of the key is None for the initial trial (no fork).

        The value is a dict with keys "child" and "parent", each mapping to a tuple of Forktime:
        The first Forktime is the ``time_attr`` and the second is the "current_step" Forktime.
        """

        self._fork_data_file: Path | None = None

        self._num_samples = num_samples
        """Number of samples from the parameter space."""

        self._seen_config_hashes: set[int] = set()

        self.prune_late_trials = prune_late_trials
        """Whether to prune trials that are slow and perform bad"""

    @classmethod
    def _deep_update_mutation(
        cls,
        mutations: dict[str, CyclicMutation[_T] | KeepMutation[_T] | dict[str, Any] | Any],
        new_skip: Optional[dict[str, Container[_T] | dict[str, Any] | None]] = None,
        keep_value: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update mutation parameters for both CyclicMutation and KeepMutation.

        Args:
            mutations: Dictionary of mutations to update
            new_skip: Optional dictionary with skip values for CyclicMutation instances
            keep_value: Optional dictionary with values to set for KeepMutation instances
        """
        # Unified iteration through mutations
        for key, distribution in mutations.items():
            # Process nested dictionaries
            if isinstance(distribution, dict):
                # Prepare deeper parameters for recursion
                deeper_skip = None
                deeper_keep = None

                if new_skip is not None and key in new_skip and isinstance(new_skip[key], dict):
                    deeper_skip = cast("dict[str, Any]", new_skip[key])

                if keep_value is not None and key in keep_value and isinstance(keep_value[key], dict):
                    deeper_keep = keep_value[key]

                # Recurse if we have anything to process deeper
                if deeper_skip is not None or deeper_keep is not None:
                    cls._deep_update_mutation(
                        distribution,
                        new_skip=deeper_skip,
                        keep_value=deeper_keep,
                    )

            # Update CyclicMutation
            elif isinstance(distribution, CyclicMutation) and new_skip is not None and key in new_skip:
                deeper_skip = new_skip[key]
                # Skip values should not be dictionaries
                assert not isinstance(deeper_skip, dict)
                distribution.update_skip(deeper_skip)

            # Update KeepMutation
            elif isinstance(distribution, KeepMutation) and keep_value is not None and key in keep_value:
                deeper_value = keep_value[key]
                # Keep values for leaf mutations should not be dictionaries
                assert not isinstance(deeper_value, dict)
                distribution.set_value(cast("_T", deeper_value))

    @overload
    def get_fork_ids(self, trial: Trial, parent: None = None, step: None = None) -> tuple[str, None]: ...

    @overload
    def get_fork_ids(self, trial: Trial, parent: Trial, step: int) -> tuple[str, str]: ...

    def get_fork_ids(
        self, trial: Trial, parent: Trial | None = None, step: Optional[int] = None
    ) -> tuple[str, str] | tuple[str, None]:
        if parent is None:
            return cast("str", self._fork_ids[trial, None]), None
        assert step is not None
        ret = self._fork_ids[trial, (parent, step)]
        if isinstance(ret, str):
            logger.error("Inconsistent fork id entry for trial %s, parent %s at step %s: %s", trial, parent, step, ret)
            return ret, None
        return ret

    def _reseed_seen_configs(self, config: dict, trial: Trial | None) -> bool | None:
        """
        If the given config has been seen before (based on its hash), modifies the "env_seed" in the config
        By appending a counter to ensure uniqueness, the env_seed is changed to a tuple of (original_seed, counter).

        If there is no "env_seed" in the config, or if it is None, no changes are made.

        Returns:
            True if the config was modified (i.e., reseeded), False if the config was already unique,
            None if no reseeding was performed due to missing "env_seed" key or its value being None.
        """
        frozen_config = deep_freeze(config)
        counter = 0
        hash_key = hash(frozen_config)
        initial_seed = self._trial_initial_seeds.get(trial, None) or config.get("env_seed")
        if initial_seed is None:
            return None
        while hash_key in self._seen_config_hashes:
            # Same config but different checkpoints should be covered via fork_from
            counter += 1
            config["env_seed"] = (initial_seed, counter)
            frozen_config = deep_freeze(config)
            hash_key = hash(frozen_config)
        if counter > 0:
            logger.info(
                "Adjusted env_seed for trial %s to avoid duplicate config after %s attempts. New env_seed: %s",
                trial or "",
                counter,
                config["env_seed"],
            )
        self._seen_config_hashes.add(hash_key)
        return counter > 0

    def on_trial_add(self, tune_controller: TuneController, trial: Trial):
        """Called when a new trial is added to the Tuner.

        Updates the trials config based on :attr:`hyperparam_mutations`.
        """
        # NOTE: On restore we might add a trial that is already at the perturbation point and should stay paused
        super().on_trial_add(tune_controller, trial)
        if self._unpickled and trial.last_result:
            last_time = trial.last_result[self._time_attr]
            # self._next_perturbation_sync is likely the inital value
            # NOTE: Should also be able to find the next perturbation interval with parent_time in fork_data
            if last_time % self._perturbation_interval == 0:
                # NOTE: This only works if no overstepping happens during training
                # NOTE 2: Scheduling a trial pause causes error - cannot do that - might leave us with only PENDING in choose_trial_to_run
                # tune_controller.pause_trial(trial, should_checkpoint=False)
                if last_time > self._next_perturbation_sync:
                    # Make them equal. In case all are pause, choose_trial_to_run will handle that case
                    self._next_perturbation_sync = last_time
                    logger.info(
                        "Updated _next_perturbation_sync to %s after adding trial %s with last_time %s",
                        self._next_perturbation_sync,
                        trial,
                        last_time,
                    )
            elif last_time > self._next_perturbation_sync:
                # set to last multiple BELOW of this trial - might allow some stragglers to catch up
                # In case all are PENDING chose_trial_to_run has to handle it
                self._next_perturbation_sync = (last_time // self._perturbation_interval) * self._perturbation_interval
                logger.info(
                    "Updated _next_perturbation_sync to %s after adding trial %s with last_time %s",
                    self._next_perturbation_sync,
                    trial,
                    last_time,
                )
            flat_results = flatten_dict(trial.last_result)
            if self.metric not in flat_results:
                logger.warning(
                    "Trial %s added with last_result %s but metric %s not found - cannot record last score.",
                    trial,
                    trial.last_result,
                    self.metric,
                )
                flat_results[self.metric] = float("nan")
                self._save_trial_state(self._trial_state[trial], last_time, flat_results, trial)
                self._trial_state[trial].last_score = None  # avoid nan sorting bug
            else:
                self._save_trial_state(self._trial_state[trial], last_time, flat_results, trial)
        # Check minibatch_size constraint
        if "minibatch_size" in trial.config:
            minibatch_size = trial.config["minibatch_size"]
            batch_size = trial.config.get("train_batch_size_per_learner", None)
            if batch_size is not None and minibatch_size > batch_size:
                # Should resample
                pass
            else:
                batch_size = trial.config.get(
                    "train_batch_size_per_learner",
                    trial.config.get("cli_args", {}).get("train_batch_size_per_learner", float("inf")),
                )
                if minibatch_size > batch_size:
                    # resample minibatch_size only
                    count = 0
                    while minibatch_size > batch_size and count < 40:
                        search_space = self._hyperparam_mutations["minibatch_size"]
                        # if isinstance(search_space, KeepMutation):
                        #    search_space.set_value(trial.config["minibatch_size"])
                        _fill_config(trial.config, "minibatch_size", search_space)
                        minibatch_size = trial.config["minibatch_size"]
                        count += 1
                    if count >= 40:
                        trial.set_status(trial.ERROR)
                        logger.error(
                            "Could not sample valid minibatch_size <= train_batch_size_per_learner "
                            "after 40 attempts. Using minibatch_size=%s, train_batch_size_per_learner=%s ."
                            "Trial might crash. Search space: %s",
                            minibatch_size,
                            batch_size,
                            self._hyperparam_mutations["minibatch_size"],
                        )

        if self._fork_data_file is None:
            self._fork_data_file = Path(trial.local_experiment_path) / f"pbt_fork_data-{get_run_id()}.csv"
            # if we restore it might already exist, only write header if not existing
            # NOTE: That when we restore the creation of the new local file can overwrite the old one!
            if not self._fork_data_file.exists():
                fork_data_file_remote = Path(trial.remote_experiment_path) / f"pbt_fork_data-{get_run_id()}.csv"
                self._fork_data_file.parent.mkdir(parents=True, exist_ok=True)
                if fork_data_file_remote.exists():
                    # copy to local location, tune will copy to remote again!
                    # First create a copy of the parent in the remote location before copying it locally
                    shutil.copyfile(fork_data_file_remote, fork_data_file_remote.with_suffix(".copy"))
                    shutil.copyfile(fork_data_file_remote, self._fork_data_file)
                else:
                    with self._fork_data_file.open("w") as f:
                        f.write(make_fork_from_csv_header())

        if FORK_FROM in trial.config:
            fork_config: ForkFromData = trial.config[FORK_FROM]
            logger.info("Adding a forked trial %s with config: %s", trial, fork_config)
            self._trial_state[trial].last_training_iteration = fork_config.get("parent_training_iteration", 0)
            self._trial_state[trial].current_env_steps = fork_config.get("parent_env_steps", None)
            # NOTE: its both unsave to use parent_trial_id or None as a fallback
            self.current_trial_keys[trial] = make_experiment_key(trial, fork_config)
            self._fork_ids[trial, None] = fork_config.get("fork_id_this_trial", trial.trial_id)
        else:
            self._trial_state[trial].last_training_iteration = 0
            self._trial_state[trial].current_env_steps = 0
            self.current_trial_keys[trial] = trial.trial_id
            self._fork_ids[trial, None] = trial.trial_id  # initial fork id is trial id
        self._trial_state[trial].last_update_timestamp = get_time()
        self._trial_state[trial].total_time_spent = 0.0  # use restored value from fork instead of 0?

        # When using more than 1 sample and seeding_strategy="same" we end up with identical configs
        # Need to change something in the config to make them different
        # TODO: Possibly avoid when using "constant" as seeding strategy

        # Convert all subdicts to frozendict and lists to tuples for hashing
        self._trial_initial_seeds[trial] = trial.config.get("env_seed", None)
        self._reseed_seen_configs(trial.config, trial)

    def _quantiles(self) -> Tuple[List[Trial], List[Trial]]:
        """Returns trials in the lower and upper `quantile` of the population.

        If there is not enough data to compute this, returns empty lists.

        This method allows for quantile_fraction > 0.5 as well.
        All trials outside the top quantile are considered in the lower quantile,
        meaning they will exploit the top-performing trials.
        """
        trials: list[Trial] = []
        for trial, state in self._trial_state.items():
            logger.debug("Trial %s, state %s", trial, state)
            if trial.is_finished():
                logger.debug("Trial %s is finished", trial)
            if state.last_score is not None and not math.isnan(state.last_score) and not trial.is_finished():
                trials.append(trial)

        # Sort trials by score; _save_trial_state takes care of mode (multiply by -1 if min mode)
        trials.sort(key=lambda t: self._trial_state[t].last_score)  # pyright: ignore[reportArgumentType]

        if len(trials) <= 1:
            return [], []

        # Calculate number of trials in top quantile
        num_top_trials = max(1, math.ceil(len(trials) * self._quantile_fraction))

        if num_top_trials > len(trials) / 2:
            num_top_trials = math.floor(len(trials) / 2)
        top_trials = trials[-num_top_trials:]
        # all other trials will exploit top trials
        bottom_trials = trials[:-num_top_trials]

        logger.debug("Split trials: %s in top quantile, %s in bottom quantile", len(top_trials), len(bottom_trials))

        return bottom_trials, top_trials

    def _distribute_exploitation_repeated(
        self, lower_quantile: list[Trial], upper_quantile: list[Trial]
    ) -> dict[Trial, Trial]:
        """
        Distribute exploitation assignments so that each top trial is never assigned more than once to the same config,
        unless the number of assignments exceeds the number of top trials.

        This approach groups lower trials by their config. For each group,
        top trials are assigned in a round-robin fashion, ensuring that within a group, no top trial is assigned
        more than once unless necessary. Assignments are rotated across groups to balance usage.

        Args:
            lower_quantile: List of trials that will exploit top trials.
            upper_quantile: List of top-performing trials to be exploited.

        Returns:
            Dictionary mapping each bottom trial to the top trial it should exploit.
        """
        if not upper_quantile or not lower_quantile:
            return {}

        assignments: dict[Trial, Trial] = {}

        # Group lower trials by config
        config_to_trials: dict[frozendict, list[Trial]] = {}
        for trial in lower_quantile:
            config = trial.config.copy()
            config.pop(FORK_FROM, None)  # Exclude FORK_FROM from grouping
            config.pop("env_seed", None)  # Exclude env_seed from grouping
            frozen = deep_freeze(config)
            config_to_trials.setdefault(frozen, []).append(trial)

        num_top = len(upper_quantile)
        group_list = list(config_to_trials.values())

        # Track usage count for each top trial
        top_usage = dict.fromkeys(upper_quantile, 0)

        # For each group, rotate the starting index so assignments are balanced across groups
        for group_idx, group_trials in enumerate(group_list):
            # Rotate starting index for each group to balance assignments
            start_idx = group_idx % num_top
            used = set()
            for idx, trial in enumerate(group_trials):
                # Try to assign each top trial once per group, rotating start
                top_idx = (start_idx + idx) % num_top
                top_trial = upper_quantile[top_idx]
                # Avoid repeats if possible

                while top_trial in used and len(used) < num_top:
                    top_idx = (top_idx + 1) % num_top
                    top_trial = upper_quantile[top_idx]
                used.add(top_trial)
                assignments[trial] = top_trial
                top_usage[top_trial] += 1

        # Assign any remaining trials not in a group (shouldn't happen, but for safety)
        assigned_trials = set(assignments)
        top_trials_cycle = itertools.cycle(upper_quantile)
        for trial in lower_quantile:
            if trial not in assigned_trials:
                assignments[trial] = next(top_trials_cycle)

        # Log the distribution
        distribution = {trial.trial_id: top_usage.get(trial, 0) for trial in upper_quantile}
        logger.debug("Exploitation distribution: %s", distribution)
        assert len(assignments) == len(lower_quantile)
        return assignments

    def _distribute_exploitation(self, lower_quantile: List[Trial], upper_quantile: List[Trial]) -> Dict[Trial, Trial]:
        """Distribute the exploitation of top trials evenly among bottom trials.

        Args:
            lower_quantile: List of trials that will exploit top trials
            upper_quantile: List of top-performing trials to be exploited

        Returns:
            Dictionary mapping each bottom trial to the top trial it should exploit
        """
        if not upper_quantile or not lower_quantile:
            return {}

        assignments: dict[Trial, Trial] = {}
        # Create cyclic assignment to ensure even distribution of upper trials
        # When having num_samples > do this in a shifting way
        # and check that the perturbed trial has (hopefully) not the same config as the top trial (identical graph)
        # NOTE: When using num_samples the first steps to first perturbation interval are duplicated
        if self._num_samples > 1:
            assignments = self._distribute_exploitation_repeated(lower_quantile, upper_quantile)
        else:
            top_trials_cycle = itertools.cycle(upper_quantile)
            for trial in lower_quantile:
                assignments[trial] = next(top_trials_cycle)

        # Log the distribution
        distribution = {trial.trial_id: 1 for trial in upper_quantile}
        for top in assignments.values():
            distribution[top.trial_id] += 1
            assert top in upper_quantile
        logger.debug("Exploitation distribution: %s", distribution)
        return assignments

    def _get_current_best_mutations(self, upper_quantile: list[Trial]):
        """
        Get the values of hyperparameter mutations for the best trials.

        Use to update the skip lists in CyclicMutation instances.
        """
        # Get all paths to leaves in the mutation space

        flat_mutation_keys = flatten_mapping_with_path(self._hyperparam_mutations)
        mutation_paths = [path for path, _ in flat_mutation_keys]

        # Collect values for each path from all trial configs
        path_to_values: dict[tuple, list] = {path: [] for path in mutation_paths}
        for trial in upper_quantile:
            config = trial.config
            for path in mutation_paths:
                try:
                    value = get_value_by_path(config, path)
                except KeyError:
                    value = None
                path_to_values[path].append(value)

        return build_nested_dict(mutation_paths, path_to_values)

    def _checkpoint_or_exploit(
        self,
        trial: Trial,
        tune_controller: "TuneController",
        upper_quantile: List[Trial],
        lower_quantile: List[Trial],
    ):
        """Checkpoint if in upper quantile, exploits if in lower.

        For trials in the lower quantile, evenly distribute which top trial
        they exploit to ensure balanced exploitation.
        """
        # Note, is iterated in order: upper_quantile, not in quantiles, lower_quantile
        if trial.storage and trial.storage and "_ray_pkg_" in trial.storage.storage_fs_path:
            logger.error("Trial %s has storage path in _ray_pkg_: %s", trial, trial.storage.storage_fs_path)
        state = self._trial_state[trial]
        # Remove any fork controlling keys from the config
        for k in self.additional_config_keys:
            trial.config.pop(k, None)

        # Create exploitation assignments if needed
        self._current_assignments = self._distribute_exploitation(lower_quantile, upper_quantile)
        # Update any CyclicMutation skip lists based on current top trials, to not resample these values.
        new_skips = self._get_current_best_mutations(upper_quantile)
        logger.debug("Updating CyclicMutation skip lists to %s", new_skips)

        flat_mutations = tree.flatten_with_path(self._hyperparam_mutations)
        keep_mutations = [(path, m) for path, m in flat_mutations if isinstance(m, KeepMutation)]
        for path, mutation in keep_mutations:
            value = KeepMutation.get_config_value(trial.config, path)
            mutation.set_value(value)
        self._deep_update_mutation(self._hyperparam_mutations, new_skip=new_skips)

        # Keep checkpoints for all trials:
        # NOTE: NEEDS the _exploit wrapper to pause the trial, otherwise this causes the last trial to be terminated
        # if it is in the lower quantile,see https://github.com/ray-project/ray/issues/57906
        if SAVE_ALL_CHECKPOINTS:
            logger.debug("Instructing %s to save.", trial)
            checkpoint = tune_controller._schedule_trial_save(trial, result=state.last_result)

        # Only keep checkpoints for top trials
        if trial in upper_quantile:
            # TODO: check this again in TuneControl the trial last result is only updated after the scheduler
            # callback. So, we override with the current result.
            logger.debug("Trial %s is in upper quantile. Saving checkpoint.", trial)
            if trial.status == Trial.PAUSED:
                if trial.temporary_state.saving_to and isinstance(
                    trial.temporary_state.saving_to, _FutureTrainingResult
                ):
                    logger.debug("Trial %s is still saving.", trial)
                    state.last_checkpoint = trial.temporary_state.saving_to
                else:
                    # Paused trial will always have an in-memory checkpoint.
                    logger.debug("Trial %s is paused. Use last available checkpoint %s.", trial, trial.checkpoint)
                    state.last_checkpoint = trial.checkpoint
            else:
                logger.debug("Keeping checkpoint of trial %s for exploit.", trial)

                # TODO: possible # FIXME does this create two checkpoint with Trainable Auto saving?
                if SAVE_ALL_CHECKPOINTS:
                    state.last_checkpoint = checkpoint  # pyright: ignore[reportPossiblyUnboundVariable]
                else:
                    state.last_checkpoint = tune_controller._schedule_trial_save(trial, result=state.last_result)

            self._num_checkpoints += 1
            trial.config["_top_pbt_is_in_upper_quantile"] = True
            trial.config["__pbt_main_branch__"] = True
        else:
            state.last_checkpoint = None  # not a top trial

        # If in lower quantile, exploit a top trial based on our distribution
        if trial in lower_quantile:
            # Get the assigned top trial to exploit
            trial_to_clone = self._current_assignments.get(trial)

            if not trial_to_clone:
                logger.warning("No exploitation assignment for trial %s. Using random selection.", trial)
                trial_to_clone = random.choice(upper_quantile)

            # Minibatch <= train_bach_size constraint
            if "minibatch_size" in self._hyperparam_mutations and isinstance(
                self._hyperparam_mutations["minibatch_size"], KeepMutation
            ):
                train_batch_size_per_learner = trial_to_clone.config.get(
                    "train_batch_size_per_learner",
                    trial_to_clone.config.get("cli_args", {}).get("train_batch_size_per_learner", float("inf")),
                )
                if trial.config["minibatch_size"] > train_batch_size_per_learner:
                    # Cannot keep Mutation minibatch size value and satisfy the constraint, do not exploit this trial.
                    trial.config["_top_pbt_perturbed"] = False
                    # Add current env step to seed data
                    if self._reseed and trial.config.get("env_seed") is not None:
                        # First _ReseedEnv can change seed to (initial, current_step)
                        reseeder = _ReseedEnv(
                            add_seed=self._trial_state[trial].current_env_steps,
                            next_trial=trial,
                            initial_seeds=self._trial_initial_seeds,
                        )
                        trial_config_reseeded = reseeder(trial.config)
                        # If config type was already seen reseed to ((initial, current_step), counter)
                        if self._reseed_seen_configs(trial_config_reseeded, trial):
                            trial.set_config(trial_config_reseeded)
                    logger.info(
                        "Trial %s cannot exploit trial %s due to minibatch size constraint. Skipping exploit. "
                        "minibatch_size=%s, train_batch_size_per_learner=%s",
                        trial,
                        trial_to_clone,
                        trial.config["minibatch_size"],
                        train_batch_size_per_learner,
                    )
                    return

            assert trial is not trial_to_clone
            assert trial_to_clone in upper_quantile
            clone_state = self._trial_state[trial_to_clone]
            last_checkpoint = clone_state.last_checkpoint

            logger.debug(
                "Trial %s is exploiting trial %s (rank %s/%s).",
                trial,
                trial_to_clone,
                upper_quantile.index(trial_to_clone) + 1,
                len(upper_quantile),
            )

            if isinstance(last_checkpoint, _FutureTrainingResult):
                training_result: _TrainingResult | None = last_checkpoint.resolve()

                if training_result:
                    clone_state.last_result = training_result.metrics
                    clone_state.last_checkpoint = training_result.checkpoint
                    last_checkpoint = clone_state.last_checkpoint
                else:
                    logger.error(
                        "PBT-scheduled checkpoint save resolved to None. Trial "
                        "%s didn't save any checkpoint before "
                        "and can't be exploited.",
                        trial_to_clone,
                    )
                    last_checkpoint = None

            if not last_checkpoint:
                logger.warning("[pbt]: no checkpoint for trial %s. Skip exploit for Trial %s", trial_to_clone, trial)
                return
            # Add current env step to seed data
            if self._reseed:
                if not isinstance(self._custom_explore_fn, _ReseedEnv):
                    logger.warning("Custom explore function is not wrapped with _ReseedEnv, reseed will not work.")
                else:
                    self._custom_explore_fn.set_next_trial(trial)
                    self._custom_explore_fn.add_seed = self._trial_state[trial_to_clone].current_env_steps
            # HACK: To save a checkpoint AND not terminate the actor we need to fake this a bit:
            if SAVE_ALL_CHECKPOINTS:
                trial_status = trial.status
                if trial_status != Trial.PAUSED:
                    trial.status = Trial.PAUSED  # fake to keep actor alive
                self._exploit(tune_controller, trial, trial_to_clone)
                if trial_status != Trial.PAUSED:
                    trial.status = trial_status  # reset
            else:
                self._exploit(tune_controller, trial, trial_to_clone)
            if self._reseed_seen_configs(trial.config, trial):
                trial.set_config(trial.config)
            # Mark trial as perturbed
            for k in self.additional_config_keys:
                trial.config.pop(k, None)
            trial.config.pop("__pbt_main_branch__", None)
            # Set info which trial was forked from
            parent_iteration = self._trial_state[trial_to_clone].last_training_iteration
            fork_data: ForkFromData = {
                "parent_trial_id": trial_to_clone.trial_id,  # NOTE: This is the constant Trial.trial_id
                "parent_trial": trial_to_clone,
                "parent_training_iteration": parent_iteration,
                "parent_time": Forktime(self._time_attr, self._trial_state[trial_to_clone].last_train_time),
                "controller": self.__class__.__name__,
            }
            forked_trial_id = make_experiment_key(trial, fork_data)
            fork_data["fork_id_this_trial"] = forked_trial_id
            if (current_env_steps := self._trial_state[trial_to_clone].current_env_steps) is not None:
                fork_data["parent_env_steps"] = current_env_steps
            # XXX: Does this reflect the correct parent fork id?
            # trial to clone is is in upper_quantile, meaning self.current_trial_keys is not updated
            # for the parent, as it will continue this is correct
            # Q: Is the parent running with this id in the logger?
            fork_data["parent_fork_id"] = self.current_trial_keys[trial_to_clone]
            trial.config[FORK_FROM] = fork_data
            trial.config["experiment_key"] = forked_trial_id
            trial.config.setdefault("original_experiment_key", trial.config["experiment_key"])
            trial.invalidate_json_state()
            # Update variables tracking the fork ids
            self._fork_ids[trial, (trial_to_clone, parent_iteration)] = forked_trial_id
            self._fork_time_data[trial, (trial_to_clone, parent_iteration)] = {
                "child": (
                    Forktime(self._time_attr, state.last_train_time),
                    Forktime(
                        "current_step", state.current_env_steps if state.current_env_steps is not None else float("nan")
                    ),
                ),
                "parent": (
                    fork_data["parent_time"],
                    Forktime("current_step", current_env_steps if current_env_steps is not None else float("nan")),
                ),
            }
            self.current_trial_keys[trial] = forked_trial_id
            assert self._fork_data_file
            with self._fork_data_file.open("a") as f:
                f.write(
                    self._write_fork_data_csv_line(
                        trial,
                        (trial_to_clone, parent_iteration),
                        forked_trial_id,
                        parent_fork_id=fork_data["parent_fork_id"],
                    )
                )
            if tune_controller._queued_trial_decisions.get(trial.trial_id, None) == self.CONTINUE:
                # WORKAROUND for https://github.com/ray-project/ray/issues/58483. Prevent KeyError during buffered training
                tune_controller._queued_trial_decisions.pop(trial.trial_id)
        else:
            trial.config["_top_pbt_perturbed"] = False
            # Add current env step to seed data
            if self._reseed and trial.config.get("env_seed") is not None:
                # First _ReseedEnv can change seed to (initial, current_step)
                reseeder = _ReseedEnv(
                    add_seed=self._trial_state[trial].current_env_steps,
                    next_trial=trial,
                    initial_seeds=self._trial_initial_seeds,
                )
                trial_config_reseeded = reseeder(trial.config)
                # If config type was already seen reseed to ((initial, current_step), counter)
                if self._reseed_seen_configs(trial_config_reseeded, trial):
                    trial.set_config(trial_config_reseeded)

    def _save_trial_state(
        self,
        state: _PBTTrialState | _PBTTrialState2,
        time: int,
        result: AlgorithmReturnData | dict,
        trial: Trial,
        *,
        update_timestamp: bool = True,
    ):
        """Save trial state, optionally updating the wall-clock timestamp.

        Args:
            state: Trial state object to update.
            time: Training time value from time_attr.
            result: Training result dictionary.
            trial: The trial being updated.
            update_timestamp: If True, updates last_update_timestamp to current wall-clock time.
                Set to False when saving state during pause/perturbation bookkeeping to avoid
                interfering with slow trial detection logic.
        """
        score = super()._save_trial_state(state, time, cast("dict", result), trial)
        # Save training iteration for the step for loggers like WandB / Comet
        state.last_training_iteration = result[TRAINING_ITERATION]  # pyright: ignore[reportAttributeAccessIssue]
        try:
            state.current_env_steps = get_current_step(result)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        except KeyError:
            state.current_env_steps = None  # pyright: ignore[reportAttributeAccessIssue]
        if update_timestamp:
            state.last_update_timestamp = get_time()  # pyright: ignore[reportAttributeAccessIssue]
        state.total_time_spent = result.get(TIME_TOTAL_S, 0.0)  # pyright: ignore[reportAttributeAccessIssue]
        return score

    def reset_exploitations(self):
        """Reset the current exploitation assignments.

        This should be called at the beginning of each perturbation round.
        """
        self._current_assignments = None

    @overload
    def _write_fork_data_csv_line(
        self, trial: Trial, parent_data: None, fork_id: str, *, parent_fork_id: str | None = None
    ) -> str: ...

    @overload
    def _write_fork_data_csv_line(
        self, trial: Trial, parent_data: tuple[Trial, int], fork_id: str, *, parent_fork_id: str
    ) -> str: ...

    def _write_fork_data_csv_line(
        self,
        trial: Trial,
        parent_data: tuple[Trial, int] | None,
        fork_id: str,
        *,
        parent_fork_id: Optional[str] = None,
    ) -> str:
        if parent_data is None:
            parent_fork_id = None
            parent_step = None
            parent_trial = None
            return ""
        assert parent_fork_id is not None
        parent_trial, parent_step = parent_data
        parent_time = self._fork_time_data[trial, parent_data]["parent"][1]
        assert parent_time[0] == "current_step"
        fork_data: ForkFromData = {
            "parent_trial_id": parent_trial.trial_id,
            "parent_trial": parent_trial,
            "parent_fork_id": parent_fork_id,
            "parent_training_iteration": parent_step,
            "parent_time": parent_time,
            "fork_id_this_trial": fork_id,
            "controller": self.__class__.__name__,
        }
        return make_fork_from_csv_line(fork_data)

    def dump_fork_data(self) -> str:
        """Format the fork data as CSV string."""
        contents = make_fork_from_csv_header()
        for (trial, parent_data), (fork_id, parent_fork_id) in self._fork_ids.items():
            contents += self._write_fork_data_csv_line(trial, parent_data, fork_id, parent_fork_id=parent_fork_id)
        return contents

    @warn_if_slow
    def _perturbation_sync_mode(self, tune_controller: TuneController, time: int):
        # Copied from PopulationBasedTraining
        logger.info("PBT: Starting perturbation")
        lower_quantile, upper_quantile = self._quantiles()
        all_trials = tune_controller.get_trials()
        not_in_quantile = [t for t in all_trials if t not in lower_quantile and t not in upper_quantile]

        # Move upper quantile trials to beginning and lower quantile
        # to end. This ensures that checkpointing of strong trials
        # occurs before exploiting of weaker ones.
        all_trials = upper_quantile + not_in_quantile + lower_quantile
        for t in all_trials:
            self._trial_state[t].last_perturbation_time = time
            self._checkpoint_or_exploit(t, tune_controller, upper_quantile, lower_quantile)

        all_train_times = [self._trial_state[t].last_train_time for t in tune_controller.get_trials()]
        max_last_train_time = max(all_train_times)
        self._next_perturbation_sync = max(
            self._next_perturbation_sync + self._perturbation_interval,
            max_last_train_time,
        )
        logger.debug("Next perturb at time %s", self._next_perturbation_sync)

    @warn_if_slow
    def on_trial_result(self, tune_controller: TuneController, trial: Trial, result: dict) -> str:
        # TODO: Can buffered training affect this negatively?
        decision = super().on_trial_result(tune_controller, trial, result)
        if decision != self.CONTINUE:
            return decision
        # do not wait for a slow and bad last trial in synch mode
        if not self._synch:
            return decision
        time_of_slow_interval = result[self._time_attr]
        if time_of_slow_interval < self._burn_in_period:
            return decision
        state = self._trial_state[trial]
        time_since_perturb = time_of_slow_interval - state.last_perturbation_time
        # if it is too early or too late, do nothing, except if it is really slow
        other_total_times = [self._trial_state[t].total_time_spent for t in self._trial_state if t is not trial]
        trial_total_time = result.get(TIME_TOTAL_S, 0)
        max_other_time = max(other_total_times)
        if (
            time_since_perturb
            # for every 5min difference allow 1% earlier termination.
            < min(0.5, max(0.2, (0.50 - 0.01 * max(0, ((trial_total_time - max_other_time) // 300)))))
            * self._perturbation_interval
            or time_since_perturb > 0.95 * self._perturbation_interval
        ):
            return decision
        # If this trial was just started late and is not slow by itself continue
        if other_total_times and (trial_total_time <= (max_other := max_other_time) - (min(max_other * 0.05, 180))):
            return decision
        # If we are less than 5 min behind other trials (excluding current), ignore
        # Exclude current trial from comparison to avoid self-comparison
        other_trial_timestamps = [
            self._trial_state[t].last_update_timestamp for t in self._trial_state if t is not trial
        ]
        if other_trial_timestamps and get_time() - max(other_trial_timestamps) < 300:
            return decision
        # Check this is in the 5% of not-yet finished trials
        if (
            still_active_trials := sum(
                self._trial_state[t].last_train_time < self._next_perturbation_sync and t != trial
                for t in tune_controller.get_live_trials()
            )
        ) > len(tune_controller.get_live_trials()) * 0.05:
            return decision
        # NOTE: The states are from the paused ahead trials and the *last* perturbation interval from the still running
        lowest_scores = [state.last_score for state in self._trial_state.values() if state.last_score is not None]
        if len(lowest_scores) == 0:
            return decision

        # We assume quantiles are sorted
        if result[self._metric] > lowest_scores[min(3, int(len(lowest_scores) * 0.33)) - 1]:
            return decision
        # last trial is slow and in worst 33%. Pause trial and if last start perturbation
        # When return of super() is CONTINUE we should not have had a perturbation this result.
        # Expect when a good trial is slow, but then we do not end up here.
        perturbation_time = max(state.last_train_time for state in self._trial_state.values())
        # Save state without updating timestamp to avoid interfering with other slow trials' detection
        self._save_trial_state(state, perturbation_time, result, trial, update_timestamp=False)
        # TODO If there are multiple slow do not perturb yet just pause the others.
        if still_active_trials == 0:
            logger.info("Last trial %s is a bad performing straggler. Starting early PBT perturbation.", trial)
            self._perturbation_sync_mode(tune_controller, perturbation_time)
        else:
            logger.info(
                "Last trial %s is a bad performing straggler. Pausing early and waiting for other slow trials", trial
            )
        return self.NOOP if trial.status == Trial.PAUSED else self.PAUSE

    def on_trial_complete(
        self, tune_controller: TuneController, trial: Trial, result: FlatLogMetricsDict | dict[str, Any]
    ):
        """Handle completed trial by cleaning up assignments."""
        # Happens on trial.Terminated, e.g. stopper returned True
        super().on_trial_complete(
            tune_controller,
            trial,
            cast("dict[str, object]", result),
        )
        # Reset assignments when trials complete to ensure proper redistribution
        self.reset_exploitations()

    # NOTE: Tune does NOT support get_state for schedulers yet, so this is custom

    def get_state(self) -> dict:
        """Get the state of the scheduler for checkpointing.

        Returns:
            Dictionary containing scheduler state. Trial objects are converted
            to trial IDs for serialization.
        """
        # TODO: possiblly use self.__dict__
        state = super().get_state() if hasattr(super(), "get_state") else {}  # pyright: ignore[reportAttributeAccessIssue]

        # Convert Trial keys to trial IDs for serialization
        state.update(
            {
                "trial_initial_seeds": {
                    (trial.trial_id if trial is not None else None): seed
                    for trial, seed in self._trial_initial_seeds.items()
                },
                "reseed": self._reseed,
                "current_trial_keys": {trial.trial_id: key for trial, key in self.current_trial_keys.items()},
                "fork_ids": {
                    (trial.trial_id, (parent.trial_id if parent else None, step) if parent_step else None): fork_ids
                    for (trial, parent_step), fork_ids in self._fork_ids.items()
                    for parent, step in ([parent_step] if parent_step else [(None, None)])
                },
                "fork_time_data": {
                    (trial.trial_id, (parent.trial_id if parent else None, step) if parent_step else None): time_data
                    for (trial, parent_step), time_data in self._fork_time_data.items()
                    for parent, step in ([parent_step] if parent_step else [(None, None)])
                },
                "fork_data_file": str(self._fork_data_file) if self._fork_data_file else None,
                "num_samples": self._num_samples,
                "seen_config_hashes": list(self._seen_config_hashes),
                "prune_late_trials": self.prune_late_trials,
            }
        )
        return state

    def set_state(self, state: dict) -> None:
        """Set the state of the scheduler from checkpoint data.

        Args:
            state: State dictionary containing scheduler state.

        Note:
            Trial objects cannot be restored from IDs alone. The restored state
            will use trial IDs as keys until trials are restarted and the mappings
            are rebuilt.
        """
        if hasattr(super(), "set_state"):
            super().set_state(state)  # pyright: ignore[reportAttributeAccessIssue]

        # Restore trial_initial_seeds with trial IDs as keys
        self._trial_initial_seeds = {
            None if trial_id is None else trial_id: seed  # type: ignore[misc]
            for trial_id, seed in state.get("trial_initial_seeds", {None: None}).items()
        }

        self._reseed = state.get("reseed", True)

        # Restore current_trial_keys with trial IDs
        self.current_trial_keys = {}
        for trial_id, key in state.get("current_trial_keys", {}).items():
            self.current_trial_keys[trial_id] = key

        # Restore fork_ids with trial IDs
        self._fork_ids = {}
        for (trial_id, parent_data), fork_ids in state.get("fork_ids", {}).items():
            self._fork_ids[(trial_id, parent_data)] = fork_ids

        # Restore fork_time_data with trial IDs
        self._fork_time_data = {}
        for (trial_id, parent_data), time_data in state.get("fork_time_data", {}).items():
            self._fork_time_data[(trial_id, parent_data)] = time_data

        # Restore fork_data_file
        fork_data_file_str = state.get("fork_data_file")
        self._fork_data_file = Path(fork_data_file_str) if fork_data_file_str else None

        self._num_samples = state.get("num_samples", 1)
        self._seen_config_hashes = set(state.get("seen_config_hashes", []))

        self.prune_late_trials = state.get("prune_late_trials", False)

        logger.info(
            "Restored TopPBTTrialScheduler state: %d trial seeds, %d fork ids, %d seen configs",
            len(self._trial_initial_seeds),
            len(self._fork_ids),
            len(self._seen_config_hashes),
        )
