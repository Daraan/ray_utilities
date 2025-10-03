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

import logging
import math
import random
from functools import partial
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable, Container, Dict, Generic, List, Optional, Tuple, TypeVar, cast

from ray.train._internal.session import _FutureTrainingResult, _TrainingResult
from ray.tune.experiment import Trial
from ray.tune.result import TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.schedulers.pbt import PopulationBasedTraining

from ray_utilities.constants import FORK_FROM, PERTURBED_HPARAMS
from ray_utilities.misc import build_nested_dict, flatten_mapping_with_path, get_value_by_path
from ray_utilities.typing import ForkFromData, Forktime

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping

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
            self.last_training_iteration: int  # not present before on_trial_add
            """The training iteration at which the last result was reported."""


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


class TopPBTTrialScheduler(PopulationBasedTraining):
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
        Grid search spaces in hyperparam_mutations are automatically converted to
        sampling functions that cycle through the provided values.

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
        custom_explore_fn: Callable[..., Any] | None = None,
        log_config: bool = True,
        require_attrs: bool = True,
        synch: bool = False,
    ):
        if custom_explore_fn is None:  # Otherwise use a wrapper
            # XXX
            if hyperparam_mutations:
                custom_explore_fn = partial(_debug_dump_new_config, mutation_keys=list(hyperparam_mutations.keys()))
        # TODO: Do we want to reload the env_seed?
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

    def on_trial_add(self, tune_controller: TuneController, trial: Trial):
        """Updates the trials config with hyperparam_mutations"""
        # Adds a new trial with config updated based on hyperparam_mutations
        super().on_trial_add(tune_controller, trial)
        if FORK_FROM in trial.config:
            self._trial_state[trial].last_training_iteration = trial.config[FORK_FROM].get("fork_training_iteration", 0)
        else:
            self._trial_state[trial].last_training_iteration = 0

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

        # Sort trials by score (in appropriate order based on mode)
        if self._mode == "max":
            trials.sort(key=lambda t: self._trial_state[t].last_score or 0.0, reverse=True)
        else:
            trials.sort(key=lambda t: self._trial_state[t].last_score or 0.0)

        if len(trials) <= 1:
            return [], []

        # Calculate number of trials in top quantile
        num_top_trials = max(1, math.ceil(len(trials) * self._quantile_fraction))

        # All other trials will exploit these top trials
        top_trials = trials[:num_top_trials]
        bottom_trials = trials[num_top_trials:]

        logger.debug("Split trials: %s in top quantile, %s in bottom quantile", len(top_trials), len(bottom_trials))

        return bottom_trials, top_trials

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
        # Create cyclic assignment to ensure even distribution
        top_trials_cycle = cycle(upper_quantile)

        for trial in lower_quantile:
            assignments[trial] = next(top_trials_cycle)

        # Log the distribution
        distribution = {trial.trial_id: 1 for trial in upper_quantile}
        for top in assignments.values():
            distribution[top.trial_id] += 1

        logger.debug("Exploitation distribution: %s", distribution)
        return assignments

    @classmethod
    def _deep_update_cyclic_mutation(
        cls,
        mutations: dict[str, CyclicMutation[_T] | dict[str, Any] | Any],
        new_skip: Optional[dict[str, Container[_T] | dict[str, Any] | None]],
    ) -> None:
        if new_skip is None:
            return
        for key, distribution in mutations.items():
            if isinstance(distribution, dict) and key in new_skip:
                deeper_skip = new_skip[key]
                assert isinstance(deeper_skip, dict)
                cls._deep_update_cyclic_mutation(distribution, cast("dict[str, Any]", deeper_skip))
            elif isinstance(distribution, CyclicMutation):
                deeper_skip = new_skip[key]
                assert not isinstance(deeper_skip, dict)
                cast("CyclicMutation[_T]", distribution).update_skip(deeper_skip)

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
        state = self._trial_state[trial]
        # Remove any fork controlling keys from the config
        for k in self.additional_config_keys:
            trial.config.pop(k, None)

        # Create exploitation assignments if needed
        self._current_assignments = self._distribute_exploitation(lower_quantile, upper_quantile)
        # Update any CyclicMutation skip lists based on current top trials, to not resample these values.
        new_skips = self._get_current_best_mutations(upper_quantile)
        logger.debug("Updating CyclicMutation skip lists to %s", new_skips)
        self._deep_update_cyclic_mutation(self._hyperparam_mutations, new_skip=new_skips)
        # TODO: a trial should just use the batch size assigned that it already has, i.e. actually no mutation, just load best checkpoint
        # FIXME: next perturbation interval does not increase linearly. Trials train longer and longer as well

        # Create a checkpoint for all trials
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
                # TODO: # FIXME does this create two checkpoint with Trainable Auto saving?
                # state.last_checkpoint = tune_controller._schedule_trial_save(trial, result=state.last_result)
                state.last_checkpoint = checkpoint

            self._num_checkpoints += 1
            trial.config["_top_pbt_is_in_upper_quantile"] = True
        else:
            state.last_checkpoint = None  # not a top trial

        # If in lower quantile, exploit a top trial based on our distribution
        if trial in lower_quantile:
            # Get the assigned top trial to exploit
            trial_to_clone = self._current_assignments.get(trial)

            if not trial_to_clone:
                logger.warning("No exploitation assignment for trial %s. Using random selection.", trial)
                trial_to_clone = random.choice(upper_quantile)

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
                logger.info("[pbt]: no checkpoint for trial %s. Skip exploit for Trial %s", trial_to_clone, trial)
                return
            self._exploit(tune_controller, trial, trial_to_clone)
            # Mark trial as perturbed
            for k in self.additional_config_keys:
                trial.config.pop(k, None)
            # Set info which trial was forked from
            fork_data: ForkFromData = {
                "parent_id": trial_to_clone.trial_id,
                "parent_trial": trial_to_clone,
                "parent_training_iteration": self._trial_state[trial_to_clone].last_training_iteration,  # pyright: ignore[reportAttributeAccessIssue]
                "parent_time": Forktime(self._time_attr, self._trial_state[trial_to_clone].last_train_time),
                "controller": self.__class__.__name__,
            }
            trial.config[FORK_FROM] = fork_data
            trial.invalidate_json_state()
        else:
            for k in self.additional_config_keys:
                trial.config.pop(k, None)
            trial.config["_top_pbt_perturbed"] = False

    def _save_trial_state(
        self, state: _PBTTrialState | _PBTTrialState2, time: int, result: AlgorithmReturnData | dict, trial: Trial
    ):
        score = super()._save_trial_state(state, time, cast("dict", result), trial)
        # Save training iteration for the step for loggers like WandB / Comet
        state.last_training_iteration = result[TRAINING_ITERATION]  # pyright: ignore[reportAttributeAccessIssue]
        return score

    def reset_exploitations(self):
        """Reset the current exploitation assignments.

        This should be called at the beginning of each perturbation round.
        """
        self._current_assignments = None

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
