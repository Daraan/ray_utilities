from __future__ import annotations

import logging
import math
import random
from functools import partial
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeVar

from ray.train._internal.session import _FutureTrainingResult
from ray.tune.experiment import Trial
from ray.tune.schedulers.pbt import PopulationBasedTraining
from ray.util.debug import log_once

from ray_utilities.constants import PERTURBED_HPARAMS

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping

    from ray.tune.execution.tune_controller import TuneController
    from ray.tune.schedulers.pbt import _PBTTrialState
    from ray.tune.search.sample import Domain


logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _grid_search_sample_function(grid_search_space: Iterable[_T], *, repeat=True) -> Callable[[], _T]:
    """
    Returns:
        - A parameterless function that returns the next grid search sample.
          If repeat=False, the function will raise StopIteration when it returned each value once.
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


class ReTuneScheduler(PopulationBasedTraining):
    # TODO:
    # [ ] - New trials are spawned as long as search_alg is not finished. Need to keep search_alg running
    # Add Trial is triggered on TuneController.step -> SearchAlg creates Trial -> Scheduler.on_trial_add
    # Possibly use no searcher and limit result by num_samples

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
        quantile_fraction: float = 0.99,  # 0.25,  # 0 for no exploit -> no top trials, 0.99 for only exploit top trial
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
        self._trial_state: dict[Trial, _PBTTrialState]
        if quantile_fraction > 0.5:
            if quantile_fraction > 1:
                raise ValueError("quantile_fraction may not be larger than 1.0")
            quantile_fraction_after_init = quantile_fraction
            quantile_fraction = 0.5
        else:
            quantile_fraction_after_init = None
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
        if quantile_fraction_after_init is not None:
            # ray code does not allow _quantile_fraction > 0.5
            self._quantile_fraction = quantile_fraction_after_init

    def on_trial_add(self, tune_controller: TuneController, trial: Trial):
        """Updates the trials config with hyperparam_mutations"""
        # Adds a new trial with config updated based on hyperparam_mutations
        return super().on_trial_add(tune_controller, trial)

    def on_trial_resultX(self, tune_controller: TuneController, trial: Trial, result: Dict) -> str:
        self._check_result(result)
        if self._metric not in result or self._time_attr not in result:
            return self.CONTINUE  # todo ray use Enum
        time = result[self._time_attr]
        state = self._trial_state[trial]
        # Continue training if burn-in period has not been reached, yet.
        if time < self._burn_in_period:
            logger.debug("Still in burn-in period: %s < %s", time, self._burn_in_period)
            return self.CONTINUE

        # Continue training if perturbation interval has not been reached, yet.
        time_since_perturb = time - state.last_perturbation_time
        if time_since_perturb < self._perturbation_interval:
            logger.debug("Perturbation interval not reached: %s < %s", time_since_perturb, self._perturbation_interval)
            return self.CONTINUE  # avoid checkpoint overhead

        logger.debug("Updating trial state for trial %s at time %s", trial, time)
        # update internal information, does not create a checkpoint!
        self._save_trial_state(state, time, result, trial)

        if not self._synch:
            state.last_perturbation_time = time
            # Divide executed trials in upper_quantile to safe
            lower_quantile, upper_quantile = self._quantiles()
            decision = self.CONTINUE
            for other_trial in tune_controller.get_trials():
                if other_trial.status in [Trial.PENDING, Trial.PAUSED]:
                    decision = self.PAUSE
                    break
            self._checkpoint_or_exploit(trial, tune_controller, upper_quantile, lower_quantile)
            return self.NOOP if trial.status == Trial.PAUSED else decision

        # calls
        self._checkpoint_or_exploit  # current_step (- last perturbation time) < self._perturbation_interval
        return super().on_trial_result(tune_controller, trial, result)

    def _quantiles(self) -> Tuple[List[Trial], List[Trial]]:
        """Returns trials in the lower and upper `quantile` of the population.

        If there is not enough data to compute this, returns empty lists.

        This method allows for quantile_fraction > 0.5 as well.
        """
        if self._quantile_fraction <= 0.5:
            return super()._quantiles()
        trials = []
        for trial, state in self._trial_state.items():
            logger.debug("Trial %s, state %s", trial, state)
            if trial.is_finished():
                logger.debug("Trial %s is finished", trial)
            if state.last_score is not None and not trial.is_finished():
                trials.append(trial)
        # called after trial_save -> last_score is not None
        trials.sort(key=lambda t: self._trial_state[t].last_score or 0.0)

        if len(trials) <= 1:
            return [], []
        num_trials_in_quantile = math.ceil(len(trials) * self._quantile_fraction)
        if num_trials_in_quantile == len(trials) and self._quantile_fraction < 1.0:
            # have at least one upper
            num_trials_in_quantile -= 1
        if num_trials_in_quantile > len(trials) / 2:
            return trials[:num_trials_in_quantile], trials[num_trials_in_quantile:]
        return (trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:])

    def _checkpoint_or_exploit(
        self,
        trial: Trial,
        tune_controller: "TuneController",
        upper_quantile: List[Trial],
        lower_quantile: List[Trial],
    ):
        """Checkpoint if in upper quantile, exploits if in lower."""
        state = self._trial_state[trial]
        # NEW: Create a checkpoint anyway
        logger.debug("Instructing %s to save.", trial)
        checkpoint = tune_controller._schedule_trial_save(trial, result=state.last_result)
        if trial in upper_quantile:
            # The trial last result is only updated after the scheduler
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
        else:
            state.last_checkpoint = None  # not a top trial

        if trial in lower_quantile:
            trial_to_clone = random.choice(upper_quantile)  # if quantile is 0, isn't this empty?
            assert trial is not trial_to_clone
            clone_state = self._trial_state[trial_to_clone]
            last_checkpoint = clone_state.last_checkpoint

            logger.debug("Trial %s is in lower quantile. Exploiting trial %s.", trial, trial_to_clone)

            if isinstance(last_checkpoint, _FutureTrainingResult):
                training_result = last_checkpoint.resolve()

                if training_result:
                    clone_state.last_result = training_result.metrics
                    clone_state.last_checkpoint = training_result.checkpoint
                    last_checkpoint = clone_state.last_checkpoint
                else:
                    logger.debug(
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

    def _check_result(self, result: dict):
        if self._time_attr not in result:
            time_missing_msg = (
                "Cannot find time_attr {} "
                "in trial result {}. Make sure that this "
                "attribute is returned in the "
                "results of your Trainable.".format(self._time_attr, result)
            )
            if self._require_attrs:
                raise RuntimeError(
                    time_missing_msg + "If this error is expected, you can change this to "
                    "a warning message by "
                    "setting PBT(require_attrs=False)"
                )
            if log_once("pbt-time_attr-error"):
                logger.warning(time_missing_msg)
        if self._metric not in result:
            metric_missing_msg = (
                "Cannot find metric {} in trial result {}. "
                "Make sure that this attribute is returned "
                "in the "
                "results of your Trainable.".format(self._metric, result)
            )
            if self._require_attrs:
                raise RuntimeError(
                    metric_missing_msg + "If this error is expected, "
                    "you can change this to a warning message by "
                    "setting PBT(require_attrs=False)"
                )
            if log_once("pbt-metric-error"):
                logger.warning(metric_missing_msg)
