import math
from typing import TYPE_CHECKING, Optional

from ray.tune.experiment import Trial
from ray.tune.schedulers.pbt import PopulationBasedTraining

import time
import logging

if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController


logger = logging.getLogger(__name__)


class RunSlowTrialsFirstMixin(PopulationBasedTraining):
    __last_candidate__scheduled = None

    _unpickled = False
    """Set to True on __setstate__ to indicate the object was unpickled."""

    def choose_trial_to_run(self, tune_controller: "TuneController") -> Optional[Trial]:
        """Ensures all trials get fair share of time (as defined by time_attr).

        This enables the PBT scheduler to support a greater number of
        concurrent trials than can fit in the cluster at any given time.
        """
        candidates: list[Trial] = []
        some_unpaused = False
        trials = tune_controller.get_trials()
        for trial in trials:
            if trial.status in (Trial.RUNNING, Trial.PENDING):
                some_unpaused = True
            if trial.status in [
                Trial.PENDING,
                Trial.PAUSED,
            ]:
                if not self._synch:
                    candidates.append(trial)
                # On restore self._next_perturbation_sync is set to the initial value, so this is likely always False
                # leading to only paused trials.
                elif self._trial_state[trial].last_train_time < self._next_perturbation_sync:
                    candidates.append(trial)
        # Take trials with lowest last_train_time and secondary by batch size
        # so slower trials get scheduled first and at best run in parallel
        candidates.sort(
            key=lambda trial: (
                self._trial_state[trial].last_train_time,
                trial.config.get("minibatch_size", 256),
                trial.config.get("train_batch_size_per_learner", 512),
            )
        )
        if (not some_unpaused or self._unpickled) and not candidates:
            if self._unpickled and all(t.status == Trial.PAUSED for t in trials):
                # If not all trials were added yet we do not want to unpause.
                # wait until more trials were added
                logger.info("All trials are paused after unpickling. Waiting to assure all trials are added...")
                if self.__last_candidate__scheduled is not None:
                    self.__last_candidate__scheduled -= 1  # we execute this function up to 10/s
            if self.__last_candidate__scheduled is not None and time.time() - self.__last_candidate__scheduled > 90:
                trials = tune_controller.get_trials()
                trial_states = [trial.status for trial in trials]
                logger.warning(
                    "No candidates found to run. Not scheduling any trial. Has %d managed trials in states: %s",
                    len(trials),
                    trial_states,
                )
                # we still update this because else this would be very frequent
                self.__last_candidate__scheduled = time.time() + 60
                if all(t.status == Trial.PAUSED for t in trials):
                    # self._next_perturbation_sync is likely the initial value, need to increase it
                    max_last_time = max(self._trial_state[trial].last_train_time for trial in trials)
                    self._next_perturbation_sync = (
                        math.floor(max_last_time / self._perturbation_interval) + 1
                    ) * self._perturbation_interval
                    logger.info(
                        "All trials are paused. Increasing _next_perturbation_sync to %d to unpause trials.",
                        self._next_perturbation_sync,
                    )
            return None
        self.__last_candidate__scheduled = time.time()
        return candidates[0] if candidates else None

    def __setstate__(self, state: dict) -> None:
        # we likely receive the state dict from right after the initialization, i.e. trial data is empty
        self._unpickled = True
        self.__last_candidate__scheduled = time.time()
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.__dict__.update(state)
