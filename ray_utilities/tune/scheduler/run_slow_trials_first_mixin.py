from typing import TYPE_CHECKING, Optional

from ray.tune.experiment import Trial
from ray.tune.schedulers.pbt import PopulationBasedTraining

if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController


class RunSlowTrialsFirstMixin(PopulationBasedTraining):
    def choose_trial_to_run(self, tune_controller: "TuneController") -> Optional[Trial]:
        """Ensures all trials get fair share of time (as defined by time_attr).

        This enables the PBT scheduler to support a greater number of
        concurrent trials than can fit in the cluster at any given time.
        """
        candidates: list[Trial] = []
        for trial in tune_controller.get_trials():
            if trial.status in [
                Trial.PENDING,
                Trial.PAUSED,
            ]:
                if not self._synch:
                    candidates.append(trial)
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
        return candidates[0] if candidates else None
