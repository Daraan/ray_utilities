from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional

from ray.tune.callback import Callback
from ray.tune.experiment import Trial

from ray_utilities.constants import FORK_FROM

if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController

logger = logging.getLogger(__name__)


class SaveTunerState(Callback):
    _scheduled_save: Literal["on_step_end", "on_trial_save", False] = False

    def schedule_tuner_save(
        self,
        tune_controller: TuneController,
        trial: Optional[Trial | list[Trial]] = None,
        log_msg: str = "",
        *,
        wait: bool = False,
    ):
        """
        Schedules a Tuner save and sync after the next trial save.
        If trial is provided, only saves when that specific trial is saved -
        However if the trial is not saved, e.g. because it is a forked trial,
        we will checkpoint the tuner at the next step end.
        """
        try:
            # NOTE the tune_controller we have is actually a wrapper - cannot use weakref to it
            self._tune_controller_ref = weakref.ref(tune_controller._tune_controller)  # pyright: ignore[reportAttributeAccessIssue]
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to create weakref for tune_controller: %s", e)
            self._tune_controller_ref = lambda: tune_controller

        if isinstance(trial, Trial):
            self._trial_ref = weakref.ref(trial)
        elif not trial:
            if isinstance(trial, list):
                logger.warning(
                    "Passed an empty list of trials to SaveTunerState.schedule_tuner_save, will save on step end instead."
                )
            self._trial_ref = None
        else:
            self._trial_ref = weakref.WeakSet(trial)
        self.log_msg = log_msg
        self.wait = wait
        # NOTE:
        # In the context of PBT the trial is the last trial but if it is perturbed no checkpoint is saved
        # => on_trial_save is not called in the near future, we should checkpoint on step end instead.
        if trial and (isinstance(trial, Iterable) or FORK_FROM not in trial.config):
            self._scheduled_save = "on_trial_save"
        else:
            self._scheduled_save = "on_step_end"

    def _save_experiment_state(self, trial: Optional[Trial] = None):
        tune_controller = self._tune_controller_ref()
        if tune_controller is None:
            return
        if self._trial_ref is None:
            trial_ref = None
            save = True
        elif isinstance(self._trial_ref, weakref.WeakSet):
            # If multiple trials were set, save after all have saved
            trial_ref = self._trial_ref
            trial_ref.discard(trial)  # pyright: ignore[reportArgumentType]
            save = len(trial_ref) == 0
        else:
            # If a specific trial was set, only save after that trial's save
            trial_ref = self._trial_ref()
            save = trial_ref == trial
        if not save:
            return
        self._scheduled_save = False
        try:
            tune_controller.checkpoint(force=True, wait=self.wait)
        except Exception as e:
            logger.error("SaveTunerState Callback - %s: Failed to save tuner state: %s", self.log_msg, e)
        else:
            logger.info(
                "SaveTunerState Callback - %s: Wrote the latest version of all result files and experiment state to %s",
                self.log_msg,
                tune_controller.experiment_path,
            )

    def on_trial_save(self, iteration: int, trials: list["Trial"], trial: "Trial", **info):
        if self._scheduled_save != "on_trial_save":
            return
        self._save_experiment_state(trial)

    def on_step_end(self, iteration: int, trials: List[Trial], **info):
        if self._scheduled_save != "on_step_end":
            return
        self._save_experiment_state(trial=None)
