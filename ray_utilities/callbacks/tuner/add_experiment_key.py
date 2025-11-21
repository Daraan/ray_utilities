from __future__ import annotations

from typing import TYPE_CHECKING

from ray.tune.callback import Callback

from ray_utilities.tune.experiments import set_experiment_key_on_trial

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial


class AddExperimentKeyCallback(Callback):
    """
    Adds ``experiment_key`` and ``original_experiment_key`` to each trial's config
    when the trial starts. If the trial is forked from another trial, the
    ``experiment_key`` is generated based on the fork information.

    The original_experiment_key should never be modified and allows to track the
    origin and side branches of forked trials.
    """

    def on_trial_start(self, iteration: int, trials: list["Trial"], trial: "Trial", **info):
        """Called after starting a trial instance.

        Arguments:
            iteration: Number of iterations of the tuning loop.
            trials: List of trials.
            trial: Trial that just has been started.
            **info: Kwargs dict for forward compatibility.

        """
        # NOTE: This callback is called *after* the trial has been started, adjusting the config here
        # will not have the desired effect of having it available during Trainable.__init__
        set_experiment_key_on_trial(trial)
