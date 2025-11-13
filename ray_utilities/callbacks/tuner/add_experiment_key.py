from __future__ import annotations

from typing import TYPE_CHECKING

from ray.tune.callback import Callback

from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import make_experiment_key

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
        if FORK_FROM not in trial.config:
            trial.config["experiment_key"] = make_experiment_key(trial)
        elif fork_id := trial.config[FORK_FROM].get("fork_id_this_trial"):
            trial.config["experiment_key"] = fork_id
        else:
            trial.config["experiment_key"] = make_experiment_key(trial, trial.config[FORK_FROM])
        if "original_experiment_key" not in trial.config:
            trial.config["original_experiment_key"] = trial.config["experiment_key"]
