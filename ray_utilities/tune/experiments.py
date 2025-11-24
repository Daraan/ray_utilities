from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import make_experiment_key

CONFIG_HASH_EXCLUDE_KEYS = ["experiment_key", "original_experiment_key", "trial_id_history", "trial_id"]
"""
When hashing the config for uniqueness, these keys should be excluded.

This list can be extended by other modules if needed.
"""


def set_experiment_key_on_trial(trial: Trial):
    """
    Adds the folowing metakeys to the trial.config:
        - experiment_key: Unique experiment key for the trial, generated via :func:`make_experiment_key`.
        - original_experiment_key: The original experiment key before any forking.
        - trial_id_history: A dict mapping integers to experiment keys representing the history of trial IDs
    """
    if FORK_FROM not in trial.config:
        trial.config["experiment_key"] = make_experiment_key(trial)
    elif fork_id := trial.config[FORK_FROM].get("fork_id_this_trial"):
        trial.config["experiment_key"] = fork_id
    else:
        trial.config["experiment_key"] = make_experiment_key(trial, trial.config[FORK_FROM])
    i = 0
    trial_id_history = trial.config.setdefault("trial_id_history", {})
    if "original_experiment_key" not in trial_id_history:
        trial_id_history["original_experiment_key"] = trial.config["experiment_key"]
    while str(i) in trial_id_history:
        # Use str for json encoding
        if trial.config["experiment_key"] == trial_id_history[str(i)]:
            return  # already recorded
        i += 1
    trial_id_history[str(i)] = trial.config["experiment_key"]
