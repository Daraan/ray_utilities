from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, NamedTuple

from ray.tune.experiment.trial import Trial
from ray.tune.logger import LoggerCallback

from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import extract_trial_id_from_checkpoint, make_experiment_key
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray_utilities.typing import ForkFromData

_logger = logging.getLogger(__name__)

TrialID = str


class ForkedTrialInfo(NamedTuple):
    parent_trial: Trial | TrialID
    """The trial or trial id this trial was forked from. Usually just an ID when loaded from a checkpoint."""
    forked_step: int | None
    """At which step the trial was forked. None if unknown, e.g. when extracted from checkpoint path."""

    @property
    def parent_is_present(self) -> bool:
        """
        Whether the parent trial is also currently tracked,
        i.e. :attr:`parent_trial` is a :class:`ray.tune.experiment.Trial`.
        """
        return isinstance(self.parent_trial, Trial)


class TrackForkedTrialsMixin(LoggerCallback):
    """
    Provides:
    - :meth:`trial_is_forked` to check whether a trial was forked from another trial
    - :meth:`get_forked_trial_info` to get information about the parent trials
      a trial was forked from.
    - `_forked_trials` attribute to track forked trials.

    In the ``trial.config`` the key :const:`FORKED_FROM` is expected to be present
    for this to work.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._forked_trials: defaultdict[Trial, list[ForkFromData]] = defaultdict(list)
        self._current_fork_ids: dict[Trial, str] = {}
        """fork_id of currently running forked trials"""

    def trial_is_forked(self, trial: Trial) -> bool:
        """Whether the given trial was forked from another trial."""
        if trial in self._forked_trials or trial in self._current_fork_ids:
            assert trial in self._current_fork_ids
            assert trial in self._forked_trials
            return True
        return False

    def get_forked_trial_info(self, trial: Trial) -> list[ForkFromData]:
        """Get information about the parent trials this trial was forked from.

        If the trial was forked multiple times (e.g. from a chain of forks),
        multiple entries are returned, in the order of forking.
        """
        return self._forked_trials.get(trial, [])

    def make_forked_trial_name(self, trial: Trial, fork_data: ForkFromData) -> str:
        trial_name = str(trial)
        parent_id = fork_data["parent_id"]
        ft = fork_data.get("parent_time", None)
        if ft is not None:  # pyright: ignore[reportUnnecessaryComparison]
            trial_name += f"_forkof_{parent_id}_{ft[0]}={ft[1]}"  # type: ignore[index]
        else:
            # Fallback: use training_iteration if fork_time not available
            trial_name += f"_forkof_{parent_id}_training_iteration={fork_data.get('fork_training_iteration', 0)}"
        return trial_name

    def make_forked_trial_id(self, trial: Trial, fork_data: ForkFromData) -> str:
        return make_experiment_key(trial, fork_data)

    def add_forked_trial_id(self, trial: Trial, fork_data: ForkFromData | None) -> str:
        """
        As we need to fork an already forked trial. We need to know the fork_id we give
        to the trial when we fork it again.
        """
        if fork_data is not None:
            fork_id = self.make_forked_trial_id(trial, fork_data)
        else:
            # assume we load for example from a checkpoint and the parent is currently not running
            # hence the id of the trial does not conflict with the parent
            fork_id = trial.trial_id
        # Every trial can have only one fork_id as it is currently running
        self._current_fork_ids[trial] = fork_id
        return fork_id

    def get_forked_trial_id(self, trial: Trial) -> str | None:
        """Get the forked_id of a trial, if it was already added."""
        return self._current_fork_ids.get(trial, None)

    def on_trial_start(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        # TODO: Is the trials list cleared when a trial ends? Probably not.
        if "cli_args" in trial.config and (checkpoint := trial.config["cli_args"].get("from_checkpoint")):
            # If the trial was started from a checkpoint, we can try to extract
            # the parent trial id from the checkpoint path.
            extracted_id = extract_trial_id_from_checkpoint(checkpoint)
            if extracted_id is not None:
                # We found a valid trial id in the checkpoint path.
                # This might be more reliable than FORK_FROM in config,
                # because that one might be missing if the user manually
                # started a new trial from a checkpoint.
                if not self.trial_is_forked(trial):
                    self._forked_trials[trial] = []
                # need to load the checkpoint first too see more information
                self._forked_trials[trial].append(
                    {
                        "parent_id": extracted_id,
                        "controller": "from_checkpoint",
                    }
                )
                _logger.info("Trial %s was started from checkpoint of trial %s", trial.trial_id, extracted_id)
            self.add_forked_trial_id(trial, fork_data=None)
        if FORK_FROM in trial.config:
            fork_data: ForkFromData = trial.config[FORK_FROM]
            parent_trial_id = fork_data["parent_id"]
            # Could be a live or past trial
            if "forkof" in parent_trial_id or "_step=" in parent_trial_id or "fork_from" in parent_trial_id:
                _logger.error("Unexpected parent trial id format: %s", parent_trial_id)
            parent_trial = next((t for t in trials if t.trial_id == parent_trial_id), None)
            if parent_trial is not None:
                fork_data["parent_trial"] = parent_trial
            self._forked_trials[trial].append(fork_data)
            self.add_forked_trial_id(trial, fork_data=fork_data)
            _logger.info(
                "Trial %s was forked from %s, fork_id of this trial %s, parent data: %s",
                trial.trial_id,
                parent_trial_id,
                self.get_forked_trial_id(trial),
                fork_data,
            )
        # calls log_trial_start
        super().on_trial_start(iteration, trials, trial, **info)

    def on_trial_complete(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        super().on_trial_complete(iteration, trials, trial, **info)
        self._current_fork_ids.pop(trial, None)
