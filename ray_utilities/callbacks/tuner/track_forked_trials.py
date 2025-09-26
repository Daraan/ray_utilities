from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, NamedTuple

from ray.tune.experiment.trial import Trial
from ray.tune.logger import LoggerCallback

from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import extract_trial_id_from_checkpoint, parse_fork_from

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
        self._forked_trials: defaultdict[Trial, list[ForkedTrialInfo]] = defaultdict(list)

    def trial_is_forked(self, trial: Trial) -> bool:
        """Whether the given trial was forked from another trial."""
        return trial in self._forked_trials

    def get_forked_trial_info(self, trial: Trial) -> list[ForkedTrialInfo]:
        """Get information about the parent trials this trial was forked from.

        If the trial was forked multiple times (e.g. from a chain of forks),
        multiple entries are returned, in the order of forking.
        """
        return self._forked_trials.get(trial, [])

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
                self._forked_trials[trial].append(ForkedTrialInfo(extracted_id, forked_step=None))
                _logger.info("Trial %s was started from checkpoint of trial %s", trial.trial_id, extracted_id)
        if FORK_FROM in trial.config:
            fork_data = parse_fork_from(trial.config[FORK_FROM])
            if fork_data is None:
                _logger.warning("Trial %s has invalid %s data: %s", trial.trial_id, FORK_FROM, trial.config[FORK_FROM])
                super().on_trial_start(iteration, trials, trial, **info)
                return
            parent_trial_id, forked_step = fork_data
            # Could be a live or past trial
            # TODO: Parent trial_id might be with extended info like _forkof_ or _step= if parent was forked
            if "forkof" in parent_trial_id or "_step=" in parent_trial_id or "fork_from" in parent_trial_id:
                _logger.error("Unexpected parent trial id format: %s", parent_trial_id)
            parent_trial = next((t for t in trials if t.trial_id == parent_trial_id), None)
            if parent_trial is not None:
                self._forked_trials[trial].append(ForkedTrialInfo(parent_trial, forked_step))
            else:  # use id only
                self._forked_trials[trial].append(ForkedTrialInfo(parent_trial_id, forked_step))
            _logger.info("Trial %s was forked from %s at step %d", trial.trial_id, parent_trial_id, forked_step)
        # calls log_trial_start
        super().on_trial_start(iteration, trials, trial, **info)
