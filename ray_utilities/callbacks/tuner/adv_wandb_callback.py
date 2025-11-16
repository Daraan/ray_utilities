from __future__ import annotations

import logging
import os
import pickle
import subprocess
import threading
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, ClassVar, List, Optional, cast

import ray
from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log, _QueueItem, _WandbLoggingActor
from ray.rllib.utils.metrics import EVALUATION_RESULTS
from ray.tune.experiment import Trial

from ray_utilities.callbacks.tuner.new_style_logger_callback import LogMetricsDictT, NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.callbacks.tuner.wandb_helpers import FutureArtifact, FutureFile
from ray_utilities.callbacks.wandb import WandbUploaderMixin
from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS, EVALUATED_THIS_STEP, FORK_FROM, get_run_id
from ray_utilities.misc import (
    close_process_pipes,
    deep_freeze,
    extract_trial_id_from_checkpoint,
    make_experiment_key,
    warn_if_slow,
)
from ray_utilities.nice_logger import ImportantLogger

try:
    from wandb import Settings as WandbSettings
    from wandb import Video
except ModuleNotFoundError:

    class _WandbNotInstalled:
        pass

    _WandbLoggingActorWithArtifactSupport = _WandbNotInstalled
else:
    from ._adv_wandb_logging_actor import _WandbLoggingActorWithArtifactSupport


from ._log_result_grouping import non_metric_results
from ._save_video_callback import SaveVideoFirstCallback

if TYPE_CHECKING:
    from ray.actor import ActorProxy
    from ray.tune.experiment import Trial

    import wandb
    from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor
    from ray_utilities.typing import ForkFromData
    from ray_utilities.typing.metrics import (
        VideoMetricsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
    )

_logger = logging.getLogger(__name__)

_gather_upload_locks: dict[object, threading.Lock] = {}


class AdvWandbLoggerCallback(
    NewStyleLoggerCallback, SaveVideoFirstCallback, TrackForkedTrialsMixin, WandbUploaderMixin, WandbLoggerCallback
):
    AUTO_CONFIG_KEYS: ClassVar[list[str]] = list(
        {
            *WandbLoggerCallback.AUTO_CONFIG_KEYS,
            *non_metric_results,
        }
    )

    _logger_actor_cls = _WandbLoggingActorWithArtifactSupport

    _logged_architectures: set[Trial]

    _monitor: ActorProxy[WandbRunMonitor] | None = None
    """The WandbRunMonitor instance used to monitor parents of forked runs and ensure history artifacts are created."""

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        group: Optional[str] = None,
        excludes: Optional[list[str]] = None,
        upload_checkpoints: bool = False,
        video_kwargs: Optional[dict] = None,
        image_kwargs: Optional[dict] = None,
        upload_intermediate: bool = False,
        upload_at_end: bool = True,
        **kwargs,
    ):
        """For ``kwargs`` see :class:`ray.air.integrations.wandb.WandbLoggerCallback`

        Args:
            upload_intermediate: If True, upload offline experiments intermediately when trials
                complete/pause (corresponds to "offline+upload"). If False, uploads only happen
                at experiment end (corresponds to "offline+upload@end").
            upload_at_end: If True, offline experiments will be uploaded at the end of the experiment.
                However, if upload_intermediate is True, this flag has no effect as uploads already happened
                intermediately. However, in case of errors one or the other might not trigger.
                One should handle uploads manually in such cases after the experiment.
        """
        kwargs.update(
            {
                "project": project,
                "group": group,
                "excludes": excludes or [],
                "upload_checkpoints": upload_checkpoints,
                "video_kwargs": video_kwargs,
                "image_kwargs": image_kwargs,
            }
        )
        super().__init__(**kwargs)
        self._trials_created = 0
        self._trials_started = 0
        """A Trial can be started multiple times due to restore."""
        self._logged_architectures = set()
        self.upload_intermediate = upload_intermediate
        """If True, offline experiments will be uploaded intermediately when trials complete/pause.

        Corresponds to "offline+upload" mode. When False, uploads only happen at experiment end
        via the setup's upload_offline_experiments() method (corresponds to "offline+upload@end").
        """

        self.upload_at_end = upload_at_end
        """If True, offline experiments will be uploaded at the end of the experiment."""

        # Gather uploads tracking
        self._trials_ending: dict[Trial, tuple[Optional[bool], Optional[ray.ObjectRef[_WandbLoggingActor]]]] = {}
        """Trials that are currently ending

        The first element of the value tuple tells whether the logging actor has finished writing the data to disk.
        The second element is the ray.ObjectRef of the logging actor. Both elements can be None if we are unsure
        of the state of the logging actor or have no access to it.
        """
        self._gather_timer: Optional[threading.Timer] = None
        self._gather_timeout_min = 10.0  # seconds to wait for more trials to finish
        self._active_trials_count = 0
        self._gatherer_threads: list[threading.Thread] = []
        self._local_threads: dict[Trial, threading.Thread] = {}
        self._local_logging = True
        self._seen_config_hashes: defaultdict[str, set[int]] = defaultdict(set)
        self._last_config_log: dict[str, tuple[int, int]] = {}
        """
        Track iteration of last config log per trial ID, we want to log duplicates from time to time but not often.
        Tracks as tuple of (training_iteration, current_step)
        """

    @property
    def _gather_uploads_lock(self) -> threading.Lock:
        """Get the lock used for gathering uploads."""
        if self not in _gather_upload_locks:
            _gather_upload_locks[self] = threading.Lock()
        return _gather_upload_locks[self]

    def get_state(self) -> dict:
        """Get the state of the callback for checkpointing.

        Returns:
            Dictionary containing callback state. Unpicklable objects like
            Ray actors, futures, threads, and locks are excluded.
        """
        state = super().get_state() if hasattr(super(), "get_state") else {}
        # Store basic counters and flags
        state.update(
            {
                "trials_created": self._trials_created,
                "trials_started": self._trials_started,
                "logged_architectures": [
                    trial.trial_id if not isinstance(trial, str) else trial for trial in self._logged_architectures
                ],
                # "upload_intermediate": self.upload_intermediate,
                "gather_timeout_min": self._gather_timeout_min,
                # "active_trials_count": self._active_trials_count,
            }
        )
        return state

    def set_state(self, state: dict) -> None:
        """Set the state of the callback from checkpoint data.

        Args:
            state: State dictionary containing callback state.

        Note:
            Ray actors, futures, threads, and locks cannot be restored and will
            be recreated as needed when trials restart.
        """
        if hasattr(super(), "set_state"):
            super().set_state(state)

        self._trials_created = state.get("trials_created", 0)
        self._trials_started = state.get("trials_started", 0)
        # Convert trial IDs back to trial objects when they restart
        # For now, just clear the set - it will be repopulated
        self._logged_architectures = set()
        # self.upload_intermediate = state.get("upload_intermediate", False)
        self._gather_timeout_min = state.get("gather_timeout_min", 10.0)
        # self._active_trials_count = state.get("active_trials_count", 0)

        _logger.info(
            "Restored AdvWandbLoggerCallback state: %d trials created, %d started",
            self._trials_created,
            self._trials_started,
        )

    def on_trial_start(self, iteration: int, trials: list["Trial"], trial: "Trial", **info):
        super().on_trial_start(iteration, trials, trial, **info)
        _logger.debug("Trials created: %d, re-started: %d", self._trials_created, self._trials_started)
        self._trials = trials  # keep them in case of a failure to access paths.
        # Track active trials for gather_uploads
        with self._gather_uploads_lock:
            self._active_trials_count = len([t for t in trials if t.status in ("RUNNING", "PENDING", "PAUSED")])

    def log_trial_start(self, trial: "Trial"):
        config = deepcopy(trial.config)

        config.pop("callbacks", None)  # Remove callbacks
        config.pop("log_level", None)

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        exclude_results += self.excludes

        # Log config keys on each result?
        if not self.log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = trial.trial_id if trial else None
        trial_name = str(trial) if trial else None

        # Project name for Wandb
        wandb_project = self.project

        # Grouping
        wandb_group = self.group or trial.experiment_dir_name if trial else None

        # remove unpickleable items!
        config: dict[str, Any] = _clean_log(config)  # pyright: ignore[reportAssignmentType]
        config = {key: value for key, value in config.items() if key not in self.excludes}
        config["run_id"] = get_run_id()
        # replace potential _ in trial_id
        # --- New Code --- : Remove nested keys
        for nested_key in filter(lambda x: "/" in x, self.excludes):
            key, sub_key = nested_key.split("/")
            if key in config:
                config[key].pop(sub_key, None)
        fork_from = fork_id = fork_iteration = None  # new run
        if "cli_args" in config:
            assert "num_jobs" not in config["cli_args"]
            assert "test" not in config["cli_args"]
            if trial.config["cli_args"].get("from_checkpoint"):
                fork_id = extract_trial_id_from_checkpoint(trial.config["cli_args"]["from_checkpoint"])
                # get id of run
                if fork_id is None:
                    _logger.error(
                        "Cannot extract trial id from checkpoint name: %s. "
                        "Make sure that it has to format id=<part1>_<sample_number>",
                        trial.config["cli_args"]["from_checkpoint"],
                    )
                else:
                    # Need to change to format '<run>?<metric>=<numeric_value>'
                    # Where metric="_step"; open state pickle to get iteration
                    ckpt_dir = Path(trial.config["cli_args"]["from_checkpoint"])
                    state = None
                    if (state_file := ckpt_dir / "state.pkl").exists():
                        with open(state_file, "rb") as f:
                            state = pickle.load(f)
                    elif (ckpt_dir / "_dict_checkpoint.pkl").exists():
                        with open(ckpt_dir / "_dict_checkpoint.pkl", "rb") as f:
                            state = pickle.load(f)["state"]
                    if state is None:
                        _logger.error(
                            "Could not find state.pkl or _dict_checkpoint.pkl in the checkpoint path. "
                            "Cannot use fork_from with wandb"
                        )
                    else:
                        iteration = state["trainable"]["iteration"]
                        fork_from = f"{fork_id}?_step={iteration}"
                fork_iteration = None  # NOTE: Cannot fork twice in same run; would need Checkpoint to determine step
        # we let take FORK_FROM a higher priority
        if FORK_FROM in trial.config:
            fork_data = cast("ForkFromData", trial.config[FORK_FROM])
            fork_id = fork_data.get("parent_fork_id", None)
            # We should always have a fork_id currently, but if not, fall back below.
            if fork_id is None:  # pyright: ignore[reportUnnecessaryComparison]
                _logger.warning("No parent_fork_id in FORK_FROM data: %s. Falling back to parent_trial_id", fork_data)
                fork_id = fork_data.get("parent_trial_id", None)
            fork_iteration = fork_data["parent_training_iteration"]
            fork_from = f"{fork_id}?_step={fork_iteration}"
            # We should not have multiple ?_step= in the id
            trial_id = self.get_forked_trial_id(trial)
            assert trial_id is not None, "Expected trial_id to be set on super for forked trial."
            trial_name = self.make_forked_trial_name(trial, fork_data)
            # Set experiment key using dict-based fork data
            config.setdefault("experiment_key", make_experiment_key(trial, fork_data))
            if trial_id != config["experiment_key"]:
                _logger.error(
                    "Logged trial_id and config['experiment'] do not match: %s vs %s",
                    trial_id,
                    config["experiment_key"],
                )
        else:
            # No fork info present in config; use non-fork key
            # Use get_trial_id to get the consistent trial ID
            trial_id = self.get_trial_id(trial)
            config.setdefault("experiment_key", make_experiment_key(trial))
        if self.is_trial_forked(trial) and FORK_FROM not in trial.config:
            assert trial in self._currently_not_forked_trials
            trial_name = None  # keep name from parent trial when continuing a fork

        # Test for invalid chars
        assert not trial_id or all(c not in trial_id for c in r"/ \ # ? % :"), f"Invalid character in: {trial_id}"
        assert fork_from is None or fork_from.count("?_step=") == 1, fork_from
        # NOTE: We never want FORK_FROM to be in the trials.config by default.

        start = time.time()
        use_monitor = self.upload_intermediate and fork_from and self.is_wandb_enabled(self.kwargs)
        if use_monitor and self._monitor is None:
            # Start the monitor to track parent runs of forked trials
            if self.project is None:
                _logger.warning("Cannot start WandbRunMonitor without wandb project name set. Using 'default'.")
            else:
                self._start_monitor_safe()
        if use_monitor and fork_from and self._monitor is not None:
            visit_page_future = self._monitor.visit_run_page.remote(fork_id)  # pyright: ignore[reportFunctionMemberAccess]
            _logger.info(
                "Visiting WandB page of parent run %s for forked trial %s. This may take up to 60 seconds.",
                fork_id,
                trial.trial_id,
            )
            ray.get(visit_page_future, timeout=60)
            end = time.time()
            _logger.info(
                "Started WandbRunMonitor actor to track parent runs of forked trials in %.1f seconds", end - start
            )
        # --- End New Code
        wandb_init_kwargs = {
            "id": trial_id,  # change if forked? e.g. + forked_from
            "name": trial_name,
            "reinit": "default",  # bool is deprecated
            "allow_val_change": True,
            "group": wandb_group,
            "project": wandb_project,
            "config": config,
            # possibly fork / resume
            "fork_from": fork_from,
        }
        wandb_init_kwargs.update(self.kwargs)
        if fork_from:
            wandb_init_kwargs.setdefault("tags", []).append("forked")
        if "__ptb_main_branch__" in trial.config:
            wandb_init_kwargs.setdefault("tags", []).append("pbt_main_branch")
        if "settings" in wandb_init_kwargs:
            # assure that we do not modify this
            wandb_init_kwargs["settings"] = cast("wandb.Settings", wandb_init_kwargs["settings"]).model_copy()
            # Set symlink to false as remote sync tries to fetch this with old values when using PBT
            wandb_init_kwargs["settings"].symlink = False
        else:
            wandb_init_kwargs["settings"] = WandbSettings(symlink=False)  # pyright: ignore[reportPossiblyUnboundVariable]
        if trial not in self._trial_logging_actors and trial not in self._local_threads:
            self._trials_created += 1

        # Determine if we need to restart the logging actor
        # The logging actor is present if:
        # 1. Trial is being forked (has fork_from and actor exists)
        # 2. Trial is being resumed after pause (actor exists but trial was paused)
        needs_restart = trial in self._trial_logging_futures or trial in self._local_threads

        if needs_restart:
            # Actor already exists, need to restart it
            if fork_from:
                # Forking scenario
                assert self.is_trial_forked(trial), "Expected trial to be tracked as forked trial."
                _logger.debug("Restarting logging actor for forked trial %s", trial.trial_id)
            else:
                # Resume scenario - trial was paused and is now continuing
                _logger.debug("Restarting logging actor for resumed trial %s", trial.trial_id)
            self._restart_logging_actor(trial, **wandb_init_kwargs)
        else:
            # No actor exists yet, start a new one
            # can be forked from a checkpoint, if not stopped does not start a new
            self._start_logging_actor(trial, exclude_results, **wandb_init_kwargs)
        self._trials_started += 1

    def _start_logging_actor(
        self, trial: "Trial", exclude_results: List[str], *, local: Optional[bool] = None, **wandb_init_kwargs
    ):
        # Allow local wandb logging without a separete remote actor and queue
        # Reuse actor if one already exists.
        # This can happen if the trial is restarted.
        if local is None:
            local = self._local_logging
        if not local:
            super()._start_logging_actor(trial, exclude_results, **wandb_init_kwargs)
            return
        if trial in self._local_threads:
            return

        self._trial_queues[trial] = Queue()
        assert issubclass(self._logger_actor_cls, _WandbLoggingActorWithArtifactSupport), (
            "Can only use adv loggers in local mode"
        )
        assert not TYPE_CHECKING or not issubclass(self._logger_actor_cls, _WandbNotInstalled)
        assert trial.local_path
        # TODO: MedianStoppingPruner seems to not end trials correctly.
        wandb_init_kwargs["reinit"] = "create_new"  # <-- create a new run
        local_logging_actor = self._logger_actor_cls(
            # Despite getcwd() maybe pointing to a temp _ray_pkg_dir_ it should be fine as we keep chdir there.
            logdir=os.getcwd(),  # <-- in init will call os.chdir, we do not want this in local mode
            queue=self._trial_queues[trial],
            exclude=exclude_results,
            to_config=self.AUTO_CONFIG_KEYS,
            dir=trial.local_path,  # <-- DO NOT USE logdir which changes chdir
            **wandb_init_kwargs,
        )

        def run_logging_actor(restarts=0):
            try:
                local_logging_actor.run()
                _logger.info(
                    "Logging actor for trial %s completed successfully with %d restarts", trial.trial_id, restarts
                )
            except Exception:
                _logger.exception("Error in logging actor for trial %s (restarts %d)", trial.trial_id, restarts)
                if wandb_init_kwargs.get("mode") == "offline" and not local_logging_actor.run_initialized:
                    # If error happend during init we cannot recover but otherwise we can also in offline mode
                    _logger.error(
                        "Cannot start logging actor for trial %s in offline mode. wandb.init failed. Settings: %s",
                        local_logging_actor._trial_name,
                        wandb_init_kwargs,
                    )
                    # If we resume and the previous run did not report finish we run into an error here.
                    if wandb_init_kwargs.get("resume") in ("must", True):
                        # if we resume a run and the previous run with the same id has not finished wandb.init will fail
                        # restarting might help but it did show it was not reliable
                        _logger.error("Error while resuming a run. Waiting and trying again...")
                        time.sleep(10)
                    else:
                        raise  # cannot recover from online errors - that happen during init
                if restarts < 4:
                    run_logging_actor(restarts=restarts + 1)
                else:
                    raise

        logging_thread = threading.Thread(
            target=run_logging_actor,
            name=f"wandb-actor-{trial.trial_id}",
        )
        logging_thread.start()

        self._local_threads[trial] = logging_thread

    def _signal_logging_actor_stop(self, trial: "Trial") -> None:
        """Signal the logging actor/thread to stop gracefully.

        For local threads, sends END signal via queue.
        For remote actors, delegates to parent class.
        """
        if trial in self._trial_queues:
            _logger.debug("Sending END signal to local logging thread for trial %s", trial.trial_id)
            super()._signal_logging_actor_stop(trial)
        else:
            _logger.warning("No queue found for local thread of trial %s", trial.trial_id)

    def _cleanup_logging_actor(self, trial: "Trial"):
        if trial in self._local_threads:
            thread = self._local_threads.pop(trial)
            if thread.is_alive():
                thread.join(timeout=10)
                if thread.is_alive():
                    _logger.warning("Logging thread for trial %s did not terminate after 10s.", trial.trial_id)
            else:
                _logger.debug("Logging thread already finished for trial %s", trial.trial_id)
        else:
            super()._cleanup_logging_actor(trial)

    def is_wandb_enabled(self, wandb_init_kwargs: dict[str, Any]) -> bool:
        """Helper to check if WandB logging is enabled based on mode."""
        return wandb_init_kwargs.get("mode") != "disabled"

    def _restart_logging_actor(self, trial: "Trial", **wandb_init_kwargs):
        """Ends the current logging actor and starts a new one. Useful for resuming with a new ID / settings.

        This is used when:
        1. A trial is forked - needs to end current run and start with new fork ID
        2. A trial is paused & resumed - needs to resume the existing run

        Note: In the normal workflow where on_trial_start is called before log_trial_start,
        the trial ID is already set in TrackForkedTrialsMixin.on_trial_start, so
        new_trial_id == previous_trial_id is always true. This method handles both:
        - Resume: same trial ID, no fork_from -> sets resume="must"
        - Fork: same trial ID but has fork_from -> creates forked run
        """
        # Get the new trial ID that we're about to start with
        # This comes from log_trial_start which gets it via get_forked_trial_id or get_trial_id
        new_trial_id = wandb_init_kwargs.get("id", trial.trial_id)

        # Get the previous trial ID that was being used before restart
        # This is the experiment_key that was previously logged
        # In normal flow: new_trial_id == previous_trial_id (both set in on_trial_start)
        previous_trial_id = self.get_trial_id(trial)

        # End current logging actor and optionally upload if in offline mode
        start = time.time()
        self.log_trial_end(trial, failed=False, gather_uploads=True, restart=True)
        end = time.time()
        if end - start > 20.0:
            _logger.warning("WandB log_trial_end took a long time: %.2fs", end - start)
        _logger.debug("Restarting WandB logging actor for trial %s", trial.trial_id)
        # Wait a bit before starting the next one
        self._cleanup_logging_actors(timeout=5, kill_on_timeout=False, trial_to_watch=trial)
        # Clean queue and futures else a new one will not be created
        end2 = time.time()
        if end2 - end > 30.0:
            _logger.warning("WandB _cleanup_logging_actors took a long time: %.2fs", end2 - end)

        self._trial_queues.pop(trial, None)
        self._trial_logging_futures.pop(trial, None)
        self._trial_logging_actors.pop(trial, None)
        self._local_threads.pop(trial, None)

        # Determine if we should resume or fork
        # Resume: when continuing the same trial without forking (same experiment_key)
        # Fork: when creating a new forked trial (different experiment_key, has fork_from)
        is_fork = "fork_from" in wandb_init_kwargs and wandb_init_kwargs["fork_from"] is not None
        is_resume = not is_fork and new_trial_id == previous_trial_id

        if is_resume:
            # We're resuming the same trial run, not forking
            wandb_init_kwargs["resume"] = "must"
            wandb_init_kwargs["id"] = previous_trial_id
            _logger.info("Resuming WandB run with ID %s", previous_trial_id)
            if self._local_logging:
                # wandb run should have finished else we might run into errors
                # We cannot kill threads but kill_on_timeout will at least put out a warning, if it still alive.
                # TODO: possibly force finish the run; need it as a variable.
                self._wait_for_trial_actor(trial, timeout=60.0)
        elif is_fork:
            # Forking - the fork_from is already set in wandb_init_kwargs
            _logger.info("Forking WandB run: new ID %s from parent %s", new_trial_id, wandb_init_kwargs["fork_from"])
            # close monitor tab of old run:
            if len(self._past_trial_ids.get(trial, ())) == 0:  # might appear during testing when init is skipped
                _logger.warning("BUG: No past trial IDs found for trial %s", trial.trial_id)
            elif self.is_wandb_enabled(wandb_init_kwargs):
                actual_previous_id = self._past_trial_ids[trial][-1]
                _logger.debug("Closing tab of %s", actual_previous_id)
                if self._start_monitor_safe():
                    self._monitor.close_run_tab.remote(actual_previous_id)  # pyright: ignore[reportOptionalMemberAccess, reportFunctionMemberAccess]
        else:
            # Starting a new trial (shouldn't normally happen in restart)
            _logger.warning(
                "Restarting new WandB run with ID %s (was %s). This should normally not execute this function.",
                new_trial_id,
                previous_trial_id,
            )
        self._start_logging_actor(trial, self._exclude_results, **wandb_init_kwargs)

    @staticmethod
    def preprocess_videos(metrics: LogMetricsDictT) -> LogMetricsDictT:
        did_copy = False
        for keys in DEFAULT_VIDEO_DICT_KEYS:
            subdir = metrics
            for key in keys[:-1]:
                if key not in subdir:
                    break
                subdir = subdir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]
            else:
                # Perform a selective deep copy on the modified items
                subdir = cast("dict[str, VideoMetricsDict]", subdir)
                if keys[-1] in subdir and "video_path" in subdir[keys[-1]]:
                    if not did_copy:
                        metrics = metrics.copy()  # pyright: ignore[reportAssignmentType]
                        did_copy = True
                    parent_dir = metrics
                    for key in keys[:-1]:
                        parent_dir[key] = parent_dir[key].copy()  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]  # fmt: skip
                        parent_dir = parent_dir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]  # fmt: skip
                    parent_dir = cast("_LogMetricsEvalEnvRunnersResultsDict", parent_dir)
                    parent_dir[keys[-1]] = video_dict = cast("VideoMetricsDict", parent_dir[keys[-1]]).copy()  # pyright: ignore[reportTypedDictNotRequiredAccess]  # fmt: skip
                    # IMPORTANT use absolute path as local path is a ray session!
                    video_dict["video"] = Video(  # pyright: ignore[reportPossiblyUnboundVariable]
                        os.path.abspath(video_dict.pop("video_path")), format="mp4"
                    )

        return metrics  # type: ignore[return-value]

    def _wait_for_trial_actor(self, trial: "Trial", timeout: float = 60.0) -> bool:
        """Wait for a trial's logging actor/thread to finish.

        Returns True if actor/thread finished within timeout, False otherwise.
        """
        if trial in self._local_threads:
            logging_thread = self._local_threads[trial]
            _logger.debug(
                "Waiting up to %.1fs for local logging thread to finish for trial %s", timeout, trial.trial_id
            )
            logging_thread.join(timeout)
            is_done = not logging_thread.is_alive()
            if not is_done:
                _logger.warning(
                    "Logging thread for trial %s did not finish after %.1f seconds", trial.trial_id, timeout
                )
            else:
                _logger.debug("Logging thread finished for trial %s", trial.trial_id)
                self._cleanup_logging_actor(trial)
            return is_done

        if trial not in self._trial_logging_futures:
            _logger.debug("No logging actor future found for trial %s", trial.trial_id)
            return True

        future = self._trial_logging_futures[trial]
        done, remaining = ray.wait([future], num_returns=1, timeout=timeout)
        if remaining:
            _logger.debug("Logging actor for trial %s did not finish after %.1f seconds", trial.trial_id, timeout)
        if done and remaining:
            _logger.warning("Got unexpectedly done and remaining for trial %s", trial.trial_id)
        for ready_future in done:
            assert self._logging_future_to_trial.pop(ready_future) == trial
            self._cleanup_logging_actor(trial)
        return bool(done and not remaining)

    def log_trial_end(
        self,
        trial: Trial,
        failed: bool = False,  # noqa: FBT001, FBT002
        *,
        gather_uploads: bool = False,
        restart: bool = False,
        **kwargs,
    ):
        # Triggers logger stop
        shutdown_start = time.time()
        if failed:
            _logger.critical("Trial %s encountered an error. Ending logging.", trial.trial_id)

        self._signal_logging_actor_stop(trial=trial)

        _logger.debug(
            "Signaled WandB logging stop for trial %s in %.1f seconds", trial.trial_id, time.time() - shutdown_start
        )

        # If we are in offline mode, try to sync this trial's run immediately
        if self.upload_intermediate and self.kwargs.get("mode", "online") == "offline":
            # Wandb dir is likely not yet saved by actor, wait for it, super does not wait that long.
            # Wait less now if we are gathering uploads, instead wait for actor a bit more during processing

            wait_time = 30 if gather_uploads else 120
            _logger.info("Waiting up to %ss for wandb writer to finish writing data to disk...", wait_time)
            done = self._wait_for_trial_actor(trial, timeout=wait_time)
            # NOTE: Actor should have synced everything at this point
            _logger.debug(
                "WandB logging actor for trial %s shutdown took %.1f seconds. Logging Actor done: %s",
                trial.trial_id,
                time.time() - shutdown_start,
                done,
            )
            # TODO: when completed, we still might need to gather uploads if this trial is a fork,
            # but not when we load a checkpoint, but when it initially was a checkpoint and then got forked
            if gather_uploads or self.is_trial_forked(trial):
                _logger.info("Gathering more trials to upload to WandB in dependency order...")
                if self.is_wandb_enabled(self.kwargs):
                    self._start_monitor_safe()
                # Gather trials that are ending and upload them in dependency order
                self._gather_and_upload_trials(trial, actor_done=done)
            else:
                _logger.info("Syncing offline WandB run for trial %s", trial.trial_id)
                self._sync_offline_run_if_available(trial)

    def _cleanup_logging_actors(
        self,
        timeout=0.0,
        kill_on_timeout: bool = False,
        trial_to_watch: Optional[Trial] = None,  # noqa: FBT001, FBT002
    ):
        """Clean up logging actors that have finished uploading to wandb.

        For local mode, waits for threads to finish. For remote mode, uses parent class logic.

        Args:
            timeout: The number of seconds to wait for actors/threads to finish.
            kill_on_timeout: Whether to force kill actors/threads that haven't finished.
        """
        # Handle local threads
        if trial_to_watch and trial_to_watch in self._local_threads:
            self._local_threads[trial_to_watch].join(timeout)
            if timeout > 0 and kill_on_timeout:
                _logger.warning("Cannot force kill thread for trial %s - thread may be stuck", trial_to_watch.trial_id)
        for trial, thread in list(self._local_threads.items()):
            if not thread.is_alive():
                self._local_threads.pop(trial, None)
                _logger.debug("Cleaned up finished logging thread for trial %s", trial.trial_id)
            elif alive_threads := sum(1 for t in self._local_threads.values() if t.is_alive()) >= len(
                self._trial_queues
            ):
                _logger.warning("Currently alive thread count:%d", alive_threads)

        super()._cleanup_logging_actors(timeout, kill_on_timeout)

    @warn_if_slow
    def log_trial_save(self, trial: "Trial"):
        if trial in self._trial_queues:  # is not checked on super.
            if self.upload_checkpoints and trial.checkpoint:
                _logger.info("Saving trial %s checkpoint - logging current WandB data.", trial.trial_id)
            super().log_trial_save(trial)
        else:
            _logger.error("Cannot log trial save for trial %s as no logging actor/queue found.", trial.trial_id)

    def _gather_and_upload_trials(self, trial: Trial, *, actor_done: Optional[bool] = None):
        """Gather trials ending and upload them in dependency order.

        This method collects trials that are ending within a timeout period,
        builds a dependency graph based on fork relationships, and uploads
        parent trials before their children.
        """
        with self._gather_uploads_lock:
            # Add this trial to the list of trials ending
            if trial not in self._trials_ending:
                # ref is None when we are in local variant
                future_ref = self._trial_logging_futures.get(trial, None)
                self._trials_ending[trial] = (actor_done, future_ref)

            # Check if we should start gathering or wait for more trials
            # Dynamically adjust gather timeout: more active trials = longer wait, more ending = shorter wait
            min_timeout = self._gather_timeout_min
            max_timeout = 90.0
            # Increase timeout with more active trials, decrease as more trials are ending
            base_timeout = 10.0 + 6.0 * max(0, self._active_trials_count - 1)
            # Reduce timeout as more trials are ending (but not below min_timeout)
            dynamic_timeout = max(
                min_timeout,
                min(
                    max_timeout,
                    base_timeout - 4.0 * (len(self._trials_ending) - 1),
                ),
            )
            if self._gather_timer is not None:
                # Cancel and reset timer if a new trial is added
                self._gather_timer.cancel()
                _logger.debug(
                    "Resetting gather timer for trial %s. New timeout: %.1f seconds (active: %d, ending: %d)",
                    trial.trial_id,
                    dynamic_timeout,
                    self._active_trials_count,
                    len(self._trials_ending),
                )
            else:
                _logger.info(
                    "Starting gather timer for trial %s. Timeout: %.1f seconds (Trials active: %d, ending: %d). "
                    "Will reset timer on new trial addition.",
                    trial.trial_id,
                    dynamic_timeout,
                    self._active_trials_count,
                    len(self._trials_ending),
                )
            self._gather_timer = threading.Timer(dynamic_timeout, self._process_gathered_uploads)
            self._gather_timer.start()

            # Check if all active trials are now ending
            if len(self._trials_ending) >= self._active_trials_count and self._active_trials_count > 0:
                _logger.info(
                    "All %d active trials are ending, canceling timer and processing uploads immediately",
                    self._active_trials_count,
                )
                self._gather_timer.cancel()
                self._gather_timer = None
                # Process uploads immediately in a separate thread to avoid blocking
                thread = threading.Thread(target=self._process_gathered_uploads, daemon=False)
                thread.start()
                self._gatherer_threads.append(thread)

    def _process_gathered_uploads(self, *, wait=False):
        """
        Process all gathered trials and upload them in dependency order.

        This method is responsible for uploading the results of all trials that have finished within
        a recent time window. It ensures that uploads are performed in an order that respects
        parent-child (fork) dependencies between trials: parent trials are uploaded before their
        forked children. The method processes all trials gathered in :attr:`_trials_ending`,
        builds a list of their offline WandB run directories, and then uploads them using :meth:`upload_paths`.
        If any trial's logging actor was still writing data to disk, it waits before uploading to ensure data
        consistency.
        This function is typically called in a background thread and is thread-safe.
        Side effects include clearing the internal list of trials to upload and triggering
        subprocesses for WandB sync.
        """
        with self._gather_uploads_lock:
            if not self._trials_ending:
                _logger.debug("No trials to upload")
                return

            trials_to_upload = self._trials_ending.copy()
            self._trials_ending.clear()
            self._gather_timer = None

        _logger.info("Processing upload for %d gathered trials", len(trials_to_upload))

        actors_to_wait_for = [
            future for done, future in trials_to_upload.values() if done is False and future is not None
        ]
        try:
            # Build trial runs list with paths
            trial_runs: list[tuple[str, Path]] = []
            for trial in trials_to_upload.keys():
                if trial.local_path:
                    wandb_dir = Path(trial.local_path) / "wandb"
                    if wandb_dir.exists():
                        offline_runs = list(wandb_dir.glob("offline-run-*"))
                        for run_dir in offline_runs:
                            # Extract trial ID from run directory or use trial.trial_id
                            trial_id = self._extract_trial_id_from_wandb_run(run_dir) or trial.trial_id
                            trial_runs.append((trial_id, run_dir))

            if not trial_runs:
                _logger.warning("No offline runs found for gathered trials")
                return

            # Parse fork relationships
            wandb_paths = [Path(trial.local_path) / "wandb" for trial in trials_to_upload if trial.local_path]
            if actors_to_wait_for:
                # Do NOT use _wait_for_trial_actor, as the actor might be already the new one.
                _logger.info("Waiting 60s for WandB logging actors that were still writing data to disk...")
                ray.wait([actors_to_wait_for], num_returns=len(actors_to_wait_for), timeout=60, fetch_local=False)
            # Upload in dependency order; no need to wait more as we should be in a thread and uploads are subprocesses
            self.upload_paths(wandb_paths, trial_runs, wait=wait, use_tqdm=True)
        except Exception:
            _logger.exception("Error processing gathered uploads")
            # Write exception traceback to a file for debugging
            error_log_path = Path("wandb_upload_errors.log")
            with error_log_path.open("a") as f:
                f.write(f"\nException occurred during gathered uploads at {time.strftime('%Y-%m-%d %H:%M:%S')}:\n")
                traceback.print_exc(file=f)

    def _sync_offline_run_if_available(self, trial: "Trial"):
        """Sync offline WandB run for the given trial if it exists."""
        try:
            # Look for offline runs that might belong to this trial
            assert trial.local_path
            wandb_dir = Path(trial.local_path) / "wandb"  # might not be accessible
            wait = 5
            while not wandb_dir.exists() and wait < 30:
                _logger.debug("WandB directory does not exist yet, waiting %s/30s: %s", wait, wandb_dir)
                time.sleep(5)  # wait for possible sync
                wait += 5
            if not wandb_dir.exists() and trial.path is not None:
                _logger.debug("WandB directory does not exist on Tuner system %s", wandb_dir)
                # Trigger a sync from local -> remote
                if trial.storage:
                    # local_experiment_path will always work but is overkill, try only wandb folder
                    sync_locations: list[tuple[str, str]] = [
                        (trial.local_experiment_path, trial.remote_experiment_path)
                    ]
                    sync_locations.insert(0, (wandb_dir.as_posix(), (Path(trial.path) / "wandb").as_posix()))
                    for local_path, remote_path in sync_locations:
                        try:
                            if trial.storage.syncer.sync_up(
                                local_path,
                                remote_path,
                            ):
                                trial.storage.syncer.wait()
                        except FileNotFoundError:  # noqa: PERF203
                            pass
                # Remote path
                wandb_dir = Path(trial.path) / "wandb"
                if not wandb_dir.exists():
                    _logger.debug("WandB directory does not exist: %s", wandb_dir)
                    return

            # Wandb file should be bound to the trial and not duplicated
            offline_runs = list(wandb_dir.glob("offline-run-*"))
            if len(offline_runs) > 1 and FORK_FROM not in trial.config:
                # This is normal when having a forked trial or it was forked in the past
                _logger.warning("Multiple wandb offline directories found in %s: %s", wandb_dir, offline_runs)

            if not offline_runs:
                _logger.error(
                    "No wandb offline experiments found to upload in %s: %s. ", wandb_dir, list(wandb_dir.glob("*"))
                )
                return
            # Sort by modification time and take the most recent

            # when not forked likely just one item
            # TODO: Save a file with commands to upload again in case a run fails!
            for run_dir in sorted(offline_runs, key=lambda p: p.stat().st_mtime, reverse=True):
                # Use wandb sync command to upload the offline run
                _logger.info("Attempting to sync offline WandB run: %s", run_dir)
                # can use Popen for non-blocking
                upload_time_start = time.time()
                result = subprocess.run(
                    ["wandb", "sync", str(run_dir), "--append"],
                    check=False,
                    text=True,
                    timeout=600,  # timeout 10 minutes
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                upload_time_end = time.time()
                if upload_time_end - upload_time_start > 30:
                    _logger.info(
                        "Uploading offline run for trial %s took %.1f seconds. "
                        "Consider switching to a non-blocking upload (Popen).",
                        trial.trial_id,
                        upload_time_end - upload_time_start,
                    )
                else:
                    _logger.debug(
                        "Uploading offline run for trial %s took %.1f seconds.",
                        trial.trial_id,
                        upload_time_end - upload_time_start,
                    )
                self._report_upload(result, trial.trial_id)
                # TODO: Move files to not upload it again (there should be parallel folders)
                if len(offline_runs) > 1:
                    time.sleep(5)  # wait a bit between uploads

        except subprocess.TimeoutExpired:
            _logger.warning("Timeout while syncing offline run for trial %s", trial.trial_id)
        except (OSError, subprocess.SubprocessError) as e:
            _logger.warning("Failed to sync offline run for trial %s: %s", trial.trial_id, e)

    @warn_if_slow
    def log_trial_result(
        self,
        iteration: int,  # noqa: ARG002
        trial: "Trial",
        result,
    ):
        """Called each time a trial reports a result."""
        if trial not in self._trial_logging_actors and trial not in self._local_threads:
            self.log_trial_start(trial)

        # Check for model_architecture.json using trial.storage filesystem for S3 support
        # Likely exist only after the first iteration
        if (
            result["training_iteration"] > 3
            and trial not in self._logged_architectures
            and trial.storage
            and os.path.exists(trial.storage.trial_working_directory)
        ):
            s3_file = False
            file_exists = False
            model_arch_path = base_path = None
            try:
                # this should be the default case when there is a callback that creates the file
                if (
                    trial.storage
                    and os.path.exists(trial.storage.trial_working_directory)
                    and "model_architecture.json" in os.listdir(trial.storage.trial_working_directory)
                ):
                    file_exists = True
                    model_arch_path = Path(trial.storage.trial_working_directory, "model_architecture.json")
                    base_path = trial.storage.trial_working_directory
                # TODO: Remove the rest for efficiency.
                # trial.path might be S3 storage, then this fails:
                elif trial.path is not None and "model_architecture.json" in os.listdir(trial.path):  # can be S3 remote
                    file_exists = True
                    model_arch_path = Path(trial.path, "model_architecture.json")
                    base_path = trial.path
                self._logged_architectures.add(trial)
            except FileNotFoundError:
                # Use storage filesystem to check if file exists (supports S3)
                _logger.warning(
                    "Did not find model_architecture.json. trial.path does not exist (likely remote path): %s",
                    trial.path,
                )
                if trial.storage and trial.storage.storage_filesystem:
                    try:
                        fs = trial.storage.storage_filesystem
                        model_arch_path = trial.storage.trial_fs_path + "/model_architecture.json"
                        file_info = fs.get_file_info(model_arch_path)
                        file_exists = file_info.type.name != "NotFound"
                        s3_file = True
                    except Exception:
                        _logger.exception("Error checking for model_architecture.json in trial %s", trial.trial_id)
            if file_exists:
                result = result.copy()
                # Upload when run ends
                if not s3_file:
                    artifact = FutureFile(str(model_arch_path), base_path, policy="end")
                else:
                    artifact = FutureArtifact(
                        "model_architecture-" + self.get_trial_id(trial),
                        type="model_architecture",
                    )
                    model_arch_path = str(model_arch_path)
                    if not model_arch_path.startswith("s3://"):
                        model_arch_path = "s3://" + model_arch_path
                    artifact.add_reference(model_arch_path, name="model_architecture-" + self.get_trial_id(trial))
                result["model_architecture"] = artifact  # pyright: ignore[reportGeneralTypeIssues]
                _logger.debug("Storing future Artifact %s", artifact.to_dict())
            # We still add to logged architectures to not check again
            self._logged_architectures.add(trial)

        if not result.get(EVALUATED_THIS_STEP, True):
            # Do not eval metric if we did not log it, ray copies the entry.
            result.pop(EVALUATION_RESULTS, None)

        result_clean = cast("dict[str, Any]", _clean_log(self.preprocess_videos(result)))
        if not self.log_config:
            # Config will be logged once log_trial_start
            result_clean.pop("config", None)  # type: ignore
        elif "config" in result_clean:
            # Check if we have seen the config for the current trial_id, log only at the steps
            # where we actually change it. Note this creates a metric called config/...
            result_clean["config"].pop("cli_args", None)  # we never modify cli_args which should be run.config
            config_hash = hash(deep_freeze(result_clean["config"]))
            # Maybe faster:
            # import hashlib
            # import json
            # config_serialized = json.dumps(result_clean["config"], sort_keys=True, default=str)
            # config_hash = hashlib.md5(config_serialized.encode("utf-8")).hexdigest()

            trial_id = self.get_trial_id(trial)
            if config_hash in self._seen_config_hashes[trial_id]:
                last_log_iteration, last_log_step = self._last_config_log.get(trial_id, (0, 0))
                # Log from time to time no not have too spikey steps.
                if (
                    last_log_iteration + 32 <= result_clean["training_iteration"]
                    or last_log_step + 16384 * 2 <= result_clean["current_step"]
                ):
                    self._last_config_log[trial_id] = (
                        result_clean["training_iteration"],
                        result_clean["current_step"],
                    )
                else:
                    result_clean.pop("config", None)
            else:
                # improve by only logging the changed key.
                self._seen_config_hashes[trial_id].add(config_hash)
                self._last_config_log[trial_id] = (
                    result_clean["training_iteration"],
                    result_clean["current_step"],
                )

            # Check if the config has
        self._trial_queues[trial].put((_QueueItem.RESULT, result_clean))

    def on_trial_error(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        ImportantLogger.important_info(
            _logger, "Trial %s encountered an error at iteration %d. Ending logging.", trial.trial_id, iteration
        )
        return super().on_trial_error(iteration, trials, trial, **info)

    def on_experiment_end(self, trials: list[Trial], **info):
        ImportantLogger.important_info(
            _logger, "Ending experiment and closing logger actors this can take a moment (timeout 1800s). Info %s", info
        )

        for queue in self._trial_queues.values():
            queue.put((_QueueItem.END, None))

        _logger.info("Waiting for all logging threads/actors to finish writing data...")
        max_wait_per_trial = 30
        for trial in trials:
            if trial in self._local_threads or trial in self._trial_logging_futures:
                _logger.debug("Waiting for logging to finish for trial %s", trial.trial_id)
                self._wait_for_trial_actor(trial, timeout=max_wait_per_trial)

        try:
            super().on_experiment_end(trials, **info)
        except KeyboardInterrupt:
            _logger.warning("Waiting for Logging actors to end interrupted by KeyboardInterrupt")

        # Handle "offline+upload@end" mode: upload all experiments at the end
        if (
            not self.upload_intermediate
            and self.upload_at_end
            and self.kwargs.get("mode", "online") == "offline"
            and self.is_wandb_enabled(self.kwargs)
        ):
            _logger.info("Processing offline+upload@end: uploading all experiments at experiment end")
            _logger.info("Final wait for WandB logging threads to complete...")
            time.sleep(5)
            for trial in trials:
                if trial in self._trial_logging_futures or trial in self._local_threads:
                    self._wait_for_trial_actor(trial, timeout=120)

            # Collect all offline runs from all trials
            trial_runs: list[tuple[str, Path]] = []
            wandb_paths: list[Path] = []
            for trial in trials:
                if trial.local_path is None:
                    continue
                wandb_dir = Path(trial.local_path) / "wandb"
                if wandb_dir.exists():
                    offline_runs = list(wandb_dir.glob("offline-run-*"))
                    for run_dir in offline_runs:
                        trial_id = self._extract_trial_id_from_wandb_run(run_dir) or trial.trial_id
                        trial_runs.append((trial_id, run_dir))
                    if offline_runs:
                        wandb_paths.append(wandb_dir)

            if trial_runs:
                _logger.info("Found %d offline runs to upload from %d trials", len(trial_runs), len(wandb_paths))
                # Start monitor if needed for fork tracking
                if self._monitor is None:
                    self._start_monitor_safe()
                # Upload all in dependency order
                self.upload_paths(wandb_paths, trial_runs, wait=True, use_tqdm=True)
            else:
                _logger.warning("No offline runs found to upload at experiment end")

        # wait and report any remaining uploads
        failed_uploads = []
        if self._unfinished_gathered_uploads:
            self._unfinished_gathered_uploads = unfinished_from_past = [
                p for p in self._unfinished_gathered_uploads if p.poll() is None
            ]
            if unfinished_from_past:
                cast("ImportantLogger", _logger).important_info(
                    "Continuing %d unfinished wandb uploads from previous gather: %s",
                    len(unfinished_from_past),
                    unfinished_from_past,
                )
                for process in unfinished_from_past:
                    exit_code = self._failure_aware_wait(process, timeout=600, upload_service_name="wandb")
                    if exit_code != 0:
                        exit_code = self._check_with_monitor_and_retry(process)
                    if exit_code != 0:
                        failed_uploads.append(process)
            if failed_uploads and trials and trials[0].local_experiment_path:
                self._update_failed_upload_file(failed_uploads, Path(trials[0].local_experiment_path))
            elif failed_uploads:
                for process in failed_uploads:
                    close_process_pipes(process)
        # Close all open monitor tabs
        try:
            if self._monitor:
                for trial in trials:
                    trial_id = self._trial_ids.get(trial)
                    if trial_id is None:  # possibly already cleaned on_trial_complete
                        continue
                    self._monitor.close_run_tab.remote(self._trial_ids[trial])  # pyright: ignore[reportFunctionMemberAccess]
                # close possible old tabs
                for trial in trials:
                    for old_id in self._past_trial_ids[trial]:
                        self._monitor.close_run_tab.remote(old_id)  # pyright: ignore[reportFunctionMemberAccess]
        except Exception:
            _logger.exception("Error during tab clearing:")

    def wait_for_gatherer_threads(self, timeout=900):
        """Wait for all gatherer threads to finish."""
        if not self._gatherer_threads:
            return
        _logger.info("Waiting for %d gatherer threads to finish...", len(self._gatherer_threads))
        threads_left = [thread for thread in self._gatherer_threads if thread.is_alive()]
        initial_count = len(threads_left)
        start = time.time()
        while threads_left and ((now := time.time() - start) < timeout):
            thread = threads_left[0]
            if not thread.is_alive():
                threads_left.pop(0)
                continue
            _logger.info(
                "Waiting for gatherer thread %s to finish... %d/%d threads left - timeout: %.0f/%.0f",
                thread.name,
                len(threads_left),
                initial_count,
                now,
                timeout,
            )
            thread.join(timeout=8)

    def __del__(self):
        self.wait_for_gatherer_threads(timeout=700)
