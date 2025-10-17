from __future__ import annotations

import logging
import os
import pickle
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast

import ray
from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log, _QueueItem, _WandbLoggingActor
from ray.tune.experiment import Trial

from ray_utilities import RUN_ID
from ray_utilities.callbacks.tuner.new_style_logger_callback import LogMetricsDictT, NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.callbacks.tuner.wandb_helpers import FutureFile
from ray_utilities.callbacks.wandb import WandbUploaderMixin
from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS, FORK_FROM
from ray_utilities.misc import (
    extract_trial_id_from_checkpoint,
    make_experiment_key,
)

if TYPE_CHECKING:
    from ray.actor import ActorProxy

    from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor
    from ray_utilities.typing import ForkFromData

from ._log_result_grouping import non_metric_results
from ._save_video_callback import SaveVideoFirstCallback

if TYPE_CHECKING:
    from ray.tune.experiment import Trial

    from ray_utilities.typing.metrics import (
        VideoMetricsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
    )

try:
    from wandb import Video
except ModuleNotFoundError:

    class _WandbNotInstalled:
        pass

    _WandbLoggingActorWithArtifactSupport = _WandbNotInstalled
else:
    from ._adv_wandb_logging_actor import _WandbLoggingActorWithArtifactSupport

_logger = logging.getLogger(__name__)


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
        upload_offline_experiments: bool = False,
        **kwargs,
    ):
        """For ``kwargs`` see :class:`ray.air.integrations.wandb.WandbLoggerCallback`"""
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
        self.upload_offline_experiments = upload_offline_experiments
        """If True, offline experiments will be uploaded on trial completion."""

        # Gather uploads tracking
        self._gather_uploads_lock = threading.Lock()
        self._trials_ending: dict[Trial, tuple[Optional[bool], Optional[ray.ObjectRef[_WandbLoggingActor]]]] = {}
        """Trials that are currently ending

        The first element of the value tuple tells whether the logging actor has finished writing the data to disk.
        The second element is the ray.ObjectRef of the logging actor. Both elements can be None if we are unsure
        of the state of the logging actor or have no access to it.
        """
        self._gather_timer: Optional[threading.Timer] = None
        self._gather_timeout_min = 10.0  # seconds to wait for more trials to finish
        self._active_trials_count = 0

    def __getstate__(self):
        # We need to be able to pickle this class but we cannot pickle locks, remove them
        state = self.__dict__.copy()
        # Remove the lock before pickling
        state.pop("_gather_uploads_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the lock after unpickling
        self._gather_uploads_lock = threading.Lock()

    def on_trial_start(self, iteration: int, trials: list["Trial"], trial: "Trial", **info):
        super().on_trial_start(iteration, trials, trial, **info)
        _logger.debug("Trials created: %d, re-started: %d", self._trials_created, self._trials_started)
        self._trials = trials  # keep them in case of a failure to access paths.
        # Track active trials for gather_uploads
        with self._gather_uploads_lock:
            self._active_trials_count = len([t for t in trials if t.status in ("RUNNING", "PENDING", "PAUSED")])

    def log_trial_start(self, trial: "Trial"):
        config = trial.config.copy()

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
        config["run_id"] = RUN_ID
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
        use_monitor = self.upload_offline_experiments and fork_from and self.is_wandb_enabled(self.kwargs)
        if use_monitor and self._monitor is None:
            # Start the monitor to track parent runs of forked trials
            if self.project is None:
                _logger.warning("Cannot start WandbRunMonitor without wandb project name set. Using 'default'.")
            else:
                # Get singleton remote monitor
                self._monitor = self._start_monitor()
        if use_monitor and fork_from:
            visit_page_future = self._monitor.visit_run_page.remote(fork_id)  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
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

        if trial not in self._trial_logging_actors:
            self._trials_created += 1

        # Determine if we need to restart the logging actor
        # The logging actor is present if:
        # 1. Trial is being forked (has fork_from and actor exists)
        # 2. Trial is being resumed after pause (actor exists but trial was paused)
        needs_restart = trial in self._trial_logging_futures

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
        self.log_trial_end(trial, failed=False, gather_uploads=True)
        _logger.info("Restarting WandB logging actor for trial %s", trial.trial_id)
        # Wait a bit before starting the next one
        self._cleanup_logging_actors(timeout=5, kill_on_timeout=False)
        # Clean queue and futures else a new one will not be created

        self._trial_queues.pop(trial, None)
        self._trial_logging_futures.pop(trial, None)
        self._trial_logging_actors.pop(trial, None)

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
        elif is_fork:
            # Forking - the fork_from is already set in wandb_init_kwargs
            _logger.info("Forking WandB run: new ID %s from parent %s", new_trial_id, wandb_init_kwargs["fork_from"])
            # close monitor tab of old run:
            if len(self._past_trial_ids.get(trial, ())) == 0:  # might appear during testing when init is skipped
                _logger.warning("BUG: No past trial IDs found for trial %s", trial.trial_id)
            elif self.is_wandb_enabled(wandb_init_kwargs):
                actual_previous_id = self._past_trial_ids[trial][-1]
                _logger.debug("Closing tab of %s", actual_previous_id)
                self._start_monitor().close_run_tab.remote(actual_previous_id)  # pyright: ignore[reportFunctionMemberAccess]
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
        future = self._trial_logging_futures[trial]
        done, remaining = ray.wait([future], num_returns=1, timeout=timeout)
        if remaining:
            _logger.debug("Logging actor for trial %s did not finish after %.1f seconds", trial.trial_id, timeout)
        if done and remaining:
            _logger.warning("Got unexpectedly done and remaining for trial %s", trial.trial_id)
        for ready_future in done:
            assert self._logging_future_to_trial.pop(ready_future) == trial
            self._cleanup_logging_actor(trial)
        return done and not remaining

    def log_trial_end(self, trial: Trial, failed: bool = False, *, gather_uploads: bool = False):  # noqa: FBT001, FBT002
        # Triggers logger stop
        shutdown_start = time.time()
        super().log_trial_end(trial, failed)

        # If we are in offline mode, try to sync this trial's run immediately
        if self.upload_offline_experiments and self.kwargs.get("mode", "online") == "offline":
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
                    self._start_monitor()
                # Gather trials that are ending and upload them in dependency order
                self._gather_and_upload_trials(trial, actor_done=done)
            else:
                _logger.info("Syncing offline WandB run for trial %s", trial.trial_id)
                self._sync_offline_run_if_available(trial)

    def _gather_and_upload_trials(self, trial: Trial, *, actor_done: Optional[bool] = None):
        """Gather trials ending and upload them in dependency order.

        This method collects trials that are ending within a timeout period,
        builds a dependency graph based on fork relationships, and uploads
        parent trials before their children.
        """
        with self._gather_uploads_lock:
            # Add this trial to the list of trials ending
            if trial not in self._trials_ending:
                self._trials_ending[trial] = (actor_done, self._trial_logging_futures.get(trial, None))

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
                threading.Thread(target=self._process_gathered_uploads, daemon=False).start()

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
                _logger.info("Waiting 30s for WandB logging actors that were still writing data to disk...")
                ray.wait([actors_to_wait_for], num_returns=len(actors_to_wait_for), timeout=60, fetch_local=False)
            # Upload in dependency order; no need to wait more as we should be in a thread and uploads are subprocesses
            self.upload_paths(wandb_paths, trial_runs, wait=wait)
        except Exception:
            _logger.exception("Error processing gathered uploads")

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

    def log_trial_result(
        self,
        iteration: int,  # noqa: ARG002
        trial: "Trial",
        result,
    ):
        """Called each time a trial reports a result."""
        if trial not in self._trial_logging_actors:
            self.log_trial_start(trial)
            # log model config
        if trial not in self._logged_architectures and "model_architecture.json" in os.listdir(trial.path):
            if trial.path is not None:
                result = result.copy()
                file_path = os.path.abspath(os.path.join(trial.path, "model_architecture.json"))
                artifact = FutureFile(file_path, Path(file_path).parent, policy="live")
                result["model_architecture"] = artifact  # pyright: ignore[reportGeneralTypeIssues]
                self._logged_architectures.add(trial)
                _logger.debug("Storing future Artifact %s", artifact.to_dict())
            else:
                _logger.error("Cannot save model_architecture as trial.path is None")

        result_clean = _clean_log(self.preprocess_videos(result))
        if not self.log_config:
            # Config will be logged once log_trial_start
            result_clean.pop("config", None)  # type: ignore
        self._trial_queues[trial].put((_QueueItem.RESULT, result_clean))

    def on_experiment_end(self, trials: list[Trial], **info):
        _logger.info("Ending experiment and closing logger actors this can take a moment. Info %s", info)
        super().on_experiment_end(trials, **info)
        # wait and report any remaining uploads
        failed_uploads = []
        if self._unfinished_gathered_uploads:
            self._unfinished_gathered_uploads = unfinished_from_past = [
                p for p in self._unfinished_gathered_uploads if p.poll() is None
            ]
            if unfinished_from_past:
                _logger.info(
                    "Continuing %d unfinished wandb uploads from previous gather: %s",
                    len(unfinished_from_past),
                    unfinished_from_past,
                )
                for process in unfinished_from_past:
                    exit_code = self._failure_aware_wait(process, timeout=600)
                    if exit_code != 0:
                        exit_code = self._check_with_monitor_and_retry(process)
                    if exit_code != 0:
                        failed_uploads.append(process)
            if failed_uploads and trials and trials[0].local_path:
                self._update_failed_upload_file(failed_uploads, Path(trials[0].local_path))
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
