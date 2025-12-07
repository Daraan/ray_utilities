"""WandB utilities for callbacks and experiment uploaders."""

from __future__ import annotations

import json
import logging
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import weakref
from bdb import BdbQuit
from enum import Enum
from pathlib import Path
from textwrap import indent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeAlias,
    cast,
    overload,
)

import numpy as np
import pandas as pd

# Note ray is only necessary for the WandbRunMonitor actor
import ray
import ray.exceptions
import tree
from pyarrow.fs import LocalFileSystem
from tqdm import tqdm

from ray_utilities.callbacks.upload_helper import AnyPopen, ExitCode, UploadHelperMixin
from ray_utilities.constants import FORK_DATA_KEYS, FORK_FROM, get_run_id
from ray_utilities.misc import (
    RE_GET_TRIAL_ID,
    ExperimentKey,
    close_process_pipes,
    get_available_memory_bytes,
    get_trials_from_tuner,
)
from ray_utilities.nice_logger import ImportantLogger

try:
    import wandb.errors
    from wandb import Api
except ImportError:
    pass

if TYPE_CHECKING:
    import wandb  # noqa: TC004
    from ray import tune
    from ray.actor import ActorProxy
    from ray.tune import ResultGrid
    from wandb.apis.public.runs import Run, Runs

    from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor as _WandbRunMonitor
    from ray_utilities.typing import ForkFromData

logger = logging.getLogger(__name__)

_failed_upload_file_lock = threading.Lock()

WANDB_SYNC_MARKER = ".wandb_synced"

FailureDictType: TypeAlias = dict["Run | RunWithVerificationFailures", list["_FailureTuple"] | Exception]


def get_wandb_failed_upload_file() -> str:
    """
    Get the name of the failed uploads file for the current RUN_ID.

    It is build as: failed_wandb_uploads-{RUN_ID}.txt
    """
    return f"failed_wandb_uploads-{get_run_id()}.txt"


_wandb_api = None

MIN_TOTAL_STEPS = int(os.environ.get("MIN_TOTAL_STEPS", "1_100_000"))
"Vor experiment verification"


def wandb_api() -> Api:
    global _wandb_api  # noqa: PLW0603
    if _wandb_api is None:
        try:
            _wandb_api = Api()  # pyright: ignore[reportPossiblyUnboundVariable]
        except NameError as e:
            logger.error("wandb.Api() not available, wandb might not be installed")
            raise ModuleNotFoundError("wandb.Api() not found") from e
        except Exception as e:
            logger.error("Failed to create wandb.Api(): %s", e)
            raise
    return _wandb_api


def _amount_auto_uploads(upper_bound: int = 5) -> tuple[int, bool]:
    """
    When parallel uploads is set to "auto" checks the current memory usage and plans
    3GB for each upload.
    This function is SLURM aware and will check for memory limits in SLURM jobs.
    """
    available_mem = get_available_memory_bytes()
    # Reserve at least 5G for other processes and limit to 75% of available memory
    available_mem = min(available_mem - 5 * 1024 * 1024 * 1024, available_mem * 0.75)
    # warn if smaller than 250 MB
    safe = True
    if available_mem < 500 * 1024 * 1024:
        safe = False
        ImportantLogger.important_warning(logger, "Less than 500MB calculated on memory left Wandb upload.")
    uploads_based_on_mem = int(max(1, available_mem // (3 * 1024 * 1024 * 1024)))
    return (min(uploads_based_on_mem, upper_bound), safe)


class WandbUploaderMixin(UploadHelperMixin):
    """Mixin for uploading WandB offline experiments with dependency ordering.

    This mixin provides methods to:
    - Parse fork relationships from wandb directories
    - Build dependency graphs for upload ordering
    - Upload trials in correct order (parents before children)
    """

    _upload_service_name = "wandb"
    project: str | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._unfinished_gathered_uploads: list[AnyPopen] = []
        self._upload_to_trial: weakref.WeakKeyDictionary[AnyPopen, str] = weakref.WeakKeyDictionary()
        """
        Mapping of uploading processes to their trial IDs.

        Filled in :meth:`upload_paths` when starting uploads.
        """

        self._monitor: Optional[ActorProxy[_WandbRunMonitor]] = None
        self._history_artifact: dict[str, list[wandb.Artifact]] = {}

    def __getstate__(self):
        state = (
            super().__getstate__()  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(super(), "__getstate__")
            else self.__dict__.copy()
        )
        # Cannot pickle processes and weakdict
        state.pop("_upload_to_trial", None)
        state["_monitor"] = None
        return state

    def __setstate__(self, state):
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.__dict__.update(state)
        self._upload_to_trial = weakref.WeakKeyDictionary()

    def wandb_upload_results(
        self,
        results: Optional[ResultGrid],
        tuner: Optional[tune.Tuner] = None,
        *,
        wait: bool = True,
        parallel_uploads: int | Literal["auto"] = "auto",
        use_tqdm: bool = False,
        skip_synced: bool = True,
    ) -> list[subprocess.Popen] | None:
        """
        Upload wandb's offline folder of the session to wandb, similar to the `wandb sync` shell command

        Args:
            results: The ResultGrid containing the results of the experiment.
            tuner: Optional tuner to get additional trial information.
            wait: If True, waits for the upload to finish before returning.
            parallel_uploads: Number of parallel uploads to by executing :class:`subprocess.Popen`
            use_tqdm: Whether to use tqdm progress bars for upload progress.
            skip_synced: If True, skip paths that contain the sync marker file.
        """
        logger.info("Uploading wandb offline experiments...")

        # Step 1: Gather all wandb paths and trial information
        wandb_paths: list[Path] = self._get_wandb_paths(results, tuner)
        # FIXME: If this is set it might upload the same directory multiple times
        global_wandb_dir = os.environ.get("WANDB_DIR", None)
        if global_wandb_dir and (global_wandb_dir := Path(global_wandb_dir)).exists():
            wandb_paths.append(global_wandb_dir)
        if not wandb_paths:
            logger.warning("No wandb offline directories found to upload.")
            return None
        uploads = self.upload_paths(
            wandb_paths, wait=wait, parallel_uploads=parallel_uploads, use_tqdm=use_tqdm, skip_synced=skip_synced
        )
        return uploads

    def _monitor_check_parent_trial(self, trial_id: str, timeout: float = 40) -> bool | None:
        parent_id = self.fork_relationships.get(trial_id, (None, None))[0]
        if not parent_id:
            # we might check a trial with no parent here
            logger.debug("No parent_id found for trial %s, cannot check with monitor", trial_id)
            return None
            # TODO: Possibly extract parent id from trial_id if possible
            _, fork_data = ExperimentKey.parse_experiment_key(trial_id)
            if fork_data:
                # contains only the pure trial id not the experiment key of the parent
                parent_id = fork_data.get("parent_trial_id")

        if self._start_monitor_safe():
            assert self._monitor is not None
            page_visit = self._monitor.visit_run_page.remote(parent_id)  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
            done, _ = ray.wait([page_visit], timeout=timeout)
            # try again
            return bool(done)
        return None

    def _get_history_artifact_name(
        self, run_id: str, version: Optional[str | int] = "latest", entity: Optional[str] = None
    ) -> str:
        """Get the full name of the history artifact for a given run ID and version."""
        if not getattr(self, "project", None):
            raise ValueError("Project must be set to construct artifact name")
        if not entity:
            entity = wandb_api().default_entity
        if isinstance(version, int):
            version = "v" + str(version)
        if entity:
            name = f"{entity}/"
        else:
            name = ""
        return name + f"{self.project}/run-{run_id}-history:{version}"

    def _wait_for_artifact(self, trial_id: str, version: str | int = "latest", max_wait_time: int = 300) -> bool:
        time_waited = 0
        while time_waited < max_wait_time:
            found, _ = self._check_for_artifact(trial_id, version=version)
            if found:
                logger.info(
                    "Found history artifact for trial %s version %s after %d seconds", trial_id, version, time_waited
                )
                return True
            time.sleep(5)
            time_waited += 5
        logger.warning("Timeout waiting for history artifact for trial %s version %s", trial_id, version)
        return False

    def _check_for_artifact(self, trial_id: str, version: str | int = "latest") -> tuple[bool, bool]:
        """
        Returns:
            bool: True if the artifact exists, False otherwise.
            bool: True if a new artifact was found, False otherwise.
        """
        if not isinstance(version, str):
            version = "v" + str(version)
        api = wandb_api()
        entity = api.default_entity
        artifact_name = self._get_history_artifact_name(trial_id, version=version, entity=entity)
        if not api.artifact_exists(artifact_name):
            return False, False
        artifact = api.artifact(artifact_name)

        artifact_run = artifact.logged_by()
        if artifact_run is None:
            logger.warning("Artifact %s has no logged_by run, cannot verify", artifact_name)
            return True, False
        aliases = artifact.aliases
        digest = artifact.digest
        if trial_id not in self._history_artifact or all(a.digest != digest for a in self._history_artifact[trial_id]):
            logger.info(
                "Found new history artifact for trial %s: %s (run: %s, logged by: %s, aliases: %s, digest: %s)",
                trial_id,
                artifact_name,
                artifact_run.id if artifact_run else "unknown",
                artifact_run.entity if artifact_run else "unknown",
                aliases,
                digest,
            )
            if trial_id not in self._history_artifact:
                self._history_artifact[trial_id] = []
            self._history_artifact[trial_id].append(artifact)
            return True, True
        return True, False

    def _check_with_monitor_and_retry(self, process: AnyPopen, timeout=30) -> int:
        logger.info("Process %s failed with returncode %s, checking parent with monitor", process, process.returncode)

        start = time.time()
        trial_id = self._upload_to_trial.get(process, "")
        parent_id = self.fork_relationships.get(trial_id, (None, None))[0]
        if parent_id is None:
            logger.warning("Found no parent for %s cannot check again", trial_id)
            return ExitCode.NO_PARENT_FOUND

        try:
            _found_before, artifact_is_new = self._check_for_artifact(parent_id)
            visit_result = self._monitor_check_parent_trial(trial_id=trial_id, timeout=min(40, timeout * 0.3))
            if visit_result is None:
                logger.warning("Monitor check returned None for trial %s, monitor may not be available", trial_id)
            time.sleep(2)
            _found_after, new_artifact_after = self._check_for_artifact(parent_id)

            # Wait for artifact with reduced timeout to avoid blocking
            max_artifact_wait = min(timeout * 0.5, 30)
            artifact_wait_start = time.time()
            while (
                not (artifact_is_new or new_artifact_after) and (time.time() - artifact_wait_start) < max_artifact_wait
            ):
                time.sleep(5)
                try:
                    logger.info("Querying for artifact again for trial %s", parent_id)
                    _found_after, new_artifact_after = self._check_for_artifact(parent_id)
                except Exception:
                    logger.exception("Error checking for artifact during retry wait")
                    break
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt during monitor check and retry for trial %s", trial_id)
            return ExitCode.TERMINATED
        except Exception:
            logger.exception("Error during monitor check and retry for trial %s", trial_id)
            return ExitCode.ERROR

        # wandb args already included
        process_retry = subprocess.Popen(
            ["wandb", "sync", *cast("Iterable[str]", process.args[2:])],  # pyright: ignore[reportIndexIssue]
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )
        end = time.time()
        remaining_timeout = max(20, timeout * 0.2, timeout - (end - start))
        exit_code = self._failure_aware_wait(
            process_retry,
            timeout=remaining_timeout,
            trial_id=self._upload_to_trial.get(process, ""),
            upload_service_name="wandb",
        )
        if exit_code != 0:
            logger.error(
                "Retry of upload for trial %s also failed with exit code %d",
                Path(process.args[2]).name,  # pyright: ignore[reportArgumentType, reportIndexIssue]
                exit_code,
            )
        return exit_code

    def upload_paths(
        self,
        wandb_paths: list[Path],
        trial_runs: Optional[list[tuple[str, Path]]] = None,
        *,
        wait: bool = True,
        parallel_uploads: int | Literal["auto"] = "auto",
        use_tqdm: bool = False,
        skip_synced: bool = True,
        wandb_args: Sequence[str] = (),
    ):
        # Step 2: Collect all trial runs with their trial IDs
        if trial_runs is None:
            logger.info("No trial_runs provided, extracting from wandb paths.", stacklevel=2)
            trial_runs = []  # (trial_id, run_dir)

            wandb_paths = wandb_paths.copy()
            wandb_paths_set = set()
            skipped_synced = 0
            for i, wandb_dir in enumerate(wandb_paths):
                # Find offline run directories, there might be multiple because of resuming
                if "offline-run-" in wandb_dir.name:
                    # already an offline run, but we need parent paths only
                    offline_runs: list[Path] = [wandb_dir]
                    wandb_paths_set.add(wandb_dir.parent)
                    wandb_paths[i] = wandb_dir.parent
                else:
                    offline_runs = list(wandb_dir.glob("offline-run-*"))
                    wandb_paths_set.add(wandb_dir)

                if not offline_runs:
                    logger.error(
                        "No wandb offline experiments found to upload in %s: %s. ", wandb_dir, list(wandb_dir.glob("*"))
                    )
                    continue

                for run_dirs in offline_runs:
                    # Check for WandB's own sync marker
                    wandb_sync_files = list(run_dirs.glob("run-*.wandb.synced"))
                    if wandb_sync_files and skip_synced:
                        logger.info(
                            "Found WandB sync marker file(s) in %s: %s. "
                            "This run was likely already uploaded to WandB - %s.",
                            run_dirs,
                            [f.name for f in wandb_sync_files],
                            "skipping" if skip_synced else "uploading anyway",
                        )

                    # Check for our custom sync marker if skip_synced is enabled
                    if skip_synced and (run_dirs / WANDB_SYNC_MARKER).exists():
                        logger.debug("Skipping already synced run directory: %s", run_dirs)
                        skipped_synced += 1
                        continue

                    trial_id = self._extract_trial_id_from_wandb_run(run_dirs)
                    if trial_id:
                        trial_runs.append((trial_id, run_dirs))
                    else:
                        logger.warning(
                            "Could not extract trial ID from %s, will upload without dependency ordering", run_dirs
                        )
                        trial_runs.append((run_dirs.name, run_dirs))

            if skipped_synced > 0:
                logger.info("Skipped %d already synced run directories", skipped_synced)
        # Keep ordered remove duplicates:
        wandb_paths = list(dict.fromkeys(wandb_paths))

        if not trial_runs:
            logger.info("No wandb offline runs found to upload.")
            return None

        # Step 3: Parse fork relationships
        self.fork_relationships = self._parse_wandb_fork_relationships(wandb_paths)
        if len(self.fork_relationships) == 0:
            ImportantLogger.important_info(
                logger, "No fork relationships found. If you work with forks something went wrong"
            )
        logger.debug("Found %d fork relationships.", len(self.fork_relationships))

        # Step 4: Build dependency-ordered upload groups
        upload_groups: list[list[tuple[str, list[Path]]]] = self._build_upload_dependency_graph(
            trial_runs, self.fork_relationships
        )
        logger.debug("Created %d upload groups with dependency ordering", len(upload_groups))

        # Step 5: Upload trials in dependency order
        uploads: list[AnyPopen] = []
        finished_uploads: set[AnyPopen] = set()
        failed_uploads: list[AnyPopen] = []
        total_uploaded = 0
        if self._unfinished_gathered_uploads:
            self._unfinished_gathered_uploads = unfinished_from_past = [
                p for p in self._unfinished_gathered_uploads if p.poll() is None
            ]
            if unfinished_from_past:
                cast("ImportantLogger", logger).important_info(
                    "Continuing %d unfinished wandb uploads from previous gather: %s",
                    len(unfinished_from_past),
                    [p.args for p in unfinished_from_past],
                )
                for process in unfinished_from_past:
                    exit_code = self._failure_aware_wait(
                        process, timeout=300, terminate_on_timeout=False, upload_service_name="wandb"
                    )
                    if exit_code in (ExitCode.WANDB_BEHIND_STEP, ExitCode.WANDB_SERVER_ERROR):
                        # use monitor to check on parent, try again
                        # how do I get the parent id?
                        exit_code = self._check_with_monitor_and_retry(process)
                    if exit_code != 0:
                        failed_uploads.append(process)

        # Use tqdm for outer loop if requested
        outer_iter = tqdm(upload_groups, desc="WandB Upload Groups", leave=True) if use_tqdm else upload_groups
        unfinished_uploads = []
        group_idx = 0

        def get_parallel_uploads(*, not_safe_to_zero: bool = False) -> int:
            if parallel_uploads == "auto":
                auto_uploads, safe = _amount_auto_uploads()
                if not safe and not_safe_to_zero:
                    auto_uploads = 0
                logger.info("Auto-detected %d parallel uploads based on available memory", auto_uploads)
                return auto_uploads
            return parallel_uploads

        try:
            for group_idx, group in enumerate(outer_iter):
                logger.info("Uploading group %d/%d with %d trials", group_idx + 1, len(upload_groups), len(group))

                # Wait for previous group to complete before starting next group
                if group_idx > 0:
                    logger.info("Waiting for previous upload group to complete...")
                    finished_or_failed = []
                    # Use tqdm for waiting on previous group if requested
                    prev_iter = (
                        tqdm(uploads, desc=f"Waiting for Group {group_idx} uploads", leave=False)
                        if use_tqdm
                        else uploads
                    )
                    for process in prev_iter:
                        exit_code = self._failure_aware_wait(
                            process,
                            timeout=150,
                            trial_id=self._upload_to_trial.get(process, ""),
                            upload_service_name="wandb",
                        )
                        if exit_code == 0:
                            finished_uploads.add(process)
                            self._create_sync_marker(process, self._upload_to_trial)
                        elif self._check_with_monitor_and_retry(process, timeout=200) == 0:
                            finished_uploads.add(process)
                            self._create_sync_marker(process, self._upload_to_trial)
                        else:
                            failed_uploads.append(process)
                        finished_or_failed.append(process)
                    uploads = [p for p in uploads if p not in finished_or_failed]

                # Use tqdm for inner loop if requested
                inner_iter = tqdm(group, desc=f"Trials in Group {group_idx + 1}", leave=False) if use_tqdm else group
                for trial_id, run_dirs in inner_iter:
                    # Manage parallel upload limit within group
                    if len(uploads) >= (n_uploads_now := get_parallel_uploads()):
                        logger.info(
                            "%d >= %d uploads already in progress waiting for some to finish before starting new ones...",
                            len(uploads),
                            n_uploads_now,
                        )
                    # process uploads that are already finished:
                    for process in (p for p in uploads if p.poll() is not None):
                        exit_code = self._failure_aware_wait(
                            process,
                            timeout=60,
                            trial_id=self._upload_to_trial.get(process, ""),
                            upload_service_name="wandb",
                        )
                        if exit_code == 0:
                            finished_uploads.add(process)
                            self._create_sync_marker(process, self._upload_to_trial)
                        elif self._check_with_monitor_and_retry(process) == 0:
                            finished_uploads.add(process)
                            self._create_sync_marker(process, self._upload_to_trial)
                        else:
                            failed_uploads.append(process)
                        uploads.remove(process)
                    while len(uploads) >= get_parallel_uploads():
                        finished_or_failed = set()
                        # Prioritize checking processes that have already finished else oldest first
                        for process in sorted(uploads, key=lambda p: p.poll() is None):
                            exit_code = self._failure_aware_wait(
                                process,
                                timeout=900,
                                trial_id=self._upload_to_trial.get(process, ""),
                                upload_service_name="wandb",
                            )
                            if exit_code == 0:
                                finished_uploads.add(process)
                                self._create_sync_marker(process, self._upload_to_trial)
                            elif self._check_with_monitor_and_retry(process) == 0:
                                finished_uploads.add(process)
                                self._create_sync_marker(process, self._upload_to_trial)
                            else:
                                failed_uploads.append(process)
                            finished_or_failed.add(process)
                        uploads = [p for p in uploads if p not in finished_or_failed]
                    waited = 0
                    while get_parallel_uploads(not_safe_to_zero=True) == 0 and waited < 120:
                        # we are close to he memory limit
                        logger.warning(
                            "Detected less memory left for wandb upload. "
                            "Pausing 30s to check again and waiting up to 2 min"
                        )
                        time.sleep(30)
                        waited += 30

                    # if the run has a parent we want to check it with the monitor first
                    logger.debug("Checking with monitor before uploading trial %s", trial_id)
                    if (visit_success := self._monitor_check_parent_trial(trial_id, timeout=40)) is not None:
                        logger.debug("Monitor visit for parent of trial %s was %s", trial_id, visit_success)
                        time.sleep(5)
                    logger.info(
                        "Uploading offline wandb run for trial %s (group %d/%d, trial %d/%d in group) from dirs:\n%s",
                        trial_id,
                        group_idx + 1,
                        len(upload_groups),
                        group.index((trial_id, run_dirs)) + 1,
                        len(group),
                        [p.name for p in run_dirs],
                    )
                    process = subprocess.Popen(
                        ["wandb", "sync", *[d.as_posix() for d in run_dirs], "--append", *wandb_args],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,  # line-buffered
                    )
                    uploads.append(process)
                    self._upload_to_trial[process] = trial_id
                    total_uploaded += 1

            # Handle final completion
            if wait:
                logger.info("Waiting for all wandb uploads to finish...")
            unfinished_uploads = uploads.copy()
            # Use tqdm for waiting on unfinished uploads if requested
            if wait and use_tqdm:
                iter_uploads = tqdm(
                    sorted(uploads, key=lambda p: p.poll() is None),
                    desc="Waiting for final wandb uploads",
                    leave=True,
                )
            else:
                iter_uploads = sorted(uploads, key=lambda p: p.poll() is None)
            for process in iter_uploads:
                exit_code = None
                if wait:
                    exit_code = self._failure_aware_wait(
                        process,
                        timeout=900,
                        trial_id=self._upload_to_trial.get(process, ""),
                        upload_service_name="wandb",
                    )
                if process.poll() is not None:
                    if exit_code is None:
                        exit_code = self._report_upload(process)
                    if exit_code == 0:
                        finished_uploads.add(process)
                        self._create_sync_marker(process, self._upload_to_trial)
                    elif self._check_with_monitor_and_retry(process) == 0:
                        finished_uploads.add(process)
                        self._create_sync_marker(process, self._upload_to_trial)
                    else:
                        failed_uploads.append(process)
                    unfinished_uploads.remove(process)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, terminating ongoing wandb uploads...")
            failed_uploads.extend(unfinished_uploads)
            logger.error(
                "Not uploaded wandb runs due to KeyboardInterrupt: %s",
                (str(upload_groups[group_idx:]).replace(", ", ",\n")),
            )

        uploads = []

        if failed_uploads:
            # Close stdout pipes for all failed uploads to prevent file descriptor leaks
            try:
                formatted_failed = "\n".join(
                    f"returncode: {p.returncode} args: {' '.join(p.args)}"  # pyright: ignore[reportArgumentType, reportCallIssue]
                    for p in failed_uploads
                )
            except TypeError:
                formatted_failed = "\n".join(f"returncode: {p.returncode} args: {p.args}" for p in failed_uploads)
            logger.error("Failed to upload %d wandb runs:\n%s", len(failed_uploads), formatted_failed)
            # parent is trial dir, grandparent is experiment path
            grand_path = wandb_paths[0].parent.parent
            failed_file = self._update_failed_upload_file(failed_uploads, grand_path, self._upload_to_trial)
            for path in wandb_paths[1:]:
                if path == grand_path:
                    continue
                # copy failure doc to other experiment
                try:
                    dest = path / get_wandb_failed_upload_file()
                    if dest.exists():
                        dest.rename(dest.with_suffix(".txt.old"))
                    shutil.copyfile(failed_file, dest)
                    logger.info("Copied file for failed uploads to %s", dest.resolve())
                except Exception:
                    logger.exception("Failed to copy failed upload file to %s", path)
        if not unfinished_uploads:
            logger.info("All wandb offline runs have been tried to upload.")
        ImportantLogger.important_info(
            logger,
            "Uploaded wandb offline runs from %d trial paths: "
            "success %d, failed %d, still in progress %d from paths: %s.",
            total_uploaded,
            len(finished_uploads),
            len(failed_uploads),
            len(unfinished_uploads),
            f"wandb paths: {wandb_paths}",
        )
        if unfinished_uploads:  # There are still processes running
            self._unfinished_gathered_uploads.extend(unfinished_uploads)
            self._unfinished_gathered_uploads = [p for p in self._unfinished_gathered_uploads if p.poll() is None]
            return unfinished_uploads
        return None

    def _create_sync_marker(self, process: AnyPopen, process_to_trial: Optional[Mapping[AnyPopen, str]] = None):
        """Create a marker file in successfully synced run directories.

        Args:
            process: The completed upload process.
            process_to_trial: Optional mapping of process to trial ID for logging.
        """
        try:
            # Extract run directories from process args
            # process.args format: ['wandb', 'sync', '/path/to/run1', '/path/to/run2', '--append']
            if not isinstance(process.args, (str, bytes)) and isinstance(process.args, Iterable):
                run_paths = [
                    arg
                    for arg in process.args[2:]
                    if arg != "--append" and not (not isinstance(arg, str) or arg.startswith("-"))
                ]
            else:
                logger.warning("Cannot extract run paths from process args: %s", process.args)
                return

            trial_id = process_to_trial.get(process, "unknown") if process_to_trial else "unknown"

            for run_path_str in run_paths:
                run_path = Path(run_path_str)
                if not run_path.exists():
                    logger.warning("Run path does not exist for marker creation: %s", run_path)
                    continue

                marker_file = run_path / WANDB_SYNC_MARKER
                try:
                    marker_file.touch(exist_ok=True)
                    logger.debug("Created sync marker for trial %s at %s", trial_id, marker_file)
                except Exception:
                    logger.exception("Failed to create sync marker at %s", marker_file)
        except Exception:
            logger.exception("Error while creating sync markers for process")

    def _update_failed_upload_file(
        self,
        failed_uploads: Iterable[AnyPopen],
        file_dir: Path,
        process_to_trial: Optional[Mapping[AnyPopen, str]] = None,
    ) -> Path:
        with _failed_upload_file_lock:
            failed_file = file_dir / get_wandb_failed_upload_file()
            with failed_file.open("a") as f:
                for process in failed_uploads:
                    try:
                        trial_id = process_to_trial.get(process, "unknown") if process_to_trial else "unknown"
                        formatted_args = (
                            " ".join(map(str, process.args))
                            if not isinstance(process.args, (str, bytes)) and isinstance(process.args, Iterable)
                            else process.args
                        )
                        err = ""
                        # Check if stdout is still open and readable
                        if process.stdout and not process.stdout.closed:
                            # Check if there's data available with a timeout
                            out_data: list[bytes] | list[str]
                            output_left: bytes | str = b""
                            try:
                                out_data, _, _ = select.select([process.stdout], [], [], 1.0)
                                if out_data:
                                    output_left = process.stdout.read()
                                if isinstance(output_left, bytes):
                                    output_left = output_left.decode("utf-8")
                            except (ValueError, OSError) as e:
                                logger.debug("Could not read from process stdout: %s", e)
                                output_left = ""
                            if output_left:
                                err = "\n" + indent(output_left, prefix=" " * 4) + "\n"
                        f.write(f"{trial_id} : {formatted_args}{err}\n")
                    finally:
                        close_process_pipes(process)
        # TODO: If we write this AFTER tune is done the file will not be synced to remote storage!
        logger.warning("Wrote details of failed uploads to %s", failed_file.resolve())
        return failed_file

    def _get_wandb_paths(self, results: Optional[ResultGrid] = None, tuner: Optional[tune.Tuner] = None) -> list[Path]:
        """
        Checks the results for wandb offline directories to upload.

        The tuner can be provided in case no results are available, e.g. due to an error,
        furthermore passing the tuner allows to check for missing wandb directories.
        """
        if results is None:
            if tuner is None:
                logger.error("No results or tuner provided to get wandb paths, cannot get paths.")
                return []
            try:
                results = tuner.get_results()  # if this works below works if we have a local tuner
            except RuntimeError as e:
                logger.error("Could not get results from tuner: %s", e)
            trials = get_trials_from_tuner(tuner)
            if trials is None:
                logger.error("Could not get trials from tuner to get wandb paths.")
                return []
            trial_paths = [Path(trial.local_path) / "wandb" for trial in trials if trial.local_path]
            if len(trial_paths) != len(trials):
                logger.error("Did not get all wandb paths %d of %d", len(trial_paths), len(trials))
            return trial_paths
        # these are in the non-temp dir, could be S3 path and non-local
        result_paths = [Path(result.path) / "wandb" for result in results]
        if tuner is None:
            logger.warning("No tuner provided cannot check for missing wandb paths.")
            return result_paths
        trials = get_trials_from_tuner(tuner)
        if trials is None:
            logger.exception("Could not get trials or their paths")
        else:
            trial_paths = [Path(trial.local_path) / "wandb" for trial in trials if trial.local_path]
            existing_in_result = sum(p.exists() for p in result_paths)  # can point to S3
            existing_in_trial = sum(p.exists() for p in trial_paths)  # are local
            if (existing_in_result > 0 and existing_in_trial > 0) and existing_in_result != existing_in_trial:
                logger.error(
                    "Count of existing trials paths did not match %d vs %d: \nResult Paths:\n%s\nTrial Paths:\n%s",
                    existing_in_result,
                    existing_in_trial,
                    result_paths,
                    trial_paths,
                )
            elif existing_in_result == 0:
                logger.info(
                    "No locally existing paths found in results or trials, possibly stored remotely: %s. Using local paths for wandb upload.",
                    results.experiment_path,
                )

            if not isinstance(results.filesystem, LocalFileSystem):
                logger.info("Result filesystem is not local, using local trial paths for wandb upload.")
                if existing_in_result != 0:
                    logger.warning("Still found existing result paths despite non-local filesystem.")
                if existing_in_trial != len(trial_paths):
                    logger.error(
                        "Not all trial paths exist locally %d of %d: %s",
                        existing_in_trial,
                        len(trial_paths),
                        trial_paths,
                    )
                return [p for p in trial_paths if p.exists()]
            if existing_in_trial >= existing_in_result:
                logger.info(
                    "Found matching number of existing wandb paths in the local trial dirs "
                    "preferring them to final storage_path."
                )
                return [p for p in trial_paths if p.exists()]
            non_existing_results = [res for res in results if not (Path(res.path) / "wandb").exists()]
            if non_existing_results:
                not_synced_trial_ids = {
                    match.group("trial_id")
                    for res in non_existing_results
                    if (match := RE_GET_TRIAL_ID.search(res.path))
                }
                non_synced_trials = [trial for trial in trials if trial.trial_id in not_synced_trial_ids]
                # Fall back to local paths (are they possibly better as primary source as no sync necessary?)
                result_paths.extend(Path(cast("str", trial.local_path)) / "wandb" for trial in non_synced_trials)
                result_paths = list(filter(lambda p: p.exists(), result_paths))
                logger.info("Added trial.paths to results, now having %d paths", len(result_paths))
        return result_paths

    @staticmethod
    def _parse_trial_id_history(
        trial_id_history: dict[str, str], fork_relationships: dict[str, tuple[str | None, int | None]]
    ) -> None:
        original_trial_id = trial_id_history[
            "original_experiment_key"
        ]  # For wandb this is wrong as we just ust the trial id there
        key_order = sorted(int(key) for key in trial_id_history.keys() if key.isdigit())
        last_parent = None
        fork_relationships.setdefault(original_trial_id, (None, None))
        for i, key in enumerate(key_order):
            trial_id = trial_id_history[str(key)]
            if trial_id == original_trial_id:
                last_parent = original_trial_id
                continue  # already added
            # Due to a bug a wrong parent might be inserted here, making an original trial dependent on a fork
            if i != 0 and ExperimentKey.FORK_SEPARATOR not in trial_id:
                logger.warning(
                    "Unexpected non-forked trial in trial history: %s "
                    "- skipping as we think this is a bug from a previous version",
                    trial_id_history,
                )
                continue
            # We do not know the parent step here, we likely do not need it, just add the key number
            fork_relationships.setdefault(trial_id, (last_parent, None) if last_parent is None else (last_parent, key))
            last_parent = trial_id

    @staticmethod
    def _update_dependencies_from_results_file(
        results_file: Path,
        fork_relationships: dict[str, tuple[str | None, int | None]] | None,
    ) -> dict[str, tuple[str | None, int | None]]:
        """Build fork relationship information from a given results file.

        Args:
            results_file: Path to the results CSV file.
        Returns:
            Dict mapping trial_id to (parent_id, parent_step) tuple.
            Non-forked trials have (None, None).
        """
        if fork_relationships is None:
            fork_relationships = {}
        try:
            with open(results_file, "r") as f:
                for line in f:
                    data = json.loads(line)  # results for this iteration
                    config = data.get("config", None)
                    if config is None:
                        if FORK_FROM in data:
                            config = data  # unexpected but as long as its there
                        else:
                            continue
                    if FORK_FROM not in config:
                        if "experiment_key" in config:
                            if (
                                ExperimentKey.FORK_SEPARATOR in config["experiment_key"]
                                or ExperimentKey.FORK_SEPARATOR in results_file.name
                            ):
                                # can be a fork that has been continued -> check for parents
                                if "trial_id_history" in config:
                                    trial_history: dict[str, str] = config["trial_id_history"]
                                    WandbUploaderMixin._parse_trial_id_history(trial_history, fork_relationships)
                                # before we add a wrong None parent return
                                continue
                            fork_relationships.setdefault(config["experiment_key"], (None, None))
                        # As it has no parent do not really need to bother adding it -> get ID from path?
                        if results_file.name == "result.json":
                            # likely a single result file from wandb upload dir
                            match = RE_GET_TRIAL_ID.search(str(results_file.parent))
                            if match:
                                trial_id = match.group("trial_id")
                                fork_relationships.setdefault(trial_id, (None, None))
                        continue
                    fork_data: ForkFromData = config[FORK_FROM]
                    trial_id = config.get("experiment_key", config[FORK_FROM].get("fork_id_this_trial"))
                    if "parent_fork_id" in fork_data:
                        fork_relationships[trial_id] = (
                            fork_data["parent_fork_id"],
                            fork_data["parent_training_iteration"],
                        )
                    elif "trial_id_history" in config:
                        # Check if there is a trial history
                        trial_history: dict[str, str] = config["trial_id_history"]
                        WandbUploaderMixin._parse_trial_id_history(
                            trial_history,
                            fork_relationships,
                        )
            trial_id_from_file = results_file.stem.split("-")[-1]
            if ExperimentKey.FORK_SEPARATOR in trial_id_from_file and trial_id_from_file not in fork_relationships:
                logger.warning(
                    "Results file points to a forked trial but could not find fork information for it. "
                    "Results file corrupted?: %s",
                    results_file,
                )
                # As a best guess add the last found experiment key as parent
                try:
                    last_found = config["experiment_key"]  # pyright: ignore[reportPossiblyUnboundVariable]
                except (NameError, KeyError):
                    pass
                else:
                    if last_found == trial_id_from_file:
                        # cannot do anything
                        return fork_relationships
                    if (
                        ExperimentKey.FORK_SEPARATOR not in last_found
                        and ExperimentKey.RIGHT_PAD_CHAR in last_found[-1]
                    ):
                        # for non fork parents wandb uses the normal id
                        last_found = data.get("trial_id", last_found)  # pyright: ignore[reportPossiblyUnboundVariable]
                    fork_relationships[trial_id_from_file] = (last_found, data.get("current_step", None))  # pyright: ignore[reportPossiblyUnboundVariable]
        except Exception:
            logger.exception("Failed to parse fork relationships from results file %s", results_file)
        return fork_relationships

    @staticmethod
    def _parse_wandb_fork_relationships(wandb_paths: Sequence[Path]) -> dict[str, tuple[str | None, int | None]]:
        """Parse fork relationship information from wandb directories.

        Returns:
            Dict mapping trial_id to (parent_id, parent_step) tuple.
            Non-forked trials have (None, None).
        """
        fork_relationships: dict[str, tuple[str | None, int | None]] = {}

        found_experiment_files: set[Path] = set()
        checked_results_files: set[Path] = set()
        for wandb_dir in wandb_paths:
            # TODO: use experiment_info_file if available, ONLY on root, not on remote
            # can only use RUN_ID if we are in the same experiment, not some later upload
            experiment_info_files = list(Path(wandb_dir.parent.parent).glob("pbt_fork_data-*.csv"))
            if len(experiment_info_files) == 1:
                experiment_info_file = experiment_info_files[0]
            else:
                # Will be no file if we have no forks.
                logger.debug(
                    "Found %d pbt_fork_data-*.csv files found in %s - expecting one (when not in temp/local dir).%s",
                    len(experiment_info_files),
                    wandb_dir.parent.parent,
                    " Using the first one." if experiment_info_files else "",
                )
                experiment_info_file = experiment_info_files[0] if experiment_info_files else None
            fork_info_file = wandb_dir.parent.parent / "wandb_fork_from.csv"

            if not fork_info_file.exists():
                if (
                    not experiment_info_file
                    or experiment_info_file in found_experiment_files
                    or not experiment_info_file.exists()
                ):
                    continue
                found_experiment_files.add(experiment_info_file)
                fork_info_file = experiment_info_file

            try:
                with open(fork_info_file, "r") as f:
                    lines = f.readlines()
                    # Check header
                    header = [p.strip() for p in lines[0].split(",")]
                    if len(lines) < 2:
                        logger.error(
                            "No fork relationship data found in info file %s - assuming an error happened.",
                            fork_info_file,
                        )
                    # moved parent_id added parent_fork_id at position 1
                    if (
                        len(lines) < 2
                        or not tuple(header[:2]) == tuple(FORK_DATA_KEYS[:2])
                        or (
                            header[0] == FORK_DATA_KEYS[0]
                            and header[1] == "parent_id"
                            and FORK_DATA_KEYS[1] == "parent_fork_id"
                        )
                    ):
                        logger.error(
                            "Unexpected or missing header formatting in fork info file %s: %s", fork_info_file, header
                        )
                        # XXX fall back to slow different method, parse all results files
                        if fork_info_file is experiment_info_file:
                            # Overall experiment
                            json_result_files = list(Path(wandb_dir.parent.parent).glob("**/result*.json"))
                        else:
                            # subdir
                            json_result_files = list(Path(wandb_dir.parent).glob("result*.json"))
                        for json_file in json_result_files:
                            if json_file in checked_results_files:
                                continue
                            checked_results_files.add(json_file)
                            fork_relationships = WandbUploaderMixin._update_dependencies_from_results_file(
                                json_file, fork_relationships
                            )
                        continue
                    iteration_idx = None
                    if "parent_training_iteration" in header:
                        iteration_idx = header.index("parent_training_iteration")
                    for line in lines[1:]:
                        line = line.strip()  # noqa: PLW2901
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            trial_id = parts[0]
                            parent_id = parts[1] if parts[1] != trial_id else None
                            parent_step = None
                            if iteration_idx is not None and len(parts) > iteration_idx:  # global pbt_fork_data format
                                if parts[iteration_idx].isdigit():
                                    parent_step = int(parts[iteration_idx])
                                else:
                                    logger.warning(
                                        "Unexpected format for iteration_idx, expected integer: %s",
                                        parts[iteration_idx],
                                    )
                            elif len(parts) >= 3 and parts[2].isdigit():  # wandb_fork_from.csv format
                                parent_step = int(parts[2])
                            elif len(parts) >= 3:
                                logger.warning("Unexpected format for parent_step, expected integer: %s", parts[2])
                            fork_relationships[trial_id] = (parent_id, parent_step)
                        else:
                            logger.error("Unexpected line formatting, expected trial_id, parent_id: %s", parts)
            except (AssertionError, KeyboardInterrupt):
                raise
            except Exception:
                logger.exception("Failed to parse fork relationships from %s", fork_info_file)

        return fork_relationships

    @staticmethod
    def _extract_trial_id_from_wandb_run(run_dir: Path) -> str:
        """Extract trial ID from wandb offline run directory name."""
        # Extract from directory name pattern like "offline-run-20240101_123456-trial_id" or "run-20240101_123456-trial_id"
        run_name = run_dir.name

        # Match pattern: [offline-]run-YYYYMMDD_hhmmss-<trial_id>
        if run_name.startswith(("offline-run-", "run-")):
            # Find the last dash which should separate the timestamp from trial_id
            parts = run_name.split("-")
            if parts[0] == "offline":
                parts = parts[1:]  # Remove 'offline' part
            if parts[0] == "run":
                parts = parts[1:]  # Remove 'run' part
            if len(parts) >= 1:  # Should have at least [offline], run, timestamp, trial_id
                # The trial_id is everything after the timestamp part
                # Find where the timestamp ends (YYYYMMDD_hhmmss pattern)
                for i, part in enumerate(parts):
                    if "_" in part and len(part) == 15:  # YYYYMMDD_hhmmss format
                        # Everything after this part is the trial_id
                        if i + 1 < len(parts):
                            trial_id = "-".join(parts[i + 1 :])
                            return trial_id
                        break

        # Fallback: use the entire directory name
        logger.warning("Could not extract trial ID from run directory name %s, using full name", run_name)
        return run_name

    def _build_upload_dependency_graph(
        self, trial_runs: list[tuple[str, Path]], fork_relationships: Mapping[str, tuple[str | None, int | None]]
    ) -> list[list[tuple[str, list[Path]]]]:
        """Build dependency-ordered groups for uploading trials.

        Returns:
            List of groups where each group can be uploaded in parallel,
            but groups must be uploaded sequentially (earlier groups before later ones).

            Each group is a list of (trial_id, [run_path1, run_path2, ...]) tuples.

            While it should not happen by construction, in cases of circular dependencies or missing parents,
            all remaining trials are grouped together and uploaded in the same batch.
        """
        # Build adjacency lists for dependencies
        dependents: dict[str, list[str]] = {}  # parent_id -> [child_id1, child_id2, ...]
        dependencies: dict[str, set[str]] = {}  # child_id -> {parent_id1, parent_id2, ...}

        # Create a mapping from trial_id to all paths for that ID
        trial_id_to_run_paths: dict[str, list[Path]] = {}
        for trial_id, run_path in trial_runs:
            if trial_id not in trial_id_to_run_paths:
                trial_id_to_run_paths[trial_id] = []
            trial_id_to_run_paths[trial_id].append(run_path)

        # Initialize dependency tracking, using unique trial IDs
        unique_trial_ids = list(trial_id_to_run_paths.keys())
        for trial_id in unique_trial_ids:
            dependencies[trial_id] = set()
            dependents[trial_id] = []

        # Build dependency graph from fork relationships, which should be complete
        for trial_id, (parent_id, _) in fork_relationships.items():
            if trial_id not in dependencies:
                dependencies[trial_id] = set()
            if parent_id and parent_id in unique_trial_ids:
                dependencies[trial_id].add(parent_id)
                if parent_id not in dependents:
                    logger.warning("Parent ID %s not in trial runs, this should not happen", parent_id)
                    dependents[parent_id] = []
                dependents[parent_id].append(trial_id)

        # Topological sort to create upload groups
        upload_groups: list[list[tuple[str, list[Path]]]] = []
        remaining_trials = set(unique_trial_ids)

        while remaining_trials:
            # Find trials with no remaining dependencies
            # A trial is ready if it has no dependencies, or all its dependencies are not in remaining_trials.
            ready_trials = [
                trial_id
                for trial_id in remaining_trials
                if not dependencies[trial_id] or not (dependencies[trial_id] & remaining_trials)
            ]

            if not ready_trials:
                # Circular dependency or missing parent - add all remaining
                logger.warning(
                    "Circular dependency or missing parents detected in fork relationships. "
                    "Adding remaining trials: %s",
                    remaining_trials,
                )
                ready_trials = list(remaining_trials)
            # Create group for this batch, including all paths for each ready trial
            # Create group for this batch, grouping all paths for each ready trial_id
            group = [
                (trial_id, sorted(trial_id_to_run_paths[trial_id]))
                for trial_id in ready_trials
            ]  # fmt: skip

            upload_groups.append(group)

            # Remove completed trials from remaining and update dependencies
            for trial_id in ready_trials:
                remaining_trials.remove(trial_id)
                # Remove this trial as a dependency for others
                for dependent_id in dependents[trial_id]:
                    dependencies[dependent_id].discard(trial_id)

        return upload_groups

    def _stop_monitor(self):
        from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor  # noqa: PLC0415

        if not WandbRunMonitor.is_remote_monitor_running():
            return

        WandbRunMonitor.get_remote_monitor(project="stop_monitor_cleanup", num_cpus=0).cleanup.remote()  # pyright: ignore[reportFunctionMemberAccess]

    def _start_monitor_safe(self, project: Optional[str] = None, entity: Optional[str] = None) -> bool:
        """
        Starts the WandbRunMonitor actor safely, catching ActorDiedError.

        Returns:
            bool: True if the monitor was started successfully, False otherwise.
        """
        try:
            self._monitor = self._start_monitor(project=project, entity=entity, stacklevel=3)
            if self._monitor is None:
                return False
        except ray.exceptions.ActorDiedError:
            self._monitor = None
            # TODO: Maybe kill actor to allow reuse.
            return False
        except Exception:
            # could be missing project
            logger.exception("Failed to start WandbRunMonitor actor for unknown reason.")
            self._monitor = None
            return False
        return True

    def _start_monitor(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        stacklevel=2,
    ) -> ActorProxy[_WandbRunMonitor] | None:
        """
        Gets or starts the WandbRunMonitor actor to monitor parent runs of forked trials.
        Raises:
            ray.exceptions.ActorDiedError: If the WandbRunMonitor actor is already dead / could not start-
        """
        if os.environ.get("RAY_UTILITIES_NO_MONITOR", "0") == "1":
            return None
        if self._monitor is not None:
            return self._monitor
        if self.project is None:
            raise ValueError("Cannot start WandbRunMonitor without wandb project name set.")
        ImportantLogger.important_info(logger, "Starting WandbRunMonitor actor...", stacklevel=2)

        from ray_utilities import get_runtime_env  # noqa: PLC0415
        from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor  # noqa: PLC0415

        actor_options = {"runtime_env": get_runtime_env()}

        try:
            self._monitor = WandbRunMonitor.get_remote_monitor(
                project=self.project, num_cpus=1, actor_options=actor_options, entity=entity
            )
            if not ray.get(self._monitor.is_initialized.remote()):  # pyright: ignore[reportFunctionMemberAccess]
                _init_future = self._monitor.initialize.remote()  # pyright: ignore[reportFunctionMemberAccess]
                try:
                    ray.get(_init_future, timeout=10)
                except ray.exceptions.GetTimeoutError:
                    # if there is a serious exception during init it will be raised now
                    logger.debug("Timed out while starting WandbRunMonitor actor.")
                ImportantLogger.important_info(
                    logger,
                    "Started WandbRunMonitor actor to track parent runs of forked trials.",
                    stacklevel=stacklevel,
                )
        except ray.exceptions.ActorDiedError as e:
            logger.error("Failed to start WandbRunMonitor actor:\n%s", e.error_msg)
            raise
        return self._monitor

    def verify_wandb_uploads(
        self,
        experiment_id: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        *,
        single_experiment: Optional[str] = None,
        verbose: int = 10,
        run_per_page: int = 32,
        experiment_results: Optional[dict[str, dict[str, Any]]] = None,
    ) -> FailureDictType:
        if output_dir is None:
            # might be S3 bucket
            output_dir = os.environ.get("RAY_UTILITIES_STORAGE_PATH", "./outputs/experiments/")
        if str(output_dir).startswith("s3://"):
            logger.error("S3 lookup not yet supported for wandb verification.")
            return {}
        logger.info("Verifying wandb uploads for experiment_key=%s", experiment_id or get_run_id())
        return verify_wandb_runs(
            project=self.project if self.project else "",
            output_dir=output_dir,
            experiment_id=experiment_id or get_run_id(),
            single_experiment=single_experiment,
            verbose=verbose,
            run_per_page=run_per_page,
        )

    def __ray_shutdown__(self):
        self.__del__()

    def __del__(self):
        # do not clean on_experiment_end as we want to access it with Setup classes as well afterwards
        try:
            if getattr(self, "_monitor", None) is not None:
                self._monitor.cleanup.remote()  # pyright: ignore[reportOptionalMemberAccess, reportFunctionMemberAccess]
                self._monitor.__ray_terminate__.remote()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
                self._monitor = None
        except KeyboardInterrupt:
            WandbUploaderMixin.__del__(self)  # need to make sure we clean monitor, do not go back to self


class VerificationFailure(str, Enum):
    NOT_GIVEN = "error not specified"
    EXCEPTION = "exception occurred"
    METRIC_MISMATCH = "metric mismatch"
    # History or not found
    ONLINE_HISTORY_INCOMPLETE = "online history incomplete"
    OFFLINE_HISTORY_BROKEN = "offline history broken"
    NO_OFFLINE_HISTORY_FOUND = "no offline history found"
    NO_ONLINE_RUN_FOUND = "no online run found"
    # Experiment level, e.g. no run has enough steps
    EXPERIMENT_INCOMPLETE = "experiment incomplete"

    def __str__(self) -> str:
        return self.value


class _FailureTuple(NamedTuple):
    metric: str
    offline_value: Any
    online_value: Any
    rel_difference: float = float("nan")
    type: VerificationFailure = VerificationFailure.NOT_GIVEN

    @property
    def minor(self):
        return abs(self.rel_difference) < 0.05


class RunWithVerificationFailures:
    def __init__(
        self,
        run_id: str,
        project: Optional[str] = None,
        group: Optional[str] = None,
        *,
        name: str,
        experiment_level: bool = False,
    ):
        self.id = run_id
        self.project = project
        self.group = group
        self._name = name
        self._experiment_level = experiment_level

    @property
    def url(self):
        return f"{self.project}/{self.id} - group {self.group}" if self.group else f"{self.project}/{self.id}"

    def __str__(self):
        if not self._experiment_level:
            return f"{self._name}(run_id={self.id}, project={self.project}, group={self.group})"
        return f"{self._name}[Experiment](run_id={self.id}, project={self.project}, group={self.group}, experiment_level=True)"  # noqa: E501


class RunNotFound(RunWithVerificationFailures):
    def __init__(
        self, run_id: str, project: Optional[str] = None, group: Optional[str] = None, *, name: Optional[str] = None
    ):
        super().__init__(run_id, project=project, group=group, name=name or "RunNotFound")


def default_experiment_validator(
    experiment_data: dict[str, Any], online_history_data: dict[str, Any], trial_dir_max_step: dict[str, Any]
) -> tuple[_FailureTuple, ...] | None:
    if len(experiment_data) != len(online_history_data):
        fail1 = _FailureTuple(
            metric="Number of runs does not match",
            offline_value=len(experiment_data),
            online_value=len(online_history_data),
            type=VerificationFailure.ONLINE_HISTORY_INCOMPLETE,
        )
    else:
        fail1 = None

    trial_failures = []
    for tdir, max_step in trial_dir_max_step.items():
        if max_step is None:
            continue
        if max_step < MIN_TOTAL_STEPS:
            trial_failures.append(
                _FailureTuple(
                    metric=f"Trial dir {tdir} has less than 1M steps",
                    offline_value=max_step,
                    online_value="N/A",
                    type=VerificationFailure.OFFLINE_HISTORY_BROKEN,
                )
            )
            break
    trial_step_failures: tuple[_FailureTuple, ...] = tuple(trial_failures)

    if any(exp["current_step"] > MIN_TOTAL_STEPS for exp in experiment_data.values()):
        return (fail1, *trial_step_failures) if fail1 else (trial_step_failures or None)
    try:
        max_off_value = max(exp["current_step"] for exp in experiment_data.values())
    except ValueError:
        # empty sequence
        max_off_value = "empty sequence - no data"
    fail2 = _FailureTuple(
        metric="No experiment with > 1M steps",
        offline_value=max_off_value,
        online_value="max(offline, online) checked",
        type=VerificationFailure.EXPERIMENT_INCOMPLETE,
    )
    return (fail2, fail1, *trial_step_failures) if fail1 else (fail2, *trial_step_failures)


def verify_wandb_runs(
    *,
    project: str,
    entity: Optional[str] = None,
    experiment_id: str,
    output_dir: Optional[str | Path] = None,
    single_experiment: Optional[str] = None,
    verbose: int = 10,
    run_per_page: int = 32,
    group_glob: str = "*",
    experiment_results: Optional[dict[str, dict[str, Any]]] = None,
    experiment_validator: Optional[
        Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None | tuple[_FailureTuple, ...]]
    ] = default_experiment_validator,
    use_tqdm: bool | None = None,
) -> FailureDictType:
    if output_dir is None:
        output_dir = os.environ.get("RAY_UTILITIES_STORAGE_PATH", "./outputs/experiments/")
    api = wandb_api()
    if entity is None:
        entity = api.default_entity
    # get all runs of the project with the given experiment key
    runs: Runs | Sequence[Run]
    if single_experiment:
        try:
            try:
                run = api.run(f"{entity}/{project}/{single_experiment}")
            except ValueError:
                run = api.run(f"{entity}/{project.removesuffix('-PPO').removesuffix('-DQN')}/{single_experiment}")
        except wandb.errors.CommError as e:
            if "Could not find run" in str(e):
                logger.error(
                    "Could not find wandb run for project %s, entity %s with run id %s",
                    project,
                    entity,
                    single_experiment,
                )
                if ".parent" in single_experiment:
                    logger.warning("Tried to verify a .parent.json file")
                else:
                    return {
                        RunNotFound(
                            single_experiment, project=project, group=group_glob if group_glob != "*" else None
                        ): Exception("No corresponding online wandb run found.")
                    }
            raise
        runs = [run]
    else:
        try:
            runs = api.runs(
                f"{entity}/{project}", filters={"config.experiment_id": experiment_id}, per_page=run_per_page
            )
            # NOTE: Runs is async on demand iterator, check if project is old or new layout with - Algorithm
            runs[0]  # noqa
        except ValueError:
            runs = api.runs(
                f"{entity}/{project.removesuffix('-PPO').removesuffix('-DQN')}",
                filters={"config.experiment_id": experiment_id},
                per_page=run_per_page,
            )
    verify_results: FailureDictType = {}
    logged_tb_once = False
    # Check offline data
    # Get runs in output_dir
    offline_run_ids = set()
    not_all_runs_complete = None
    if output_dir is not None:
        # Supports tmpdir with driver_artifacts subdir
        offline_results = list(Path(output_dir).glob("*" + experiment_id + "/**/result*.json"))

        if len(offline_results) == 0 and not next(Path(output_dir).glob("*" + experiment_id), None):
            # output_dir could already be the experiment_dir, or the run has crashed very early then the parent dir exists.
            # TODO: With the introduction of project/group/subdir this does not work anymore
            # NOTE: output_dir might be changed to backup dir!
            if single_experiment:
                output_dir, offline_results = find_experiment_dir(
                    output_dir,
                    "*"
                    + experiment_id
                    + (
                        f"/**/result*{single_experiment}*.json"
                        if ExperimentKey.FORK_SEPARATOR in single_experiment
                        else "result.json"
                    ),
                    project=project,
                    group_glob=group_glob,
                )
            else:
                output_dir, offline_results = find_experiment_dir(
                    output_dir, "*" + experiment_id + "/**/result*.json", project=project, group_glob=group_glob
                )
            if not offline_results:
                logger.error(
                    "No offline results found for experiment_id %s or project %s in %s or %s. "
                    "Checking all subdirs, this can be very slow and might not be successful!",
                    experiment_id,
                    project,
                    output_dir,
                    os.environ.get("RAY_UTILITIES_BACKUP_STORAGE_PATH", "<no backup path set>"),
                )
                if sys.argv[0] in ("", "upload_wandb.py"):
                    # TODO: input does not work with the parallel upload
                    try:
                        choice = input(f"\ncheck all subdirs of {output_dir} for (y/n/path of {experiment_id}):\n")
                        if choice.lower() == "y":
                            offline_results = list(Path(output_dir).glob("**/result*.json"))
                        elif Path(choice).exists():
                            offline_results = list(Path(choice).glob("**/result*.json"))
                    except EOFError:
                        # non-interactive
                        logger.info("No input available, skipping full subdir search.")
        not_all_runs_complete = False
        offline_results_without_parent = sum(1 for path in offline_results if "parent" not in path.name)
        trial_dirs = {p.parent for p in offline_results}
        # For each trial dir we need one experiment that has been trained until the end >1.1M steps
        # TODO: still some run might still be incomplete. For each trial dir we need one run trained until end
        if not single_experiment and offline_results_without_parent != len(runs):
            logger.error(
                "Offline results count %d does not match wandb runs %d", offline_results_without_parent, len(runs)
            )
            not_all_runs_complete = True
        elif not single_experiment and verbose > 2:
            logger.info("🟢 Number of offline runs to online runs match")
        elif single_experiment and len(offline_results) != 1:
            if ExperimentKey.FORK_SEPARATOR in run.id:  # pyright: ignore[reportPossiblyUnboundVariable]
                offline_results = [
                    path
                    for path in offline_results
                    if run.id in path.stem  # pyright: ignore[reportPossiblyUnboundVariable]
                ]
            else:
                offline_results = [path for path in offline_results if path.name == "result.json"]
            if len(offline_results) != 1:
                logger.error(
                    "🔴 Found %d offline results for single experiment %s: %s",
                    len(offline_results),
                    single_experiment,
                    offline_results,
                )
        run_id_to_trial_map = {}
        if single_experiment:
            offline_run_ids = {run.id}  # pyright: ignore[reportPossiblyUnboundVariable]
            if offline_results:
                run_id_to_trial_map[run.id] = next(iter(offline_results)).parent  # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            unknown_count = 0
            for offline_path in offline_results:
                if offline_path.name == "result.json":
                    match = RE_GET_TRIAL_ID.search(str(offline_path.parent.name))
                    if match:
                        run_id = match["trial_id"]
                    else:
                        logger.error("Could not extract trial_id from offline path %s", offline_path)
                        run_id = f"unknown-{unknown_count}"
                        unknown_count += 1
                elif offline_path.name.endswith("patched.json"):
                    continue  # assume there is a normal result file as well
                else:
                    run_id = offline_path.stem.rsplit("-", 1)[-1]
                offline_run_ids.add(run_id)
                run_id_to_trial_map[run_id] = offline_path.parent
    else:
        logger.warning("No output_dir provided, cannot check for offline wandb data.")
        run_id_to_trial_map = {}
    all_experiment_results = experiment_results if experiment_results is not None else {}
    online_history_data = {k: {} for k in all_experiment_results.keys()}
    trial_dir_max_step = {k: {} for k in all_experiment_results.keys()}
    if use_tqdm is None:
        use_tqdm = sys.argv[0] in ("", "upload_wandb.py")
    for run in tqdm(runs, desc=f"Verifying {experiment_id}", unit="runs", disable=not use_tqdm):
        if run.id not in offline_run_ids:
            logger.warning("Got unexpected online wandb run without offline data: %s", run.id)
        offline_run_ids.discard(run.id)
        try:
            failures = verify_wandb_run_history(
                run=run,
                output_dir=output_dir,
                verbose=verbose,
                experiment_data=all_experiment_results.setdefault(experiment_id, {}),
                online_history_data=online_history_data.setdefault(experiment_id, {}),
            )
            verify_results[run] = failures
            if run.id in run_id_to_trial_map:
                trial_dir_max_step.setdefault(experiment_id, {})
                trial_dir_max_step[experiment_id][run_id_to_trial_map[run.id]] = max(
                    trial_dir_max_step[experiment_id].get(run_id_to_trial_map[run.id], 0),
                    all_experiment_results[experiment_id][run.id]["current_step"],
                    online_history_data[experiment_id][run.id].current_step.max(),
                )
            else:
                logger.warning("No trial dir found for run id %s to update max step.", run.id)
        except BdbQuit:
            raise
        except Exception as e:  # noqa: PERF203
            if not logged_tb_once:
                logger.exception("Failed to verify wandb run %s", run.id)
                logged_tb_once = True
            else:
                logger.error("Failed to verify wandb run %s: %s", run.id, str(e))
            verify_results[run] = e
    if not verify_results:
        logger.warning(
            "No wandb runs found for project %s, entity %s with experiment_key %s",
            project,
            entity,
            experiment_id,
        )
        verify_results = {
            RunWithVerificationFailures(experiment_id, project, experiment_level=True, name="NoRunsFound"): [
                _FailureTuple("no offline/online runs found", 0, 0, type=VerificationFailure.NO_OFFLINE_HISTORY_FOUND)
            ]
        }  # noqa: E501
    if len(offline_run_ids) > 0:
        # If we check just a single run ignore
        try:
            experiment_dir = next(Path(output_dir).glob("*" + experiment_id))
        except StopIteration:
            logger.error("Could not find any offline data in %s for experiment_id %s", output_dir, experiment_id)
            experiment_dir = None
        logger.error(
            "Found %d offline wandb runs in %s without corresponding online run: %s",
            len(offline_run_ids),
            experiment_dir if experiment_dir else "Path(output_dir).glob('*' + experiment_id)",
            offline_run_ids,
        )
        verify_results.update(
            {
                RunNotFound(run_id, project=project, group=group_glob if group_glob != "*" else None): Exception(
                    "No corresponding online wandb run found."
                )
                for run_id in offline_run_ids
            }
        )
    for run, failure in verify_results.items():
        if isinstance(failure, Exception):
            logger.error("Verification for wandb run %s (%s) failed with exception: %s", run.id, run.url, str(failure))
        elif failure:
            if all(f.minor for f in failure):
                logger.warning("Wandb run %s (%s) history has minor discrepancies: %s", run.id, run.url, failure)
            else:
                logger.error("Wandb run %s (%s) history verification failed: %s", run.id, run.url, failure)
        else:
            ImportantLogger.important_info(logger, "Wandb run %s (%s) history verified successfully.", run.id, run.url)
    if not single_experiment and all_experiment_results and experiment_validator:
        experiment_level_failure = experiment_validator(
            all_experiment_results[experiment_id], online_history_data[experiment_id], trial_dir_max_step[experiment_id]
        )
        if experiment_level_failure:
            logger.error(
                "Experiment-level validation for experiment_id %s failed: %s",
                experiment_id,
                experiment_level_failure,
            )
            for i, failure in enumerate(experiment_level_failure):
                verify_results[
                    RunWithVerificationFailures(
                        experiment_id, project, name=f"ExperimentLevel {i}", experiment_level=True
                    )
                ] = [failure]
        elif not_all_runs_complete:
            verify_results[
                RunWithVerificationFailures(experiment_id, project, name="ExperimentLevel", experiment_level=True)
            ] = [
                _FailureTuple(
                    "Not all runs completed",
                    len(all_experiment_results[experiment_id]),
                    len(online_history_data[experiment_id]),
                    type=VerificationFailure.EXPERIMENT_INCOMPLETE,
                )
            ]
        else:
            ImportantLogger.important_info(
                logger,
                "Experiment-level validation for experiment_id %s passed successfully.",
                experiment_id,
            )
    return verify_results


def find_experiment_dir(
    output_dir: str | Path, glob_pattern: str, *, project: str, group_glob: str = "*"
) -> tuple[Path, list[Path]]:
    """
    Args:
        glob_pattern: e.g. "*" + experiment_id
    """
    experiment_paths = list(Path(output_dir).glob(glob_pattern))
    if not experiment_paths:
        # check with project group pattern
        experiment_paths = list(Path(output_dir).glob(f"{project}/{group_glob}/{glob_pattern}"))
    if not experiment_paths:
        # Maybe unsorted
        experiment_paths = list(Path(output_dir).glob(f"{project}/{glob_pattern}"))
    if (
        not experiment_paths
        and (backup_dir := os.environ.get("RAY_UTILITIES_BACKUP_STORAGE_PATH"))
        and backup_dir != output_dir
    ):
        experiment_paths = list(Path(backup_dir).glob(glob_pattern))
        if not experiment_paths:
            # check with project group pattern
            experiment_paths = list(Path(backup_dir).glob(f"{project}/{group_glob}/{glob_pattern}"))
        if not experiment_paths:
            # Maybe unsorted
            experiment_paths = list(Path(backup_dir).glob(f"{project}/{glob_pattern}"))
        return Path(backup_dir), experiment_paths
    return Path(output_dir), experiment_paths


__logged_msg_for_path: set[tuple[str, str | Path]] = set()


@overload
def verify_wandb_run_history(
    *,
    output_dir: Optional[str | Path] = None,
    run: Run,
    verbose: int = 10,
    experiment_data: Optional[dict[str, dict[str, float]]] = None,
    online_history_data: Optional[dict[str, pd.DataFrame]] = None,
) -> list[_FailureTuple]: ...


@overload
def verify_wandb_run_history(
    project: str,
    run_id: str,
    entity: Optional[str] = None,
    *,
    experiment_id: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    run: Optional[Run] = None,
    verbose: int = 10,
    experiment_data: Optional[dict[str, dict[str, float]]] = None,
    online_history_data: Optional[dict[str, pd.DataFrame]] = None,
) -> list[_FailureTuple]: ...


def verify_wandb_run_history(
    project: Optional[str] = None,
    run_id: Optional[str] = None,
    entity: Optional[str] = None,
    *,
    experiment_id: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    run: Optional[Run] = None,
    verbose: int = 10,
    group_glob: str = "*",
    experiment_data: Optional[dict[str, dict[str, float]]] = None,
    online_history_data: Optional[dict[str, pd.DataFrame]] = None,
) -> list[_FailureTuple]:
    """Verify the online wandb history against the offline JSON file.

    The "result*run_id.json" file is read from the ``output_dir`` which should be the parent folder for all experiments.
    It is assumed that the `run_id`
    stored in output_dir. The last logged current_step and training_iteration

    Args:
        project: Wandb project name.
        run_id: The Wandb run ID.
        entity: Wandb entity (user or team). If None, uses default entity.
        output_dir: Directory where offline wandb experiment data is stored.
            Defaults to RAY_UTILITIES_STORAGE_PATH or "./outputs/experiments/".
        experiment_id: The experiment ID will be extracted from the run_id if possible.
            If provided will be checked for equality. See :const:``RUN_ID``.
        run: Optional wandb Run object. If not provided, will fetch from Wandb API.
        experiment_data: Gather key metrics of this run, when comparing multiple runs this can be used to
            to e.g. detect failures in incomplete experiments, e.g. no run with steps > 1M.
    """
    if output_dir is None:
        output_dir = os.environ.get("RAY_UTILITIES_STORAGE_PATH", "./outputs/experiments/")
        # TODO: can point to S3 bucket
    if run:
        if run_id is not None and run.id != run_id:
            raise ValueError(f"Provided run ID {run_id} does not match run.id {run.id}")
        run_id = cast("str", run.id)
        if isinstance(run.config, str):
            config = json.loads(run.config)
            assert isinstance(config, dict)
            run._attrs["config"] = config
        assert isinstance(run.config, dict)
        extracted_experiment_id = cast("str | dict[Literal['value'], Any]", run.config["run_id"])
        if isinstance(extracted_experiment_id, dict):
            extracted_experiment_id = extracted_experiment_id["value"]
        assert isinstance(extracted_experiment_id, str)
        if experiment_id and experiment_id != extracted_experiment_id:
            raise ValueError(
                f"Provided experiment_id {experiment_id} "
                "does not match extracted experiment ID {extracted_experiment_id} from run config"
            )
        experiment_id = extracted_experiment_id
        project = project or run.project
        group_glob = run.group if run.group and group_glob == "*" else group_glob
        # NOTE: We might change : -> - in group names or directories, replace with wildcard
        group_glob = group_glob.replace(":", "?").replace("=", "?")
    else:
        if run_id is None or project is None:
            raise ValueError("Either run and/or run_id and project must be provided to verify wandb history.")
        if ExperimentKey.RUN_ID_SEPARATOR not in run_id:
            if not experiment_id:
                raise ValueError(
                    f"Cannot extract experiment ID from run_id {run_id}. Please provide experiment_id argument."
                )
        else:
            extracted_experiment_id = run_id.split("X", 1)[0]
            if experiment_id and experiment_id != extracted_experiment_id:
                raise ValueError(
                    f"Provided experiment_id {experiment_id} does not match extracted experiment ID "
                    f"{extracted_experiment_id} from run_id {run_id}"
                )
            experiment_id = extracted_experiment_id

    _found_dir, experiment_paths = find_experiment_dir(
        output_dir, f"*{experiment_id}", project=project, group_glob=group_glob
    )

    if not experiment_paths:
        # When using a local path we have already the correct dir
        if (experiment_id, output_dir) not in __logged_msg_for_path:
            logger.info(
                "Did not find experiment paths for id %s in %s. Assuming correct path was given",
                experiment_id,
                output_dir,
            )
            __logged_msg_for_path.add((experiment_id, output_dir))
        experiment_paths = [output_dir]
    if len(experiment_paths) > 1:
        logger.warning(
            "Multiple experiment paths found for experiment ID %s: %s. Using the first one.",
            experiment_id,
            experiment_paths,
        )
    experiment_path = Path(experiment_paths[0])
    if not experiment_path.exists():
        raise FileNotFoundError(str(experiment_path))
    # TODO: If the run_id a fork is created "from_checkpoint" this does fail
    if ExperimentKey.FORK_SEPARATOR in run_id or (run and FORK_FROM in run.config):
        # Limit glob operation to 60 seconds using a thread
        # Update progress_files iteratively as files are found

        progress_files = []
        stop_event = threading.Event()

        def glob_files():
            nonlocal progress_files
            # Use generator to update progress_files as files are found
            for f in experiment_path.glob(f"*/result*{run_id}.json"):
                if stop_event.is_set():
                    break
                progress_files.append(f)

        glob_thread = threading.Thread(target=glob_files)
        glob_thread.start()
        glob_thread.join(timeout=60)
        if glob_thread.is_alive():
            stop_event.set()
            logger.error(
                "Timeout: globbing for progress files exceeded 60 seconds for run ID %s in %s "
                "pattern */result*{%s}.json",
                run_id,
                experiment_path,
                run_id,
            )
            # In Python, threads cannot be forcibly killed, so we just log and proceed.
            # Return whatever files have been found so far.
        if len(progress_files) == 0 and "artifacts" in experiment_path.parts:
            progress_files = list((experiment_path).glob(f"driver_artifacts/**/result*{run_id}.json"))
    else:
        # if it is not a fork then the normal result.json
        progress_files = list(experiment_path.glob(f"*id={run_id}*/result.json"))
        if len(progress_files) == 0 and "artifacts" in experiment_path.parts:
            progress_files = list((experiment_path).glob(f"driver_artifacts/*id={run_id}*/result.json"))
    if len(progress_files) != 1:
        logger.error(
            "Expected exactly one progress file for run ID %s neither in output dir %s nor (potential) backup: %s, found %d: %s"
            "\n- Was it moved to backup and restored from there?",
            run_id,
            output_dir,
            experiment_path,
            len(progress_files),
            progress_files,
        )
        if not progress_files:
            logger.error("Cannot verify wandb history without offline progress file.")
            return [
                _FailureTuple(
                    "Error: No offline history found",
                    float("nan"),
                    float("nan"),
                    type=VerificationFailure.NO_OFFLINE_HISTORY_FOUND,
                )
            ]

    with open(progress_files[0], "r") as f:
        records = [json.loads(line) for line in f]
    flat_records = [dict(tree.flatten_with_path(rec)) for rec in records]

    # Find all unique column keys
    all_keys = set()
    for rec in flat_records:
        all_keys.update(rec.keys())

    # Ensure all records have all keys (fill missing with None)
    maxlen = max(len(k) for k in all_keys) if len(all_keys) > 0 else 0

    for rec in flat_records:
        for key in all_keys:
            padded_key = tuple(list(key) + [""] * (maxlen - len(key)))
            rec.setdefault(key, None)
            if key != padded_key:
                rec[padded_key] = rec.pop(key)

    # Convert to DataFrame
    offline_data = pd.DataFrame(flat_records)

    # Set MultiIndex columns
    try:
        offline_data.columns = pd.MultiIndex.from_tuples(cast("list[tuple[str, ...]]", offline_data.columns))
    except TypeError as e:
        # likely empty list
        return [
            _FailureTuple(
                str(e),
                float("nan"),
                float("nan"),
                type=VerificationFailure.EXCEPTION,
            )
        ]

    last_step = offline_data["current_step"].iloc[-1]
    last_iteration = offline_data["training_iteration"].iloc[-1]

    if run is None:
        api = wandb_api()
        if entity is None:
            entity = api.default_entity
        run = cast("Run", api.run(f"{entity}/{project}/{run_id}"))
    online_history = cast(
        "pd.DataFrame", run.history(samples=2000, keys=["current_step", "training_iteration"], pandas=True)
    )
    failures: list[_FailureTuple] = []
    if len(online_history) == 0:
        # data incomplete
        logger.error(
            "No online wandb history data found for run %s/%s%s. Last server step %s, offline step %s",
            project,
            "" if group_glob in ("*", "", None) else group_glob + "/",
            run_id,
            run.lastHistoryStep,
            last_iteration,
        )
        failures.append(
            _FailureTuple(
                "No online history found",
                last_iteration,
                run.lastHistoryStep,
                type=VerificationFailure.NO_ONLINE_RUN_FOUND,
            )
        )
        # TODO: Could clean sync marker, could check local dir in /tmp
        return failures
    if len(online_history) != len(offline_data) and len(online_history) < 2000:
        # if it is a forked run skip until fork_point
        if FORK_FROM in run.config:
            fork_point = run.config[FORK_FROM]["parent_training_iteration"]
            if len(online_history) != len(offline_data[offline_data["training_iteration"] > fork_point]):
                logger.error(
                    "❌ Mismatch in number of history entries for forked run %s after fork at iteration %d: "
                    "offline %d vs online %d. %s",
                    run_id,
                    fork_point,
                    len(offline_data[offline_data["training_iteration"] > fork_point]),
                    len(online_history),
                    "offline history broken"
                    if len(online_history) > len(offline_data)
                    else "online history incomplete",
                )
                failures.append(
                    _FailureTuple(
                        "num_history_entries_after_fork",
                        len(offline_data[offline_data["training_iteration"] > fork_point]),
                        len(online_history),
                        type=(
                            VerificationFailure.OFFLINE_HISTORY_BROKEN
                            if len(online_history) > len(offline_data)
                            else VerificationFailure.ONLINE_HISTORY_INCOMPLETE
                        ),
                    )
                )
        else:
            logger.error(
                "❌ Mismatch in number of history entries for run %s: offline %d vs online %d",
                run_id,
                len(offline_data),
                len(online_history),
            )
            failures.append(
                _FailureTuple(
                    "num_history_entries",
                    len(offline_data),
                    len(online_history),
                    type=(
                        VerificationFailure.OFFLINE_HISTORY_BROKEN
                        if len(online_history) > len(offline_data)
                        else VerificationFailure.ONLINE_HISTORY_INCOMPLETE
                    ),
                )
            )

    last_log_step = online_history.iloc[-1]._step
    online_iterations = online_history.iloc[-1].training_iteration
    online_last_step = online_history.iloc[-1].current_step
    for metric_name, offline_value, online_value in zip(
        ("current_step", "training_iteration"), (last_step, last_iteration), (online_last_step, online_iterations)
    ):
        if offline_value != online_value:
            if isinstance(offline_value, np.number):
                offline_value = offline_value.item()  # noqa: PLW2901
            if isinstance(online_value, np.number):
                online_value = online_value.item()  # noqa: PLW2901
            # relative_diff if possibly
            try:
                rel_diff = round(
                    abs(offline_value - online_value) / max(abs(offline_value), abs(online_value), 1e-8) * 100, 2
                )
            except TypeError:
                rel_diff = float("nan")
            failures.append(
                _FailureTuple(
                    metric_name, offline_value, online_value, rel_diff, type=VerificationFailure.METRIC_MISMATCH
                )
            )
            if rel_diff > 0.5:
                logger.error(
                    "❌ Mismatch in %18s: offline last %8s %s online last %8s (%.1f %%)."
                    "On WandB only logged until %6d step total %3d entries. (run id: %s)",
                    metric_name,
                    offline_value,
                    ">" if offline_value > online_value else "<",
                    online_value,
                    rel_diff,
                    last_log_step,
                    online_history.shape[0],
                    run_id,
                )
            elif verbose > 2:
                logger.warning(
                    "⚠️ Minor mismatch in %18s: offline last %8s vs online last %8s (%.1f %%)."
                    "On WandB only logged until %6d step total %3d entries. (run id: %s)",
                    metric_name,
                    offline_value,
                    online_value,
                    rel_diff,
                    last_log_step,
                    online_history.shape[0],
                    run_id,
                )
        elif verbose > 3:
            logger.debug(
                "🟩 Metric '%s' matches: offline == online: %s",
                metric_name,
                online_value,
            )
        if experiment_data is not None:
            experiment_data.setdefault(run_id, {})[metric_name] = max(offline_value, online_value)
    if online_history_data is not None:
        online_history_data[run_id] = online_history
    return failures
