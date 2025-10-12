"""WandB utilities for callbacks and experiment uploaders."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence, cast

# Note ray is only necessary for the WandbRunMonitor actor
import ray
import ray.exceptions
from wandb import Api

from ray_utilities._runtime_constants import RUN_ID
from ray_utilities.callbacks.upload_helper import AnyPopen, ExitCode, UploadHelperMixin
from ray_utilities.constants import FORK_DATA_KEYS
from ray_utilities.misc import RE_GET_TRIAL_ID, ExperimentKey

if TYPE_CHECKING:
    import wandb
    from ray import tune
    from ray.actor import ActorProxy
    from ray.tune import ResultGrid

    from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor


logger = logging.getLogger(__name__)

_failed_upload_file_lock = threading.Lock()


_wandb_api = None


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
        self._upload_to_trial: dict[AnyPopen, str] = {}
        self._monitor: Optional[ActorProxy[WandbRunMonitor]] = None
        self._history_artifact: dict[str, list[wandb.Artifact]]

    def wandb_upload_results(
        self,
        results: Optional[ResultGrid],
        tuner: Optional[tune.Tuner] = None,
        *,
        wait: bool = True,
        parallel_uploads: int = 5,
    ) -> list[subprocess.Popen] | None:
        """
        Upload wandb's offline folder of the session to wandb, similar to the `wandb sync` shell command

        Args:
            results: The ResultGrid containing the results of the experiment.
            tuner: Optional tuner to get additional trial information.
            wait: If True, waits for the upload to finish before returning.
            parallel_uploads: Number of parallel uploads to by executing :class:`subprocess.Popen`
        """
        logger.info("Uploading wandb offline experiments...")

        # Step 1: Gather all wandb paths and trial information
        wandb_paths: list[Path] = self._get_wandb_paths(results, tuner)
        # FIXME: If this is set it might upload the same directory multiple times
        global_wandb_dir = os.environ.get("WANDB_DIR", None)
        if global_wandb_dir and (global_wandb_dir := Path(global_wandb_dir)).exists():
            wandb_paths.append(global_wandb_dir)
        uploads = self.upload_paths(wandb_paths, wait=wait, parallel_uploads=parallel_uploads)
        return uploads

    def _monitor_check_trial(self, trial_id: str, timeout: float = 40) -> bool | None:
        parent_id = self.fork_relationships.get(trial_id, (None, None))[0]
        if not parent_id:
            # we might check a trial with no parent here
            logger.info("No parent_id found for trial %s, cannot check with monitor", trial_id)
            return None
            # extract parent id from trial_id if possible
            _, fork_data = ExperimentKey.parse_experiment_key(trial_id)
            if fork_data:
                # contains only the pure trial id not the experiment key of the parent
                parent_id = fork_data.get("parent_trial_id")

        self._monitor = self._start_monitor()
        page_visit = self._monitor.visit_run_page.remote(parent_id)  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        done, _ = ray.wait([page_visit], timeout=timeout)
        # try again
        return bool(done)

    def _get_history_artifact_name(
        self, run_id: str, version: Optional[str | int] = "latest", entity: Optional[str] = None
    ) -> str:
        """Get the full name of the history artifact for a given run ID and version."""
        if not self.project:
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
        entity_path = entity + "/" if entity else ""
        artifact = api.artifact(artifact_name)
        run: wandb.Run = api.run(f"{entity_path}{self.project}/{trial_id}")
        artifact_run = artifact.logged_by()
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

    def _check_with_monitor_and_retry(self, process: AnyPopen, timeout=120) -> int:
        logger.info("Process %s failed with returncode %d, checking parent with monitor", process, process.returncode)

        start = time.time()
        trial_id = self._upload_to_trial.get(process, "")
        breakpoint()
        found_before, new_artifact = self._check_for_artifact(trial_id)
        self._monitor_check_trial(trial_id=trial_id, timeout=timeout)
        time.sleep(2)
        found_after, new_artifact_after = self._check_for_artifact(trial_id)
        while not (new_artifact or new_artifact_after) and (time.time() - start) < timeout:
            time.sleep(5)
            found_after, new_artifact_after = self._check_for_artifact(trial_id)
        process_retry = subprocess.Popen(
            ["wandb", "sync", *cast("Iterable[str]", process.args[2:])],  # pyright: ignore[reportIndexIssue]
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )
        end = time.time()
        exit_code = self._failure_aware_wait(
            process_retry,
            timeout=max(20, timeout * 0.2, timeout - (end - start)),
            trial_id=self._upload_to_trial.get(process, ""),
        )
        if exit_code != 0:
            logger.error(
                "Retry of upload for trial %s also failed with exit code %d",
                Path(process.args[2]).name,
                exit_code,  # pyright: ignore[reportArgumentType, reportIndexIssue]
            )
        return exit_code

    def upload_paths(
        self,
        wandb_paths: Sequence[Path],
        trial_runs: Optional[list[tuple[str, Path]]] = None,
        *,
        wait: bool = True,
        parallel_uploads: int = 5,
    ):
        # Step 2: Collect all trial runs with their trial IDs
        if trial_runs is None:
            logger.info("No trial_runs provided, extracting from wandb paths.", stacklevel=2)
            trial_runs = []  # (trial_id, run_dir)

            for wandb_dir in wandb_paths:
                # Find offline run directories
                offline_runs = list(wandb_dir.glob("offline-run-*"))
                if len(offline_runs) > 1:
                    logger.warning("Multiple wandb offline directories found in %s: %s", wandb_dir, offline_runs)

                if not offline_runs:
                    logger.error(
                        "No wandb offline experiments found to upload in %s: %s. ", wandb_dir, list(wandb_dir.glob("*"))
                    )
                    continue

                for run_dir in offline_runs:
                    trial_id = self._extract_trial_id_from_wandb_run(run_dir)
                    if trial_id:
                        trial_runs.append((trial_id, run_dir))
                    else:
                        logger.warning(
                            "Could not extract trial ID from %s, will upload without dependency ordering", run_dir
                        )
                        trial_runs.append((run_dir.name, run_dir))

        if not trial_runs:
            logger.info("No wandb offline runs found to upload.")
            return None

        # Step 3: Parse fork relationships
        self.fork_relationships = self._parse_wandb_fork_relationships(wandb_paths)
        logger.debug("Found %d fork relationships: %s", len(self.fork_relationships), self.fork_relationships)

        # Step 4: Build dependency-ordered upload groups
        upload_groups: list[list[tuple[str, Path]]] = self._build_upload_dependency_graph(
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
                logger.warning(
                    "Continuing %d unfinished wandb uploads from previous gather: %s",
                    len(unfinished_from_past),
                    unfinished_from_past,
                )
                for process in unfinished_from_past:
                    exit_code = self._failure_aware_wait(process, timeout=300)
                    if exit_code in (ExitCode.WANDB_BEHIND_STEP, ExitCode.WANDB_SERVER_ERROR):
                        # use monitor to check on parent, try again
                        # how do I get the parent id?
                        exit_code = self._check_with_monitor_and_retry(process)
                    if exit_code != 0:
                        failed_uploads.append(process)

        for group_idx, group in enumerate(upload_groups):
            logger.info("Uploading group %d/%d with %d trials", group_idx + 1, len(upload_groups), len(group))

            # Wait for previous group to complete before starting next group
            if group_idx > 0:
                logger.info("Waiting for previous upload group to complete...")
                finished_or_failed = []
                for process in uploads:
                    exit_code = self._failure_aware_wait(
                        process, timeout=900, trial_id=self._upload_to_trial.get(process, "")
                    )
                    if exit_code == 0:
                        finished_uploads.add(process)
                    elif self._check_with_monitor_and_retry(process) == 0:
                        finished_uploads.add(process)
                    else:
                        failed_uploads.append(process)
                    finished_or_failed.append(process)
                uploads = [p for p in uploads if p not in finished_or_failed]

            # Upload trials in current group (can be parallel within group)
            for trial_id, run_dir in group:
                # Manage parallel upload limit within group
                if len(uploads) >= parallel_uploads:
                    logger.info(
                        "%d >= %d uploads already in progress waiting for some to finish before starting new ones...",
                        len(uploads),
                        parallel_uploads,
                    )
                # process uploads that are already finished:
                for process in (p for p in uploads if p.poll() is not None):
                    exit_code = self._failure_aware_wait(
                        process, timeout=60, trial_id=self._upload_to_trial.get(process, "")
                    )
                    if exit_code == 0:
                        finished_uploads.add(process)
                    elif self._check_with_monitor_and_retry(process) == 0:
                        finished_uploads.add(process)
                    else:
                        failed_uploads.append(process)
                    uploads.remove(process)
                while len(uploads) >= parallel_uploads:
                    finished_or_failed = set()
                    # Prioritize checking processes that have already finished else oldest first
                    for process in sorted(uploads, key=lambda p: p.poll() is None):
                        exit_code = self._failure_aware_wait(
                            process, timeout=900, trial_id=self._upload_to_trial.get(process, "")
                        )
                        if exit_code == 0:
                            finished_uploads.add(process)
                        elif self._check_with_monitor_and_retry(process) == 0:
                            finished_uploads.add(process)
                        else:
                            failed_uploads.append(process)
                        finished_or_failed.add(process)
                    uploads = [p for p in uploads if p not in finished_or_failed]

                # if the run has a parent we want to check it with the monitor first
                logger.debug("Checking with monitor before uploading trial %s", trial_id)
                if (visit_success := self._monitor_check_trial(trial_id, timeout=40)) is not None:
                    logger.debug("Monitor visit for parent of trial %s was %s", trial_id, visit_success)
                    time.sleep(5)
                logger.info(
                    "Uploading offline wandb run for trial %s from: %s (group %d/%d, trial %d/%d in group)",
                    trial_id,
                    run_dir,
                    group_idx + 1,
                    len(upload_groups),
                    group.index((trial_id, run_dir)) + 1,
                    len(group),
                )
                process = subprocess.Popen(
                    ["wandb", "sync", run_dir.as_posix(), "--append"],
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
        for process in sorted(uploads, key=lambda p: p.poll() is None):
            exit_code = None
            if wait:
                exit_code = self._failure_aware_wait(
                    process, timeout=900, trial_id=self._upload_to_trial.get(process, "")
                )
            if process.poll() is not None:
                if exit_code is None:
                    exit_code = self._report_upload(process)
                if exit_code == 0:
                    finished_uploads.add(process)
                elif self._check_with_monitor_and_retry(process) == 0:
                    finished_uploads.add(process)
                else:
                    failed_uploads.append(process)
                unfinished_uploads.remove(process)
        uploads = []

        if failed_uploads:
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
                    dest = path / f"failed_wandb_uploads-{RUN_ID}.txt"
                    if dest.exists():
                        dest.rename(dest.with_suffix(".txt.old"))
                    shutil.copyfile(failed_file, dest)
                    logger.info("Copied file for failed uploads to %s", dest.resolve())
                except Exception:
                    logger.exception("Failed to copy failed upload file to %s", path)
        if not unfinished_uploads:
            logger.info("All wandb offline runs have been tried to upload.")
        logger.info(
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

    def _update_failed_upload_file(
        self, failed_uploads: Iterable[AnyPopen], file_dir: Path, process_to_trial: Optional[dict[AnyPopen, str]] = None
    ) -> Path:
        with _failed_upload_file_lock:
            failed_file = file_dir / f"failed_wandb_uploads-{RUN_ID}.txt"
            with failed_file.open("a") as f:
                for process in failed_uploads:
                    trial_id = process_to_trial.get(process, "unknown") if process_to_trial else "unknown"
                    formatted_args = (
                        " ".join(map(str, process.args))
                        if not isinstance(process.args, (str, bytes)) and isinstance(process.args, Iterable)
                        else process.args
                    )
                    err = ""
                    if process.stdout:
                        out_left = process.stdout.read()
                        if isinstance(out_left, bytes):
                            out_left = out_left.decode("utf-8")
                        if out_left:
                            err = "\n" + indent(out_left, prefix=" " * 4) + "\n"
                    f.write(f"{trial_id} : {formatted_args}{err}\n")
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
                assert tuner._local_tuner is not None
                trials = (
                    tuner._local_tuner.get_results()._experiment_analysis.trials  # pyright: ignore[reportOptionalMemberAccess]
                )
            except RuntimeError as e:
                if (
                    not tuner._local_tuner or not tuner._local_tuner.get_run_config().callbacks
                ):  # assume there is a logger
                    raise RuntimeError("Cannot get trials as local tuner or callbacks are missing.") from e
                # Import here to avoid circular dependency
                from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback  # noqa: PLC0415

                wandb_cb = next(
                    cb
                    for cb in tuner._local_tuner.get_run_config().callbacks  # pyright: ignore[reportOptionalIterable]
                    if isinstance(cb, AdvWandbLoggerCallback)
                )  # pyright: ignore[reportOptionalIterable]
                trials = wandb_cb._trials
            trial_paths = [Path(trial.local_path) / "wandb" for trial in trials if trial.local_path]
            if len(trial_paths) != len(trials):
                logger.error("Did not get all wandb paths %d of %d", len(trial_paths), len(trials))
            return trial_paths
        result_paths = [Path(result.path) / "wandb" for result in results]  # these are in the non-temp dir
        if tuner is None:
            logger.warning("No tuner provided cannot check for missing wandb paths.")
            return result_paths
        try:
            # compare paths for completeness
            assert tuner._local_tuner is not None
            trials = tuner._local_tuner.get_results()._experiment_analysis.trials
            trial_paths = [Path(trial.local_path) / "wandb" for trial in trials if trial.local_path]
        except Exception:
            logger.exception("Could not get trials or their paths")
        else:
            existing_in_result = sum(p.exists() for p in result_paths)
            existing_in_trial = sum(p.exists() for p in trial_paths)
            if existing_in_result != existing_in_trial:
                logger.error(
                    "Count of existing trials paths did not match %d vs %d: \nResult Paths:\n%s\nTrial Paths:\n%s",
                    existing_in_result,
                    existing_in_trial,
                    result_paths,
                    trial_paths,
                )
            non_existing_results = [res for res in results if not (Path(res.path) / "wandb").exists()]
            # How to get the trial id?
            if non_existing_results:
                not_synced_trial_ids = {
                    match.group("trial_id")
                    for res in non_existing_results
                    if (match := RE_GET_TRIAL_ID.search(res.path))
                }
                non_synced_trials = [trial for trial in trials if trial.trial_id in not_synced_trial_ids]
                result_paths.extend(Path(cast("str", trial.local_path)) / "wandb" for trial in non_synced_trials)
                result_paths = list(filter(lambda p: p.exists(), result_paths))
                logger.info("Added trial.paths to results, now having %d paths", len(result_paths))
        return result_paths

    @staticmethod
    def _parse_wandb_fork_relationships(wandb_paths: Sequence[Path]) -> dict[str, tuple[str | None, int | None]]:
        """Parse fork relationship information from wandb directories.

        Returns:
            Dict mapping trial_id to (parent_id, parent_step) tuple.
            Non-forked trials have (None, None).
        """
        fork_relationships: dict[str, tuple[str | None, int | None]] = {}

        for wandb_dir in wandb_paths:
            # TODO: use experiment_info_file if available
            experiment_info_file = Path(wandb_dir) / f"pbt_fork_data-{RUN_ID}.csv"
            fork_info_file = wandb_dir.parent.parent / "wandb_fork_from.csv"
            if not fork_info_file.exists():
                continue

            try:
                with open(fork_info_file, "r") as f:
                    lines = f.readlines()
                    # Check header
                    header = [p.strip() for p in lines[0].split(",")]
                    assert tuple(header[:2]) == tuple(FORK_DATA_KEYS[:2])
                    assert len(lines) >= 2
                    for line in lines[1:]:
                        line = line.strip()  # noqa: PLW2901
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            trial_id = parts[0]
                            parent_id = parts[1] if parts[1] != trial_id else None
                            parent_step = None
                            if len(parts) >= 3 and parts[2].isdigit():
                                parent_step = int(parts[2])
                            elif len(parts) >= 3:
                                logger.warning("Unexpected format for parent_step, expected integer: %s", parts[2])
                            fork_relationships[trial_id] = (parent_id, parent_step)
                        else:
                            logger.error("Unexpected line formatting, expected trial_id, parent_id: %s", parts)
            except AssertionError:
                raise
            except Exception:
                logger.exception("Failed to parse fork relationships from %s", fork_info_file)

        return fork_relationships

    def _extract_trial_id_from_wandb_run(self, run_dir: Path) -> str:
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
    ) -> list[list[tuple[str, Path]]]:
        """Build dependency-ordered groups for uploading trials.

        Returns:
            List of groups where each group can be uploaded in parallel,
            but groups must be uploaded sequentially (earlier groups before later ones).
        """
        # Build adjacency lists for dependencies
        dependents: dict[str, list[str]] = {}  # parent_id -> [child_id1, child_id2, ...]
        dependencies: dict[str, set[str]] = {}  # child_id -> {parent_id1, parent_id2, ...}

        trial_id_to_run = dict(trial_runs)

        # Initialize dependency tracking, trial_runs might be not complete
        for trial_id, _ in trial_runs:
            dependencies[trial_id] = set()
            dependents[trial_id] = []

        # Build dependency graph from fork relationships, which should be complete
        for trial_id, (parent_id, _) in fork_relationships.items():
            if parent_id and parent_id in trial_id_to_run:
                if trial_id not in dependencies:
                    dependencies[trial_id] = set()
                dependencies[trial_id].add(parent_id)
                dependents[parent_id].append(trial_id)

        # Topological sort to create upload groups
        upload_groups: list[list[tuple[str, Path]]] = []
        remaining_trials = {trial_id for trial_id, _ in trial_runs}

        while remaining_trials:
            # Find trials with no remaining dependencies
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

            # Create group for this batch
            group = [(trial_id, trial_id_to_run[trial_id]) for trial_id in ready_trials]
            upload_groups.append(group)

            # Remove completed trials from remaining and update dependencies
            for trial_id in ready_trials:
                remaining_trials.remove(trial_id)
                # Remove this trial as a dependency for others
                for dependent_id in dependents[trial_id]:
                    dependencies[dependent_id].discard(trial_id)

        return upload_groups

    def _start_monitor(self, project: Optional[str] = None) -> ActorProxy[WandbRunMonitor]:
        """Gets or starts the WandbRunMonitor actor to monitor parent runs of forked trials."""
        if self._monitor is not None:
            return self._monitor
        if self.project is None:
            raise ValueError("Cannot start WandbRunMonitor without wandb project name set.")
        from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor  # noqa: PLC0415

        self._monitor = WandbRunMonitor.get_remote_monitor(project=self.project)
        if not ray.get(self._monitor.is_initialized.remote()):  # pyright: ignore[reportAttributeAccessIssue]
            _init_future = self._monitor.initialize.remote()  # pyright: ignore[reportFunctionMemberAccess]
            try:
                ray.get(_init_future, timeout=10)
                logger.info("Started WandbRunMonitor actor to track parent runs of forked trials.")
            except ray.exceptions.GetTimeoutError:
                # if there is a serious exception during init it will be raised now
                logger.debug("Timed out while starting WandbRunMonitor actor.")
        return self._monitor
