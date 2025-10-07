"""WandB utilities for callbacks and experiment uploaders."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, cast

if TYPE_CHECKING:
    from ray import tune
    from ray.tune import ResultGrid

from ray_utilities._runtime_constants import RUN_ID
from ray_utilities.callbacks.upload_helper import AnyPopen, UploadHelperMixin
from ray_utilities.misc import RE_GET_TRIAL_ID

logger = logging.getLogger(__name__)

_failed_upload_file_lock = threading.Lock()


class WandbUploaderMixin(UploadHelperMixin):
    """Mixin for uploading WandB offline experiments with dependency ordering.

    This mixin provides methods to:
    - Parse fork relationships from wandb directories
    - Build dependency graphs for upload ordering
    - Upload trials in correct order (parents before children)
    """

    _upload_service_name = "wandb"

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

    def upload_paths(
        self,
        wandb_paths,
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
        fork_relationships = self._parse_wandb_fork_relationships(wandb_paths)
        logger.debug("Found %d fork relationships: %s", len(fork_relationships), fork_relationships)

        # Step 4: Build dependency-ordered upload groups
        upload_groups: list[list[tuple[str, Path]]] = self._build_upload_dependency_graph(
            trial_runs, fork_relationships
        )
        logger.debug("Created %d upload groups with dependency ordering", len(upload_groups))

        # Step 5: Upload trials in dependency order
        uploads: list[AnyPopen] = []
        finished_uploads: set[AnyPopen] = set()
        failed_uploads: list[AnyPopen] = []
        total_uploaded = 0
        upload_to_trial: dict[AnyPopen, str] = {}

        for group_idx, group in enumerate(upload_groups):
            logger.info("Uploading group %d/%d with %d trials", group_idx + 1, len(upload_groups), len(group))

            # Wait for previous group to complete before starting next group
            if group_idx > 0:
                logger.info("Waiting for previous upload group to complete...")
                finished_or_failed = []
                for process in uploads:
                    exit_code = self._failure_aware_wait(
                        process, timeout=900, trial_id=upload_to_trial.get(process, "")
                    )
                    if exit_code == 0:
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
                    exit_code = self._failure_aware_wait(process, timeout=60, trial_id=upload_to_trial.get(process, ""))
                    if exit_code == 0:
                        finished_uploads.add(process)
                    else:
                        failed_uploads.append(process)
                    uploads.remove(process)
                while len(uploads) >= parallel_uploads:
                    finished_or_failed = set()
                    # Prioritize checking processes that have already finished else oldest first
                    for process in sorted(uploads, key=lambda p: p.poll() is None):
                        exit_code = self._failure_aware_wait(
                            process, timeout=900, trial_id=upload_to_trial.get(process, "")
                        )
                        if exit_code == 0:
                            finished_uploads.add(process)
                        else:
                            failed_uploads.append(process)
                        finished_or_failed.add(process)
                    uploads = [p for p in uploads if p not in finished_or_failed]

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
                    ["wandb", "sync", run_dir.as_posix()],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # line-buffered
                )
                uploads.append(process)
                upload_to_trial[process] = trial_id
                total_uploaded += 1

        # Handle final completion
        if wait:
            logger.info("Waiting for all wandb uploads to finish...")
        unfinished_uploads = uploads.copy()
        for process in sorted(uploads, key=lambda p: p.poll() is None):
            exit_code = None
            if wait:
                exit_code = self._failure_aware_wait(process, timeout=900, trial_id=upload_to_trial.get(process, ""))
            if process.poll() is not None:
                if exit_code is None:
                    exit_code = self._report_upload(process)
                if exit_code == 0:
                    finished_uploads.add(process)
                else:
                    failed_uploads.append(process)
                unfinished_uploads.remove(process)
        uploads = []

        if failed_uploads:
            logger.error(
                "Failed to upload %d wandb runs:\n%s", len(failed_uploads), "\n".join(map(str, failed_uploads))
            )
            with _failed_upload_file_lock:
                failed_file = Path(f"failed_wandb_uploads-{RUN_ID}.txt")
                with failed_file.open("a") as f:
                    for process in failed_uploads:
                        trial_id = upload_to_trial.get(process, "unknown")
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
                            err = "\n" + indent(out_left, prefix=" " * 4) + "\n"
                        f.write(f"{trial_id} : {formatted_args}{err}\n")
                logger.warning("Wrote details of failed uploads to %s", failed_file.resolve())
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
            return unfinished_uploads
        return None

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

    def _parse_wandb_fork_relationships(self, wandb_paths: list[Path]) -> dict[str, tuple[str | None, int | None]]:
        """Parse fork relationship information from wandb directories.

        Returns:
            Dict mapping trial_id to (parent_id, parent_step) tuple.
            Non-forked trials have (None, None).
        """
        fork_relationships: dict[str, tuple[str | None, int | None]] = {}

        for wandb_dir in wandb_paths:
            fork_info_file = wandb_dir.parent / "wandb_fork_from.txt"
            if not fork_info_file.exists():
                continue

            try:
                with open(fork_info_file, "r") as f:
                    lines = f.readlines()
                    # Check header
                    header = [p.strip() for p in lines[0].split(",")]
                    assert header[:2] == ["trial_id", "parent_id"]
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

        # Initialize dependency tracking
        for trial_id, _ in trial_runs:
            dependencies[trial_id] = set()
            dependents[trial_id] = []

        # Build dependency graph from fork relationships
        for trial_id, (parent_id, _) in fork_relationships.items():
            if parent_id and parent_id in trial_id_to_run:
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
