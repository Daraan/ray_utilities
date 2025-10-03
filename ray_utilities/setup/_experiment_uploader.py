from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Mapping, Optional, TypeAlias, cast

from typing_extensions import Sentinel, TypeVar

from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback
from ray_utilities.comet import CometArchiveTracker
from ray_utilities.misc import RE_GET_TRIAL_ID

# pyright: enableExperimentalFeatures=true


if TYPE_CHECKING:
    import argparse

    from ray import tune
    from ray.tune import ResultGrid

    from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser


logger = logging.getLogger(__name__)

ParserType_co = TypeVar("ParserType_co", bound="DefaultArgumentParser", covariant=True, default="DefaultArgumentParser")
"""TypeVar for the ArgumentParser type of a Setup, bound and defaults to DefaultArgumentParser."""

NamespaceType: TypeAlias = "argparse.Namespace | ParserType_co"  # Generic, formerly union with , prefer duck-type

_ATTRIBUTE_NOT_FOUND = Sentinel("_ATTRIBUTE_NOT_FOUND")


class CometUploaderMixin(Generic[ParserType_co]):
    comet_tracker: CometArchiveTracker | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setup_args: NamespaceType[ParserType_co] | _ATTRIBUTE_NOT_FOUND = getattr(self, "args", _ATTRIBUTE_NOT_FOUND)
        if setup_args is _ATTRIBUTE_NOT_FOUND:
            logger.info(
                "No args attribute found, likely due to `parse_args=False`, "
                "cannot initialize comet tracker. Need to be setup later manually if desired."
            )
            self.comet_tracker = None
        elif setup_args.comet:
            self.comet_tracker = CometArchiveTracker()
        else:
            self.comet_tracker = None

    def comet_upload_offline_experiments(self):
        """Note this does not check for args.comet"""
        if self.comet_tracker is None:
            if not hasattr(self, "args") or str(self.args.comet).lower() in ("false", "none", "0"):  # pyright: ignore[reportAttributeAccessIssue]
                logger.debug("No comet tracker / args.comet defined. Will not upload offline experiments.")
            else:
                logger.warning(
                    "No comet tracker setup but args.comet=%s. Cannot upload experiments. Upload them manually instead.",
                    self.args.comet,  # pyright: ignore[reportAttributeAccessIssue]
                )
            return
        self.comet_tracker.upload_and_move()


class WandbUploaderMixin:
    def wandb_upload_offline_experiments(
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

        # Step 2: Collect all trial runs with their trial IDs
        trial_runs: list[tuple[str, Path]] = []  # (trial_id, run_dir)

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
        logger.info("Found %d fork relationships: %s", len(fork_relationships), fork_relationships)

        # Step 4: Build dependency-ordered upload groups
        upload_groups = self._build_upload_dependency_graph(trial_runs, fork_relationships)
        logger.info("Created %d upload groups with dependency ordering", len(upload_groups))

        # Step 5: Upload trials in dependency order
        uploads: list[subprocess.Popen[bytes]] = []
        finished_uploads: set[subprocess.Popen[bytes]] = set()
        failed_uploads: list[subprocess.Popen[bytes]] = []
        total_uploaded = 0

        for group_idx, group in enumerate(upload_groups):
            logger.info("Uploading group %d/%d with %d trials", group_idx + 1, len(upload_groups), len(group))

            # Wait for previous group to complete before starting next group
            if group_idx > 0:
                logger.info("Waiting for previous upload group to complete...")
                for process in uploads:
                    exit_code = self._report_wandb_upload(process)
                    if exit_code == 0:
                        finished_uploads.add(process)
                    else:
                        failed_uploads.append(process)
                uploads = [p for p in uploads if p not in finished_uploads]

            # Upload trials in current group (can be parallel within group)
            for trial_id, run_dir in group:
                # Manage parallel upload limit within group
                uploads_in_progress = len(uploads) - len(finished_uploads)
                if uploads_in_progress >= parallel_uploads:
                    logger.info(
                        "Waiting for %d wandb uploads to finish before starting new ones.",
                        uploads_in_progress,
                    )
                    for process in uploads:
                        exit_code = self._report_wandb_upload(process)
                        if exit_code == 0:
                            finished_uploads.add(process)
                        else:
                            failed_uploads.append(process)
                    uploads = [p for p in uploads if p not in finished_uploads]

                logger.info("Uploading offline wandb run for trial %s from: %s", trial_id, run_dir)
                process = subprocess.Popen(
                    ["wandb", "sync", run_dir.as_posix()],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                uploads.append(process)
                total_uploaded += 1

        # Handle final completion
        # TODO: Can these two sections be simplified
        if wait:
            logger.info("Waiting for all wandb uploads to finish...")
            for process in uploads:
                exit_code = self._report_wandb_upload(process)
                if exit_code == 0:
                    finished_uploads.add(process)
                else:
                    failed_uploads.append(process)
            uploads = []

            if failed_uploads:
                logger.warning("Failed to upload %d wandb runs", len(failed_uploads))

            logger.info(
                "Uploaded all %d wandb offline runs from %s.",
                total_uploaded,
                results.experiment_path if results else f"local wandb fallback paths: {wandb_paths}",
            )
            return None

        # Report on completed uploads without waiting
        unfinished_uploads = uploads.copy()
        for process in uploads:
            if process.poll() is not None:
                exit_code = self._report_wandb_upload(process)
                if exit_code == 0:
                    finished_uploads.add(process)
                else:
                    failed_uploads.append(process)
                unfinished_uploads.remove(process)

        if not unfinished_uploads:
            logger.info("All wandb offline runs have been uploaded.")
            return None

        logger.info(
            "Uploaded %d wandb offline runs from %s, %d still in progress.",
            total_uploaded,
            results.experiment_path if results else f"local wandb fallback paths: {wandb_paths}",
            len(unfinished_uploads),
        )
        # There are still processes running
        return unfinished_uploads

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
                )  # pyright: ignore[reportOptionalMemberAccess]
            except RuntimeError as e:
                if not tuner._local_tuner or tuner._local_tuner.get_run_config().callbacks:  # assume there is a logger
                    raise RuntimeError("Cannot get trials") from e
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
            trials = tuner._local_tuner.get_results()._experiment_analysis.trials  # pyright: ignore[reportOptionalMemberAccess]
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
            except Exception as e:
                logger.warning("Failed to parse fork relationships from %s: %s", fork_info_file, e)

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

    @staticmethod
    def _report_wandb_upload(process: subprocess.Popen[bytes], run_dir: Optional[Path] = None, *, wait: bool = True):
        """Wait and report the output of a WandB upload process."""
        run_dir = run_dir or Path(process.args[-1])  # pyright: ignore[reportArgumentType, reportIndexIssue]
        exit_code = 0
        if wait:
            process.wait()
        if process.stdout:
            stdout = process.stdout.read()
            msg = stdout if isinstance(stdout, str) else stdout.decode("utf-8")
            print(msg)
            if "error" in msg.lower():  # pyright: ignore[reportOperatorIssue]
                # This likely happens when we want to upload a fork_from where the parent is not yet uploaded
                logger.error("Error during wandb upload of offline run %s: %s", run_dir, msg)
                exit_code = 1
        if process.returncode != 0 or process.stderr:
            stderr = process.stderr.read() if process.stderr else b""
            logger.error(
                "Failed to upload wandb offline run %s with exit code %d. Output: %s",
                run_dir,
                process.returncode,
                stderr.decode("utf-8"),
            )
            exit_code = process.returncode or 1
        return exit_code

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


class ExperimentUploader(WandbUploaderMixin, CometUploaderMixin[ParserType_co]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args: NamespaceType[ParserType_co]

    def upload_offline_experiments(self, results: Optional[ResultGrid] = None, tuner: Optional[tune.Tuner] = None):
        unfinished_wandb_uploads = None
        if self.args.wandb and "upload" in self.args.wandb:
            if results is None:
                logger.error(
                    "Wandb upload requested, but no results provided. This will not upload any offline experiments."
                )
            try:  # if no results (due to a failure) get them in a more hacky way.
                # Do not wait to start uploading to comet.
                unfinished_wandb_uploads = self.wandb_upload_offline_experiments(results, tuner, wait=False)
            except Exception:
                logger.exception("Error while uploading offline experiments to WandB: %s")
        if self.args.comet and "upload" in self.args.comet:
            logger.info("Uploading offline experiments to Comet")
            try:
                self.comet_upload_offline_experiments()
            except Exception:
                logger.exception("Error while uploading offline experiments to Comet")
        failed_runs = []
        if unfinished_wandb_uploads:
            for process in unfinished_wandb_uploads:
                exit_code = self._report_wandb_upload(process, wait=True)
                if exit_code != 0:
                    try:
                        failed_runs.append(" ".join(process.args))  # pyright: ignore[reportArgumentType, reportCallIssue]
                    except TypeError:
                        failed_runs.append(str(process.args))
        if failed_runs:
            logger.error("Failed to upload the following wandb runs. Commands to run:\n%s", "\n".join(failed_runs))
