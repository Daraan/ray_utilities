from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Optional, TypeAlias

import pyarrow.fs
from typing_extensions import Sentinel, TypeVar

from ray_utilities._runtime_constants import COMET_OFFLINE_DIRECTORY
from ray_utilities.callbacks.comet import COMET_FAILED_UPLOAD_FILE, CometArchiveTracker
from ray_utilities.callbacks.upload_helper import ExitCode
from ray_utilities.callbacks.wandb import WandbUploaderMixin, get_wandb_failed_upload_file
from ray_utilities.constants import get_run_id
from ray_utilities.misc import get_trials_from_tuner
from ray_utilities.nice_logger import ImportantLogger
from ray_utilities.setup.tuner_setup import TunerSetup

# pyright: enableExperimentalFeatures=true


if TYPE_CHECKING:
    import argparse

    from ray import tune
    from ray.tune import ResultGrid

    from ray_utilities.callbacks.upload_helper import AnyPopen
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
            return None
        return self.comet_tracker.upload_and_move()


class ExperimentUploader(WandbUploaderMixin, CometUploaderMixin[ParserType_co]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args: NamespaceType[ParserType_co]

    def upload_offline_experiments(
        self,
        results: Optional[ResultGrid] = None,
        tuner: Optional[tune.Tuner] = None,
        *,
        use_tqdm: bool = False,
        skip_synced: bool = True,
    ) -> None:
        unfinished_wandb_uploads = None
        failed_runs: list[str] = []
        failed_processes: list[AnyPopen] = []
        try:
            if self.args.wandb and "upload" in self.args.wandb:
                if results is None:
                    logger.warning(
                        "Wandb upload requested, but no results provided. This will not upload any offline experiments."
                    )
                try:  # if no results (due to a failure) get them in a more hacky way.
                    # Do not wait to start uploading to comet.
                    unfinished_wandb_uploads = self.wandb_upload_results(
                        results, tuner, wait=False, use_tqdm=use_tqdm, skip_synced=skip_synced
                    )
                except Exception:
                    logger.exception("Error while uploading offline experiments to WandB: %s")
            if self.args.comet and "upload" in self.args.comet:
                ImportantLogger.important_info(logger, "Uploading offline experiments to Comet")
                try:
                    self.comet_upload_offline_experiments()
                except Exception:
                    logger.exception("Error while uploading offline experiments to Comet")
            if unfinished_wandb_uploads:
                for process in unfinished_wandb_uploads:
                    exit_code = self._failure_aware_wait(process, timeout=900, trial_id="", upload_service_name="wandb")
                    if exit_code != 0:
                        if exit_code == ExitCode.COMET_ALREADY_UPLOADED:
                            logger.warning(
                                "Experiment was already uploaded to Comet, but zip file was not yet moved. "
                                "Possibility that earlier run did not finish and need to be reuploaded (investigate)"
                            )
                        try:
                            failed_runs.append(" ".join(process.args))  # pyright: ignore[reportArgumentType, reportCallIssue]
                        except TypeError:
                            failed_runs.append(str(process.args))
                        failed_processes.append(process)
        finally:
            # Output failed uploads file contents
            # TODO: experiment path could point to remote S3 path
            experiment_path: str | None = None
            local_experiment_path: str | None = None
            filesystem = None
            if tuner and tuner._local_tuner:
                run_config = tuner._local_tuner.get_run_config()
                try:
                    tuner_setup = TunerSetup(setup=self)
                    experiment_name = tuner_setup.get_experiment_name()
                    if (self.project and self.project not in experiment_name) or get_run_id() not in experiment_name:
                        logger.warning(
                            "Experiment name %s does not match expected format with project %s and run id %s",
                            experiment_name,
                            self.project,
                            get_run_id(),
                        )
                except Exception:
                    logger.exception("Could not get experiment name from tuner setup")
                    experiment_name = f"{self.project}-{self.args.algorithm.upper()}-{get_run_id()}"
                experiment_path = os.path.join(
                    run_config.storage_path or ".",
                    experiment_name,
                )
                if trials := get_trials_from_tuner(tuner):
                    local_experiment_path = trials[0].local_experiment_path
            if results:
                # XXX Check if paths actually match
                if experiment_path is not None and str(
                    experiment_path.removeprefix("s3://")  # from tuner can be full URI
                ) != results.experiment_path.removeprefix("s3://"):
                    logger.error(
                        "Experiment path from results (%s) does not match path from tuner (%s). Using results path.",
                        results.experiment_path,
                        experiment_path,
                    )
                filesystem = results.filesystem
            elif experiment_path is not None:
                filesystem, experiment_path = pyarrow.fs.FileSystem.from_uri(experiment_path)
            if experiment_path and os.path.exists(experiment_path):
                local_experiment_path = experiment_path
            if failed_runs:
                logger.error("Failed to upload the following wandb runs. Commands to run:\n%s", "\n".join(failed_runs))
                if experiment_path is None and local_experiment_path is None:
                    logger.error("Cannot determine experiment path to log failed uploads.")
                elif local_experiment_path:
                    self._update_failed_upload_file(
                        failed_processes, Path(local_experiment_path), self._upload_to_trial
                    )
                else:
                    failed_file_path = self._update_failed_upload_file(
                        failed_processes,
                        Path(tempfile.gettempdir()),
                        self._upload_to_trial,
                    )
                    experiment_path = tempfile.gettempdir()
                    # Move tempfile to real experiment_path
                    if filesystem is None:
                        try:
                            filesystem, experiment_path = pyarrow.fs.FileSystem.from_uri(experiment_path)
                        except Exception:
                            logger.exception("Could not create filesystem for wandb error file")
                    assert experiment_path is not None
                    if filesystem is not None:
                        # Copy the failed_file_path from local to the remote experiment_path using pyarrow Filesystem
                        with failed_file_path.open("rb") as src_file:
                            filesystem.create_dir(experiment_path, recursive=True)
                            remote_failed_file = str(Path(experiment_path) / failed_file_path.name)
                            with filesystem.open_output_stream(remote_failed_file) as dst_file:
                                dst_file.write(src_file.read())
            if experiment_path is not None:
                self.report_failed_uploads(experiment_path)

    def report_failed_uploads(self, experiment_path: str) -> None:
        """Report failed uploads stored in the failed upload file."""
        # Comet
        if self.args.comet and "upload" in self.args.comet:
            comet_fail_file = Path(COMET_OFFLINE_DIRECTORY) / COMET_FAILED_UPLOAD_FILE
            if comet_fail_file.exists():
                COMET_LOGGER = logging.getLogger("comet_ml.offline")
                with comet_fail_file.open("r") as f:
                    lines = f.readlines()
                if not lines:
                    logger.info("No failed uploads to report, file %s is empty.", comet_fail_file.resolve())
                    return
                # There is the possibility of duplicates. TODO: investigate why comet duplicates in the 100s
                seen = set()
                lines_cleaned = [line for line in lines if not (line in seen or seen.add(line))]
                logger.error(
                    "Reporting %d (%d with duplicated lines) failed uploads from file %s:",
                    len(lines_cleaned),
                    len(lines),
                    comet_fail_file.resolve(),
                )
                for line in lines_cleaned:
                    COMET_LOGGER.error(" - %s", line)
            else:
                logger.info("No failed uploads to report, file %s does not exist.", comet_fail_file.resolve())

        # WandB
        if self.args.wandb and "upload" in self.args.wandb:
            wandb_fail_file = Path(experiment_path, get_wandb_failed_upload_file())
            if wandb_fail_file.exists():
                with wandb_fail_file.open("r") as f:
                    lines = f.readlines()
                if not lines:
                    logger.info("No failed uploads to report, file %s is empty.", wandb_fail_file.resolve())
                    return
                logger.error("Reporting %d failed uploads from file %s:", len(lines), wandb_fail_file.resolve())
                for line in lines:
                    logger.error(" - %s", line.strip())
            else:
                logger.info("No failed uploads to report, file %s does not exist.", wandb_fail_file.name)
