from __future__ import annotations

import io
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast
from urllib.error import HTTPError

import ray
import ray.exceptions
import wandb.errors
from pyarrow.fs import FileSystem
from ray.air.constants import TRAINING_ITERATION
from ray.air.integrations import wandb as ray_wandb
from ray.air.integrations.wandb import _QueueItem, _WandbLoggingActor
from ray.tune.utils import flatten_dict

import wandb
from ray_utilities.callbacks.tuner.wandb_helpers import (
    FutureArtifact,
    FutureFile,
)
from ray_utilities.callbacks.wandb import wandb_api
from ray_utilities.misc import make_fork_from_csv_header, make_fork_from_csv_line, parse_fork_from
from ray_utilities.nice_logger import ImportantLogger

if TYPE_CHECKING:
    from ray.actor import ActorProxy

    from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor
    from ray_utilities.typing import ForkFromData
    from ray_utilities.typing.metrics import AnyFlatLogMetricsDict

__all__ = ["_WandbLoggingActorWithArtifactSupport"]

logger = logging.getLogger(__name__)


def _is_allowed_type_patch(obj):
    """Return True if type is allowed for logging to wandb"""
    if _original_is_allowed_type(obj):
        return True
    return isinstance(obj, (FutureFile, FutureArtifact))


_original_is_allowed_type = ray_wandb._is_allowed_type
ray_wandb._is_allowed_type = _is_allowed_type_patch


def clean_invalid_characters(name: str) -> str:
    """Only alphanumeric, hypen and underscore are allowed in wandb names."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


class _LocalWandbLoggingActor(_WandbLoggingActor):
    """Logging actor that works with the run object instead of the wandb module directly."""

    _run: wandb.Run
    _dummy_actor: Optional[ActorProxy[_DummyRemoteActor]] = None

    def run(self):
        # Since we're running in a separate process already, use threads.
        try:
            strout = io.StringIO()
            strerr = io.StringIO()
            start = time.time()
            with redirect_stderr(strerr), redirect_stdout(strout):
                self._run = run = self._wandb.init(*self.args, **self.kwargs)
            end = time.time()
            ImportantLogger.important_info(
                logger,
                "WandB Logging Actor for trial %s initialized wandb run %s\n%s\n%s",
                self._trial_name,
                run.id,
                strout.getvalue(),
                strerr.getvalue(),
            )
        except ValueError:
            logger.exception("WandB Logging Actor for trial %s: Failed to initialize wandb run.", self._trial_name)
            raise
        run.config.trial_log_path = self.kwargs["dir"]  # do not use self._logdir # self._logdir

        ray_wandb._run_wandb_process_run_info_hook(run)

        while True:
            item_type, item_content = self.queue.get()
            if item_type == _QueueItem.END:
                logger.info("WandB logging actor for trial %s ending.", self._trial_name)
                break

            if item_type == _QueueItem.CHECKPOINT:
                self._handle_checkpoint(item_content)
                continue

            assert item_type == _QueueItem.RESULT
            log, config_update = self._handle_result(item_content)
            try:
                run.config.update(config_update, allow_val_change=True)
                run.log(log, step=log.get(TRAINING_ITERATION))
            except HTTPError as e:
                # Ignore HTTPError. Missing a few data points is not a
                # big issue, as long as things eventually recover.
                logger.warning("Failed to log result to w&b: %s", e)
            except FileNotFoundError as e:
                logger.error(
                    "FileNotFoundError: Did not log result to Weights & Biases. "
                    "Possible cause: relative file path used instead of absolute path. "
                    "Error: %s",
                    e,
                )
        logger.debug("Finishing WandB logging actor for trial %s", self._trial_name)
        # run.settings.quiet = True # redirect all output while we call finish
        # WandBs stderr out patch blocks output (possibly tqdm related), but also online in offline mode.
        stderr = io.StringIO()
        stdout = io.StringIO()
        start = time.time()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            run.finish()
            end = time.time()
            logger.info(
                "Finished WandB logging actor for trial %s in %.1f seconds\n%s\n%s",
                self._trial_name,
                end - start,
                stderr.getvalue(),
                stdout.getvalue(),
            )

    def _handle_checkpoint(self, checkpoint_path: str):
        artifact_name = f"checkpoint_{clean_invalid_characters(self._trial_name)}"[:128]
        if len(artifact_name) > 128:
            *front, back = self._trial_name.split("id=")
            back = back[:50]
            if len(front) == 1:
                front_name: str = front[0][:70]
                artifact_name = f"checkpoint_{clean_invalid_characters(front_name + ' ... id=' + back)}"[:128]
            else:  # multiple 'id=' in name, just truncate
                artifact_name = artifact_name[:128]
        artifact = self._wandb.Artifact(name=artifact_name, type="model")
        artifact.add_dir(checkpoint_path)
        self._run.log_artifact(artifact)

    def register_dummy_actor(self, actor: ActorProxy[_DummyRemoteActor]):
        self._dummy_actor = actor

    def __del__(self):
        if self._dummy_actor:
            ray.kill(self._dummy_actor)


class _WandbLoggingActorWithArtifactSupport(_LocalWandbLoggingActor, _WandbLoggingActor):
    _monitor: Optional[ActorProxy[WandbRunMonitor]] = None

    def run(self, retries=0):
        fork_from = self.kwargs.get("fork_from", None) is not None
        online = self.kwargs.get("mode", "online") == "online"
        if fork_from:
            # Write info about forked trials, to know in which order to upload trials
            # This in the trial dir, no need for a Lock
            info_file = Path(self._logdir).parent / "wandb_fork_from.csv"
            if not info_file.exists():
                # write header
                info_file.write_text(make_fork_from_csv_header())
            fork_data_tuple = parse_fork_from(self.kwargs["fork_from"])
            with info_file.open("a") as f:
                if fork_data_tuple is not None:
                    parent_id, parent_step = fork_data_tuple
                    line = make_fork_from_csv_line(
                        {
                            "parent_trial_id": parent_id,
                            "fork_id_this_trial": self.kwargs["id"],
                            "parent_training_iteration": cast("int", parent_step),
                            "parent_time": ("_step", cast("float", parent_step)),
                        },
                        optional=True,
                    )
                    f.write(line)
                else:
                    logger.error("Could not parse fork_from: %s", self.kwargs["fork_from"])
                    f.write(f"{self.kwargs['id']}, {self.kwargs['fork_from']}\n")
            # proactively check parent before trying to get run
            if online and os.environ.get("RAY_UTILITIES_NO_MONITOR", "1") != "1":
                self._wait_for_missing_parent_data(timeout=50)
            time.sleep(2)
        try:
            return super().run()
        except wandb.errors.CommError as e:  # pyright: ignore[reportPossiblyUnboundVariable]
            # Note: the error might only be a log message and the actual error is just a timeout
            if (
                "fromStep is greater than the run's last step" in str(e)
                or "history not materialized yet" in str(e)
                or "contact support" in str(e)
                or (online and fork_from)
            ):
                # Happens if the parent run is not yet fully synced, we need to wait for the newest history artifact
                if not fork_from:
                    raise  # should only happen on forks
                if retries >= 4:
                    logger.error(
                        "WandB communication error. online mode: %s, fork_from: %s - Error: %s", online, fork_from, e
                    )
                    if not online:
                        raise
                    logger.warning("Retries failed for wandb. Switching wandb to offline mode")
                    self.kwargs["mode"] = "offline"
                    self.kwargs["reinit"] = "create_new"
                    return super().run()
                logger.warning("WandB communication error, using browser to open parent run: %s", e)
                if isinstance(self._wait_for_missing_parent_data(timeout=20), Exception):
                    # we can likely not recover from this
                    time.sleep(5)
                    return self.run(retries=retries + 4)

                return self.run(retries=retries + 1)
            if not online:
                logger.exception("WandB communication error in offline mode. Cannot recover.")
                raise
            if fork_from:
                logger.error("WandB communication error when using fork_from")
            logger.exception("WandB communication error. Trying to switch to offline mode.")
            self.kwargs["mode"] = "offline"
            self.kwargs["reinit"] = "create_new"
            return super().run()
            # TODO: communicate to later upload offline run
        except Exception:
            logger.exception("WandB Logging Actor failed: %s")
            raise

    def _handle_result(self, result: dict) -> tuple[dict, dict]:
        config_update = result.get("config", {}).copy()
        log = {}
        flat_result: AnyFlatLogMetricsDict | dict[str, Any] = flatten_dict(result, delimiter="/")

        for k, v in flat_result.items():
            if any(k.startswith(item + "/") or k == item for item in self._exclude):
                continue
            if any(k.startswith(item + "/") or k == item for item in self._to_config):
                config_update[k] = v
            elif isinstance(v, FutureFile):
                try:
                    file_path = str(v.global_str)
                    base_path = v.base_path

                    # Handle S3 paths by downloading the file first
                    if file_path.startswith("s3://"):
                        logger.warning(
                            "Adding a S3 file to wandb via temporary download. Consider logging an Artifact that points to it instead."
                        )
                        # TODO: Should use an Artifact in this situation
                        try:
                            s3_fs, s3_path = FileSystem.from_uri(file_path)
                            # Create a temporary local file to download to
                            local_file = tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=Path(file_path).suffix,
                                dir=self._logdir,  # in local mode this will be the working dir TODO if correct
                            )
                            local_file_path = local_file.name
                            local_file.close()

                            # Download from S3
                            logger.debug("Downloading S3 file %s to %s for wandb upload", file_path, local_file_path)
                            with s3_fs.open_input_stream(s3_path) as s3_file:
                                with open(local_file_path, "wb") as local_f:
                                    local_f.write(s3_file.read())

                            # Upload the local file to wandb
                            self._run.save(local_file_path, base_path=self._logdir)

                            # Clean up temporary file
                            try:
                                os.unlink(local_file_path)
                            except Exception as cleanup_error:
                                logger.warning(
                                    "Failed to clean up temporary file %s: %s", local_file_path, cleanup_error
                                )
                        except Exception as s3_error:
                            logger.error("Failed to download and log S3 artifact %s: %s", file_path, s3_error)
                    else:
                        # Local file path, use directly
                        assert Path(file_path).exists(), f"File path for wandb upload does not exist: {file_path}"
                        self._run.save(file_path, base_path=base_path, policy=v.policy)
                except (HTTPError, Exception) as e:
                    logger.error("Failed to log artifact %s %s %s: %s", v.global_str, v.base_path, v, e)
            elif isinstance(v, FutureArtifact):
                # not serializable
                artifact = wandb.Artifact(
                    name=clean_invalid_characters(v.name),
                    type=v.type,
                    description=v.description,
                    metadata=v.metadata,
                    incremental=v.incremental,
                    **v.kwargs,
                )
                for file_dict in v._added_files:
                    artifact.add_file(**file_dict)
                for dir_dict in v._added_dirs:
                    artifact.add_dir(**dir_dict)
                for ref_dict in v._added_references:
                    artifact.add_reference(**ref_dict)
                try:
                    self._run.log_artifact(artifact)
                except (HTTPError, Exception):
                    logger.exception("Failed to log artifact: %s")
            elif isinstance(v, float) and math.isnan(v):
                # HACK: Currently wandb fails to log metric on forks if the parent has NaN metrics
                # # see https://github.com/wandb/wandb/issues/1069 until then do not upload to wandb
                continue
            elif not _is_allowed_type_patch(v):
                continue
            else:
                log[k] = v

        config_update.pop("callbacks", None)  # Remove callbacks
        return log, config_update

    def _wait_for_missing_parent_data(self, timeout=20) -> bool | Exception:
        if not self._monitor:
            from ray_utilities.callbacks._wandb_monitor import WandbRunMonitor  # noqa: PLC0415

            try:
                self._monitor = WandbRunMonitor.get_remote_monitor(
                    entity=self.kwargs.get("entity", wandb_api().default_entity), project=self.kwargs["project"]
                )
                if not ray.get(self._monitor.is_initialized.remote()):  # pyright: ignore[reportFunctionMemberAccess]
                    self._monitor.initialize.remote()  # pyright: ignore[reportFunctionMemberAccess]
            except ray.exceptions.ActorDiedError as e:
                # i.g. credentials missing
                logger.error("Could not get WandbRunMonitor actor. It died: %s", e.error_msg)
                return e
            except ValueError as ve:
                logger.exception("Could not get WandbRunMonitor actor:")
                return ve
        parent_id = self.kwargs["fork_from"].split("?")[0]
        if "config" in self.kwargs:
            assert parent_id == cast("ForkFromData", self.kwargs["config"]["fork_from"]).get(
                "parent_fork_id", parent_id
            )
        logger.debug("Checking run page of parent %s from %s", parent_id, self.kwargs["id"])
        page_visit = self._monitor.visit_run_page.remote(parent_id)  # pyright: ignore[reportFunctionMemberAccess]
        done, _ = ray.wait([page_visit], timeout=timeout)  # wait for page visit to finish
        return bool(done)


class _DummyRemoteActor:
    """A dummy remote actor used for testing or as a placeholder for wandb logging."""

    def __init__(self):
        self._stop = False
        self._event = threading.Event()

    def run(self):
        """Blocks and waits until the stop event is set, periodically sleeping."""
        while not self._event.is_set():
            time.sleep(10)

    @ray.method(num_returns=0, concurrency_group="stop", max_task_retries=-1)
    def stop(self):
        logger.debug("Stopping %s for wandb logging.", self.__class__.__name__)
        self._event.set()
