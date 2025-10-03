from __future__ import annotations

import abc
import logging
import os
import pickle
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast
from urllib.error import HTTPError

from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log, _QueueItem, _WandbLoggingActor
from ray.tune.utils import flatten_dict

from ray_utilities import RUN_ID
from ray_utilities.callbacks.tuner.new_style_logger_callback import LogMetricsDictT, NewStyleLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.comet import _LOGGER
from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS, FORK_FROM
from ray_utilities.misc import extract_trial_id_from_checkpoint, make_experiment_key, parse_fork_from

if TYPE_CHECKING:
    from ray_utilities.typing.metrics import AnyFlatLogMetricsDict
    from ray_utilities.typing import ForkFromData

from ._log_result_grouping import non_metric_results
from ._save_video_callback import SaveVideoFirstCallback

if TYPE_CHECKING:
    from ray.tune.experiment import Trial
    from wandb.sdk.interface.interface import PolicyName

    from ray_utilities.typing.metrics import (
        VideoMetricsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
    )

try:
    from wandb import Artifact, Video
except ImportError:
    pass  # wandb not installed
else:
    from ray.air.integrations import wandb as ray_wandb
    from wandb.errors import CommError

    def _is_allowed_type_patch(obj):
        """Return True if type is allowed for logging to wandb"""
        if _original_is_allowed_type(obj):
            return True
        return isinstance(obj, (FutureFile, FutureArtifact))

    _original_is_allowed_type = ray_wandb._is_allowed_type
    ray_wandb._is_allowed_type = _is_allowed_type_patch

_logger = logging.getLogger(__name__)


class _WandbLoggingActorWithArtifactSupport(_WandbLoggingActor):
    def run(self, retries=0):
        fork_from = self.kwargs.get("fork_from", None) is not None
        if fork_from:
            # Write info about forked trials, to know in which order to upload trials
            info_file = Path(self._logdir) / "wandb_fork_from.txt"
            if not info_file.exists():
                # write header
                info_file.write_text("trial_id, parent_id, parent_step, step_metric\n")
            fork_data = parse_fork_from(self.kwargs["fork_from"])
            with info_file.open("a") as f:
                if fork_data is not None:
                    parent_id, parent_step = fork_data
                    f.write(f"{self.kwargs['id']}, {parent_id}, {parent_step}, _step\n")
                else:
                    _logger.error("Could not parse fork_from: %s", self.kwargs["fork_from"])
                    f.write(f"{self.kwargs['id']}, {self.kwargs['fork_from']}\n")
        try:
            return super().run()
        except CommError as e:  # pyright: ignore[reportPossiblyUnboundVariable]
            # NOTE: its possible that wandb is stuck because of async logging and we never reach here :/
            # breakpoint()
            online = self.kwargs.get("mode", "online") == "online"
            if "fromStep is greater than the run's last step" in str(e):
                # This can happen if the parent run is not yet fully synced.
                if not fork_from:
                    raise  # should only happen on forks
                if retries >= 5:
                    _logger.error(
                        "WandB communication error. online mode: %s, fork_from: %s - Error: %s", online, fork_from, e
                    )
                    if not online:
                        raise
                    _logger.warning("Switching to offline mode")
                    self.kwargs["mode"] = "offline"
                    return super().run()
                _logger.warning("WandB communication error, retrying after 10s: %s", e)
                time.sleep(10)
                return self.run(retries=retries + 1)
            if not online:
                _logger.exception("WandB communication error in offline mode. Cannot recover.")
                raise
            if fork_from:
                _logger.error("WandB communication error when using fork_from")
            _logger.exception("WandB communication error. Trying to switch to offline mode.")
            self.kwargs["mode"] = "offline"
            super().run()
            # TODO: communicate to later upload offline run

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
                    self._wandb.save(v.global_str, base_path=v.base_path)
                except (HTTPError, Exception) as e:  # noqa: BLE001
                    _logger.error("Failed to log artifact: %s", e)
            elif isinstance(v, FutureArtifact):
                # not serializable
                artifact = Artifact(  # pyright: ignore[reportPossiblyUnboundVariable]
                    name=v.name,
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
                    self._wandb.log_artifact(artifact)
                except (HTTPError, Exception) as e:
                    _logger.error("Failed to log artifact: %s", e)
            elif not _is_allowed_type_patch(v):
                continue
            else:
                log[k] = v

        config_update.pop("callbacks", None)  # Remove callbacks
        return log, config_update


class AdvWandbLoggerCallback(
    NewStyleLoggerCallback, SaveVideoFirstCallback, TrackForkedTrialsMixin, WandbLoggerCallback
):
    AUTO_CONFIG_KEYS: ClassVar[list[str]] = list(
        {
            *WandbLoggerCallback.AUTO_CONFIG_KEYS,
            *non_metric_results,
        }
    )

    _logger_actor_cls = _WandbLoggingActorWithArtifactSupport

    _logged_architectures: set[Trial]

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

    def on_trial_start(self, iteration: int, trials: list["Trial"], trial: "Trial", **info):
        super().on_trial_start(iteration, trials, trial, **info)
        _logger.debug("Trials created: %d, re-started: %d", self._trials_created, self._trials_started)
        self._trials = trials  # keep them in case of a failure to access paths.

    def log_trial_start(self, trial: "Trial"):
        # breakpoint()
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
            fork_id = fork_data["parent_id"]
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
            config.setdefault("experiment_key", make_experiment_key(trial))

        # TODO: trial_id of a forked trial might be very long

        # Test for invalid chars
        assert not trial_id or all(c not in trial_id for c in r"/ \ # ? % :"), f"Invalid character in: {trial_id}"
        assert fork_from is None or fork_from.count("?_step=") == 1, fork_from
        # FIXME: fork_from is not stable on WandB when there is no longer wait time, wandb.init likely raises error
        # NOTE: We never want FORK_FROM to be in the trials.config by default.
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
        if fork_from and trial in self._trial_logging_futures:
            assert self.is_trial_forked(trial), "Expected trial to be tracked as forked trial."
            self._restart_logging_actor(trial, **wandb_init_kwargs)
        else:
            # can be forked from a checkpoint, if not stopped does not start a new
            self._start_logging_actor(trial, exclude_results, **wandb_init_kwargs)
        self._trials_started += 1

    def _restart_logging_actor(self, trial: "Trial", **wandb_init_kwargs):
        """Ends the current logging actor and starts a new one. Useful for resuming with a new ID / settings."""
        self.log_trial_end(trial, failed=False)
        _logger.info("Restarting WandB logging actor for trial %s", trial.trial_id)
        # Wait a bit before starting the next one
        self._cleanup_logging_actors(timeout=5, kill_on_timeout=False)
        # Clean queue and futures else a new one will not be created
        self._trial_queues.pop(trial, None)
        self._trial_logging_futures.pop(trial, None)
        self._trial_logging_actors.pop(trial, None)
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

    def on_trial_complete(self, iteration: int, trials: list["Trial"], trial: "Trial", **info):
        """Called when a trial has completed. Triggers sync for offline runs."""
        # Call parent method to handle normal trial completion
        super().on_trial_complete(iteration, trials, trial, **info)

        # TODO: Also sync if the trial will be perturbed; but it will not reach on_trial_complete!
        # Furthermore on_trial_complete there will be multiple folders.

        # If we are in offline mode, try to sync this trial's run immediately
        if "offline" in self.kwargs.get("mode", "") and self.upload_offline_experiments:
            # Wandb dir is likely not yet saved by actor, wait for it, super does not wait that long.
            self._cleanup_logging_actors(timeout=120, kill_on_timeout=False)
            _LOGGER.info("Syncing offline WandB run for trial %s", trial.trial_id)
            # NOTE: Actor should have synced everything at this point
            self._sync_offline_run_if_available(trial)

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
                result = subprocess.run(
                    ["wandb", "sync", str(run_dir)],
                    check=False,
                    text=True,
                    timeout=600,  # timeout 10 minutes
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode == 0 and "error" not in result.stdout.lower():
                    _logger.info("Successfully synced offline run for trial %s\n%s", trial.trial_id, result.stdout)
                elif "not found (<Response [404]>)" in result.stdout:
                    _logger.error(
                        "Could not sync run for trial %s "
                        "(Is it a forked_run? - The parent needs to be uploaded first): %s",
                        trial.trial_id,
                        result.stdout,
                    )
                else:
                    _logger.error("Error during syncing offline run for trial %s: %s", trial.trial_id, result.stdout)
                if result.returncode != 0 or result.stderr:
                    _logger.error("Failed to sync offline run for trial %s: %s", trial.trial_id, result.stderr)
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
                _LOGGER.debug("Storing future Artifact %s", artifact.to_dict())
            else:
                _LOGGER.error("Cannot save model_architecture as trial.path is None")

        result_clean = _clean_log(self.preprocess_videos(result))
        if not self.log_config:
            # Config will be logged once log_trial_start
            result_clean.pop("config", None)  # type: ignore
        self._trial_queues[trial].put((_QueueItem.RESULT, result_clean))


class _WandbFuture(abc.ABC):
    @abc.abstractmethod
    def json_encode(self) -> dict[str, Any]: ...

    def to_dict(self):
        return self.json_encode()


class FutureFile(_WandbFuture):
    """A file to be logged to WandB for this run, has to be compatible with :meth:`wandb.save`."""

    def __init__(
        self,
        glob_str: str | os.PathLike,
        base_path: str | os.PathLike | None = None,
        policy: PolicyName = "live",
    ) -> None:
        self.global_str = glob_str
        self.base_path = base_path
        self.policy = policy

    def json_encode(self) -> dict[str, Any]:
        return {
            "glob_str": self.global_str,
            "base_path": self.base_path,
            "policy": self.policy,
        }


class FutureArtifact(_WandbFuture):
    def __init__(
        self,
        name: str,
        type: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        incremental: bool = False,
        **kwargs,
    ):
        if not re.match(r"^[a-zA-Z0-9_\-.]+$", name):
            raise ValueError(
                f"Artifact name may only contain alphanumeric characters, dashes, "
                f"underscores, and dots. Invalid name: {name}"
            )
        self.name = name
        self.type = type
        self.description = description
        self.metadata = metadata
        self.incremental = incremental
        self.kwargs = kwargs
        self._added_dirs = []
        self._added_files = []
        self._added_references = []

    def add_reference(self, uri: Any | str, name: str | None = None, **kwargs) -> None:
        self._added_references.append({"uri": uri, "name": name, **kwargs})

    def add_file(
        self,
        local_path: str,
        name: str | None = None,
        *,
        is_tmp: bool | None = False,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        self._added_files.append(
            {
                "local_path": local_path,
                "name": name,
                "is_tmp": is_tmp,
                "overwrite": overwrite,
                **kwargs,
            }
        )

    def add_dir(
        self,
        local_path: str,
        name: str | None = None,
        **kwargs,
    ) -> None:
        self._added_dirs.append(
            {
                "local_path": local_path,
                "name": name,
                **kwargs,
            }
        )

    def json_encode(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "metadata": self.metadata,
            "incremental": self.incremental,
            "kwargs": self.kwargs,
            "added_dirs": self._added_dirs,
            "added_files": self._added_files,
            "added_references": self._added_references,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.json_encode()
