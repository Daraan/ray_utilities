from __future__ import annotations

import abc
import logging
import os
import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast
from urllib.error import HTTPError

from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log, _QueueItem, _WandbLoggingActor
from ray.tune.utils import flatten_dict

from ray_utilities import run_id
from ray_utilities.comet import _LOGGER
from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS
from ray_utilities.misc import RE_GET_TRIAL_ID

from ._save_video_callback import SaveVideoFirstCallback

if TYPE_CHECKING:
    from ray.tune.experiment import Trial
    from wandb.sdk.interface.interface import PolicyName

    from ray_utilities.typing.metrics import (
        AutoExtendedLogMetricsDict,
        VideoMetricsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
    )

try:
    from wandb import Artifact, Video
except ImportError:
    pass  # wandb not installed
else:
    from ray.air.integrations import wandb as ray_wandb

    def _is_allowed_type_patch(obj):
        """Return True if type is allowed for logging to wandb"""
        if _original_is_allowed_type(obj):
            return True
        return isinstance(obj, (FutureFile, FutureArtifact))

    _original_is_allowed_type = ray_wandb._is_allowed_type
    ray_wandb._is_allowed_type = _is_allowed_type_patch

_logger = logging.getLogger(__name__)


class _WandbLoggingActorWithArtifactSupport(_WandbLoggingActor):
    def _handle_result(self, result: dict) -> tuple[dict, dict]:
        config_update = result.get("config", {}).copy()
        log = {}
        flat_result = flatten_dict(result, delimiter="/")

        for k, v in flat_result.items():
            if any(k.startswith(item + "/") or k == item for item in self._exclude):
                continue
            if any(k.startswith(item + "/") or k == item for item in self._to_config):
                config_update[k] = v
            if isinstance(v, FutureFile):
                try:
                    self._wandb.save(v.global_str, base_path=v.base_path)
                except (HTTPError, Exception) as e:
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


class AdvWandbLoggerCallback(SaveVideoFirstCallback, WandbLoggerCallback):
    AUTO_CONFIG_KEYS: ClassVar[list[str]] = list(
        {
            *WandbLoggerCallback.AUTO_CONFIG_KEYS,
            "trainable_name",
            "experiment_group",
            "experiment_name",
            "run_id",
            "experiment_key",
        }
    )

    _logger_actor_cls = _WandbLoggingActorWithArtifactSupport

    _logged_architectures: set[Trial]
    if not TYPE_CHECKING:  # keep signature of parent

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._trials_created = 0
            self._logged_architectures = set()

    def on_trial_start(self, iteration: int, trials: list["Trial"], trial: "Trial", **info):
        super().on_trial_start(iteration, trials, trial, **info)
        if self._trials_created != len(trials):
            _logger.warning(
                "Number of created trials %d does not match the number of tracked trials %d.",
                len(trials),
                self._trials_created,
            )
        self._trials = trials  # keep them in case of a failure to access paths.

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
        config["run_id"] = run_id
        config.setdefault("experiment_id", run_id)
        # replace potential _ in trial_id
        if "_" in trial.trial_id:
            _trial_number = int(trial.trial_id.split("_")[-1])
            # cannot check for id as they are possibly created out of order, 00000 last, 00001 first
        config["experiment_key"] = f"{run_id:0<20}xXx{trial.trial_id}xXx{self._trials_created:0>4}".replace("_", "x00x")
        # --- New Code --- : Remove nested keys
        for nested_key in filter(lambda x: "/" in x, self.excludes):
            key, sub_key = nested_key.split("/")
            if key in config:
                config[key].pop(sub_key, None)
        assert "num_jobs" not in config["cli_args"]
        assert "test" not in config["cli_args"]
        fork_from = None  # new run
        if trial.config["cli_args"].get("from_checkpoint"):
            match = RE_GET_TRIAL_ID.search(trial.config["cli_args"]["from_checkpoint"])
            # get id of run
            if match is None:
                # Deprecated:
                # possible old format without id=
                match = re.search(rf"(?:id=)?([a-zA-Z0-9]+_[0-9]{5})", trial.config["cli_args"]["from_checkpoint"])
                if match is None:
                    _logger.error(
                        "Cannot extract trial id from checkpoint name: %s. "
                        "Make sure that it has to format id=<part1>_<sample_number>",
                        trial.config["cli_args"]["from_checkpoint"],
                    )
            else:
                ckpt_trial_id = match.groupdict()["trial_id"]
                # Need to change to format '<run>?<metric>=<numeric_value>'
                # Where metric="_step"
                # open state pickle to get iteration
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
                    fork_from = f"{ckpt_trial_id}?_step={iteration}"
        # --- End New Code
        wandb_init_kwargs = {
            "id": trial_id,
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

        self._start_logging_actor(trial, exclude_results, **wandb_init_kwargs)
        self._trials_created += 1

    @staticmethod
    def preprocess_videos(metrics: dict[Any, Any] | AutoExtendedLogMetricsDict) -> dict[Any, Any]:
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
                        metrics = metrics.copy()
                        did_copy = True
                    parent_dir = metrics
                    for key in keys[:-1]:
                        parent_dir[key] = parent_dir[key].copy()  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]  # fmt: skip
                        parent_dir = parent_dir[key]  # pyright: ignore[reportGeneralTypeIssues, reportTypedDictNotRequiredAccess]  # fmt: skip
                    parent_dir = cast("_LogMetricsEvalEnvRunnersResultsDict", parent_dir)
                    parent_dir[keys[-1]] = video_dict = cast("VideoMetricsDict", parent_dir[keys[-1]]).copy()  # pyright: ignore[reportTypedDictNotRequiredAccess]  # fmt: skip
                    # IMPORTANT use absolute path as local path is a ray session!
                    video_dict["video"] = Video(os.path.abspath(video_dict.pop("video_path")), format="mp4")  # pyright: ignore[reportPossiblyUnboundVariable] # fmt: skip

        return metrics  # type: ignore[return-value]

    def log_trial_result(
        self,
        iteration: int,  # noqa: ARG002
        trial: "Trial",
        result: "dict | AutoExtendedLogMetricsDict",
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
