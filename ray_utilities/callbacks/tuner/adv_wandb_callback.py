from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log, _QueueItem

from ray_utilities.constants import DEFAULT_VIDEO_DICT_KEYS
from ray_utilities.misc import RE_GET_TRIAL_ID

from ._save_video_callback import SaveVideoFirstCallback

if TYPE_CHECKING:
    from ray.tune.experiment import Trial

    from ray_utilities.typing.metrics import (
        AutoExtendedLogMetricsDict,
        VideoMetricsDict,
        _LogMetricsEvalEnvRunnersResultsDict,
    )

try:
    from wandb import Video
except ImportError:
    pass  # wandb not installed

_logger = logging.getLogger(__name__)


class AdvWandbLoggerCallback(SaveVideoFirstCallback, WandbLoggerCallback):
    AUTO_CONFIG_KEYS: ClassVar[list[str]] = list({*WandbLoggerCallback.AUTO_CONFIG_KEYS, "trainable_name"})

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

        result_clean = _clean_log(self.preprocess_videos(result))
        if not self.log_config:
            # Config will be logged once log_trial_start
            result_clean.pop("config", None)  # type: ignore
        self._trial_queues[trial].put((_QueueItem.RESULT, result_clean))
