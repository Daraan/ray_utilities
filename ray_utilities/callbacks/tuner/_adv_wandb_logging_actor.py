from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.error import HTTPError

from ray.air.integrations import wandb as ray_wandb
from ray.air.integrations.wandb import _WandbLoggingActor
from ray.tune.utils import flatten_dict

import wandb
import wandb.errors
from ray_utilities.callbacks.tuner.wandb_helpers import (
    FutureArtifact,
    FutureFile,
    get_wandb_web_monitor,
    wandb_api,
    wandb_monitor_lock,
)
from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import make_fork_from_csv_header, make_fork_from_csv_line, parse_fork_from

if TYPE_CHECKING:
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


class _WandbLoggingActorWithArtifactSupport(_WandbLoggingActor):
    def run(self, retries=0):
        if False:
            from ray_utilities.callbacks.tuner.wandb_helpers import _wandb_web_monitor

            logger.warning(f"wandb web monitor at import: {_wandb_web_monitor}")
            try:
                with get_wandb_web_monitor(
                    entity=self.kwargs.get("entity", wandb_api().default_entity), project=self.kwargs["project"]
                ) as monitor:
                    logger.warning(f"wandb web monitor in context: {monitor}")
                    ...
            except Exception as e:
                logger.exception("Failed to get wandb web monitor: %s", str(e))
            from ray_utilities.callbacks.tuner.wandb_helpers import _wandb_web_monitor as wb2

            logger.warning(f"wandb web monitor after import: {wb2}")
        fork_from = self.kwargs.get("fork_from", None) is not None
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
        try:
            return super().run()
        except wandb.errors.CommError as e:  # pyright: ignore[reportPossiblyUnboundVariable]
            # NOTE: its possible that wandb is stuck because of async logging and we never reach here :/
            online = self.kwargs.get("mode", "online") == "online"
            if "fromStep is greater than the run's last step" in str(e):
                # This can happen if the parent run is not yet fully synced, we need to wait for the newest history artifact
                if not fork_from:
                    raise  # should only happen on forks
                if retries >= 5:
                    logger.error(
                        "WandB communication error. online mode: %s, fork_from: %s - Error: %s", online, fork_from, e
                    )
                    if not online:
                        raise
                    logger.warning("Switching to offline mode")
                    self.kwargs["mode"] = "offline"
                    return super().run()
                logger.warning("WandB communication error, retrying after 10s: %s", e)
                time.sleep(10)
                return self.run(retries=retries + 1)
            if not online:
                logger.exception("WandB communication error in offline mode. Cannot recover.")
                raise
            if fork_from:
                logger.error("WandB communication error when using fork_from")
            logger.exception("WandB communication error. Trying to switch to offline mode.")
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
                    logger.error("Failed to log artifact: %s", e)
            elif isinstance(v, FutureArtifact):
                # not serializable
                artifact = wandb.Artifact(
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
                    logger.error("Failed to log artifact: %s", e)
            elif not _is_allowed_type_patch(v):
                continue
            else:
                log[k] = v

        config_update.pop("callbacks", None)  # Remove callbacks
        return log, config_update

    def _wait_for_missing_parent_data(self):
        with wandb_monitor_lock:
            monitor = get_wandb_web_monitor(
                entity=self.kwargs.get("entity", wandb_api().default_entity), project=self.kwargs["project"]
            )
            if FORK_FROM in self.kwargs:
                ...
