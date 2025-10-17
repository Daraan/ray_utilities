from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast
from urllib.error import HTTPError

import ray
from ray.air.integrations import wandb as ray_wandb
from ray.air.integrations.wandb import _WandbLoggingActor
from ray.tune.utils import flatten_dict

import wandb
import wandb.errors
from ray_utilities.callbacks.tuner.wandb_helpers import (
    FutureArtifact,
    FutureFile,
)
from ray_utilities.callbacks.wandb import wandb_api
from ray_utilities.misc import make_fork_from_csv_header, make_fork_from_csv_line, parse_fork_from

if TYPE_CHECKING:
    from ray_utilities.typing import ForkFromData
    from ray.actor import ActorProxy

    from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor
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
    _monitor: Optional[ActorProxy[WandbRunMonitor]] = None

    def run(self, retries=0):
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
            # proactively check parent before trying to get run
            self._wait_for_missing_parent_data(timeout=50)
            time.sleep(2)
        try:
            return super().run()
        except wandb.errors.CommError as e:  # pyright: ignore[reportPossiblyUnboundVariable]
            # NOTE: its possible that wandb is stuck because of async logging and we never reach here :/
            online = self.kwargs.get("mode", "online") == "online"
            # Note: the error might only be a log message and the actual error is just a timeout
            if (
                "fromStep is greater than the run's last step" in str(e)
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
                self._wait_for_missing_parent_data(timeout=20)

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

    def _wait_for_missing_parent_data(self, timeout=20):
        if not self._monitor:
            from ray_utilities.callbacks._wandb_monitor import WandbRunMonitor  # noqa: PLC0415

            self._monitor = WandbRunMonitor.get_remote_monitor(
                entity=self.kwargs.get("entity", wandb_api().default_entity), project=self.kwargs["project"]
            )
            if not ray.get(self._monitor.is_initialized.remote()):  # pyright: ignore[reportFunctionMemberAccess]
                self._monitor.initialize.remote()  # pyright: ignore[reportFunctionMemberAccess]
        parent_id = self.kwargs["fork_from"].split("?")[0]
        if "config" in self.kwargs:
            assert parent_id == cast("ForkFromData", self.kwargs["config"]["fork_from"]).get(
                "parent_fork_id", parent_id
            )
        logger.debug("Checking run page of parent %s from %s", parent_id, self.kwargs["id"])
        page_visit = self._monitor.visit_run_page.remote(parent_id)  # pyright: ignore[reportFunctionMemberAccess]
        done, _ = ray.wait([page_visit], timeout=timeout)  # wait for page visit to finish
        return bool(done)
