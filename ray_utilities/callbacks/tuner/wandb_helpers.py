from __future__ import annotations

import abc
import enum
import logging
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import dotenv
import ray
from ray.air.util.node import _force_on_current_node
from ray.util.queue import Queue


if TYPE_CHECKING:
    from wandb.sdk.interface.interface import PolicyName

    from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor


_logger = logging.getLogger(__name__)
try:
    import wandb  # noqa: F401
except ModuleNotFoundError:
    _logger.error("wandb.Api() not available, wandb might not be installed")

wandb_monitor_lock = threading.RLock()


class MonitorQueueActions(enum.Enum):
    START = enum.auto()
    OPEN_RUN = enum.auto()
    CHECK_ARTIFACT = enum.auto()
    END = enum.auto()


def create_monitor_queue(actor_options: dict[str, Any] | None = None):
    return Queue(
        actor_options={
            "num_cpus": 0,
            **_force_on_current_node(),
            "max_restarts": -1,
            "max_task_retries": -1,
        }
        | (actor_options or {}),
    )


_wandb_web_monitor: WandbRunMonitor | None = None
"""Singleton instance of WandbRunMonitor, created on first use."""

_wandb_web_monitor_ref: ray.ObjectRef[WandbRunMonitor] | None = None
"""Ray ObjectRef to the singleton instance of WandbRunMonitor, created on first use."""


def setup_wandb_monitor(entity: str, project: str):  # -> tuple[WandbRunMonitor, ObjectRef[WandbRunMonitor] | Any]:
    """Setup the global WandbRunMonitor instance."""
    global _wandb_web_monitor, _wandb_web_monitor_ref  # noqa: PLW0603
    with wandb_monitor_lock:
        if _wandb_web_monitor is None:
            from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor  # noqa: PLC0415

            try:
                _wandb_web_monitor = WandbRunMonitor(entity=entity, project=project)
            except Exception:
                _logger.exception("Failed to create WandbRunMonitor")
                raise
            _wandb_web_monitor_ref = ray.put(_wandb_web_monitor)
        elif _wandb_web_monitor.entity != entity or _wandb_web_monitor.project != project:
            _logger.warning(
                "WandbRunMonitor already initialized with entity=%s, project=%s. "
                "Requested entity=%s, project=%s. Returning existing monitor.",
                _wandb_web_monitor.entity,
                _wandb_web_monitor.project,
                entity,
                project,
            )
    return _wandb_web_monitor, _wandb_web_monitor_ref


@contextmanager
def get_wandb_web_monitor(entity: str, project: str) -> Generator[WandbRunMonitor, Any, None]:
    """
    Contextmanager to acquire a lock and get the WandbRunMonitor instance.
    Get or create the WandbRunMonitor instance for monitoring forked trials.
    If a monitor already exists with different entity or project, a warning is logged and the existing monitor is returned.
    """
    with wandb_monitor_lock:
        global _wandb_web_monitor  # noqa: PLW0603
        if _wandb_web_monitor is None:
            # import lazy to avoid selenium dependency if not used
            viewer_env_file = Path("~/.comet_api_key.env").expanduser()
            if not dotenv.load_dotenv(viewer_env_file) and (
                "WANDB_VIEWER_MAIL" not in os.environ or "WANDB_VIEWER_PW" not in os.environ
            ):
                _logger.error(
                    "WandbRunMonitor needs viewer credentials to log in to wandb. "
                    "Please create a file at %s with WANDB_VIEWER_MAIL and WANDB_VIEWER_PW, "
                    "or set these environment variables.",
                    viewer_env_file,
                )
            from ray_utilities.callbacks._wandb_monitor.wandb_run_monitor import WandbRunMonitor  # noqa: PLC0415

            try:
                _wandb_web_monitor = WandbRunMonitor(entity=entity, project=project)
            except Exception:
                _logger.exception("Failed to create WandbRunMonitor")
                raise
        elif _wandb_web_monitor.entity != entity or _wandb_web_monitor.project != project:
            _logger.warning(
                "WandbRunMonitor already initialized with entity=%s, project=%s. "
                "Requested entity=%s, project=%s. Returning existing monitor.",
                _wandb_web_monitor.entity,
                _wandb_web_monitor.project,
                entity,
                project,
            )
        print(f"Entering wandb web monitor context for entity={entity}, project={project}")
        yield _wandb_web_monitor
        print("Leaving wandb web monitor context")


class _WandbFuture(abc.ABC):
    @abc.abstractmethod
    def json_encode(self) -> dict[str, Any]: ...

    def to_dict(self):
        return self.json_encode()


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
