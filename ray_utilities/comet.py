# ruff: noqa: PLC0415
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence, cast

import comet_ml

from ray_utilities.constants import COMET_OFFLINE_DIRECTORY

_api: Optional[comet_ml.API] = None
"""Singleton instance of the Comet API client to make use of caching."""

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "comet_assure_project_exists",
    "comet_upload_offline_experiments",
    "get_comet_api",
    "get_default_workspace",
]


def get_comet_api() -> comet_ml.API:
    """Create a persistent Comet API client that makes use of caching."""
    global _api  # noqa: PLW0603
    if _api is None:
        _api = comet_ml.API()
    return _api


def get_default_workspace() -> str:
    """
    Returns the default Comet workspace.

    this looks up env custom environment variable COMET_DEFAULT_WORKSPACE
    or the first workspace in the list of workspaces.
    """
    try:
        return os.environ.get("COMET_DEFAULT_WORKSPACE") or get_comet_api().get_default_workspace()
    except IndexError as e:
        raise ValueError(
            "COMET_DEFAULT_WORKSPACE is not set and no comet workspaces were found. Create a workspace first."
        ) from e


def comet_upload_offline_experiments(tracker: Optional[CometArchiveTracker] = None):
    if tracker is None:
        tracker = _default_comet_archive_tracker
    tracker.upload_and_move()


def comet_assure_project_exists(workspace_name: str, project_name: str, project_description: Optional[str] = None):
    api = get_comet_api()
    projects = cast("list[str]", api.get(workspace_name))
    if project_name not in projects:
        api.create_project(
            workspace_name,
            project_name,
            project_description=project_description,
        )


class CometArchiveTracker:
    """Cheap tracker that checks for *.zip archives in the given path and updates the list of archives."""

    def __init__(
        self,
        track: Optional[Sequence[Path]] = None,
        *,
        auto: bool = True,
        path: str | Path = Path(COMET_OFFLINE_DIRECTORY),
    ):
        self.path = Path(path)
        self._initial_archives = set(self.get_archives())
        self.archives = list(track) if track else []
        self._auto = auto
        self._called_upload: bool = False

    def get_archives(self):
        return list(self.path.glob("*.zip"))

    def update(self, new_archives: Optional[Sequence[Path]] = None):
        if self._auto:
            archives_now = self.get_archives()
            self.archives.extend([p for p in archives_now if p not in self._initial_archives])
        elif new_archives is None:
            _LOGGER.warning("Should provide a (possibly empty) list of new archives to update when auto=False")
        if new_archives:
            self.archives.extend(new_archives)
        self.archives = [p for p in set(self.archives) if p.exists()]

    def _upload(self, archives: Optional[Sequence[Path]] = None):
        self._called_upload = True
        if archives and self._auto:
            _LOGGER.warning(
                "Auto mode is enabled, will upload all archives. "
                "To suppress this warning use update(archives) before upload."
            )
        if self._auto:
            self.update(archives)
            archives = self.archives
        if not archives:
            _LOGGER.info("No archives to upload")
            return
        archives_str = [str(p) for p in self.archives]
        _LOGGER.info("Uploading Archives: %s", archives_str)
        comet_ml.offline.main_upload(archives_str, force_upload=False)

    def upload_and_move(self):
        self._upload()
        self.move_archives()

    def make_uploaded_dir(self):
        new_dir = self.path / "uploaded"
        new_dir.mkdir(exist_ok=True)
        return new_dir

    def move_archives(self):
        if not self._called_upload:
            _LOGGER.warning("Called move_archives without calling upload first.")
        new_dir = self.make_uploaded_dir()
        for path in self.archives:
            path.rename(new_dir / path.name)


_default_comet_archive_tracker = CometArchiveTracker()
