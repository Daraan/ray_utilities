"""Comet ML integration utilities for experiment tracking and offline experiment management.

This module provides utilities for integrating with `Comet ML <https://www.comet.ml/>`_
experiment tracking, including API management, workspace handling, and offline experiment
uploading. It's designed to work seamlessly with Ray Tune experiments and provides
automatic project creation and experiment archiving.

The module handles Comet ML offline directory configuration and provides tools for
batch uploading offline experiments that were run without internet connectivity.

Key Components:
    - :func:`get_comet_api`: Persistent API client with caching
    - :func:`comet_assure_project_exists`: Automatic project creation
    - :class:`CometArchiveTracker`: Offline experiment management
    - :func:`comet_upload_offline_experiments`: Batch upload offline experiments

Example:
    Basic Comet ML integration::

        from ray_utilities.comet import get_comet_api, comet_assure_project_exists

        # Ensure project exists
        workspace = "my-workspace"
        project = "my-ray-experiment"
        comet_assure_project_exists(workspace, project, "Ray Tune hyperparameter optimization")

        # Use in Ray Tune with CometLoggerCallback
        from ray.air.integrations.comet import CometLoggerCallback

        callbacks = [
            CometLoggerCallback(
                project_name=project,
                workspace=workspace,
            )
        ]

    Offline experiment management::

        from ray_utilities.comet import CometArchiveTracker

        # Track and upload offline experiments
        tracker = CometArchiveTracker(auto=True)
        # ... run experiments offline ...
        tracker.upload_and_move()  # Upload all new experiments

Constants:
    :data:`COMET_OFFLINE_DIRECTORY`: Path where offline Comet experiments are stored

See Also:
    :mod:`comet_ml`: The Comet ML Python SDK
    :class:`ray.air.integrations.comet.CometLoggerCallback`: Ray's Comet integration
    :data:`ray_utilities.constants.COMET_OFFLINE_DIRECTORY`: Offline storage configuration
"""

# ruff: noqa: PLC0415
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence, cast

import comet_ml

from ray_utilities.constants import COMET_OFFLINE_DIRECTORY

_api: Optional[comet_ml.API] = None
"""
Singleton instance of the Comet API client to make use of caching.

Use :func:`get_comet_api` to access this instance
and initialize it if it first if it is not already created.
"""

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "COMET_OFFLINE_DIRECTORY",
    "CometArchiveTracker",
    "comet_assure_project_exists",
    "comet_upload_offline_experiments",
    "get_comet_api",
    "get_default_workspace",
]


def get_comet_api() -> comet_ml.API:
    """Create a persistent Comet API client that makes use of caching.

    This function maintains a singleton :class:`comet_ml.API` instance to avoid
    repeated API initialization overhead and enable caching of API responses.

    Returns:
        A :class:`comet_ml.API` instance that can be used for workspace and project
        management operations.

    Example:
        >>> api = get_comet_api()
        >>> workspaces = api.get_workspaces()
        >>> projects = api.get("my-workspace")

    Note:
        The API client requires proper Comet ML authentication via API key.
        See `Comet ML documentation <https://www.comet.ml/docs/python-sdk/API/>`_
        for authentication setup.
    """
    global _api  # noqa: PLW0603
    if _api is None:
        _api = comet_ml.API()
    return _api


def get_default_workspace() -> str:
    """
    Get the default Comet workspace name from environment or API.

    This function retrieves the default workspace name by first checking the
    ``COMET_DEFAULT_WORKSPACE`` environment variable, and if not set, using
    the first workspace from the user's Comet ML account.

    Returns:
        The default workspace name to use for Comet ML projects.

    Raises:
        ValueError: If no ``COMET_DEFAULT_WORKSPACE`` is set and no workspaces
            are found in the user's Comet ML account.

    Example:

        >>> workspace = get_default_workspace()
        >>> print(f"Using workspace: {workspace}")
        Using workspace: my-default-workspace

        Setting via environment variable::

            >>> import os
            >>> os.environ["COMET_DEFAULT_WORKSPACE"] = "my-custom-workspace"
            >>> get_default_workspace()
            'my-custom-workspace'

    Note:
        It's recommended to set the ``COMET_DEFAULT_WORKSPACE`` environment variable
        rather than relying on the first workspace from the API, as workspace order
        may not be deterministic.
    """
    try:
        return os.environ.get("COMET_DEFAULT_WORKSPACE") or get_comet_api().get_default_workspace()
    except IndexError as e:
        raise ValueError(
            "COMET_DEFAULT_WORKSPACE is not set and no comet workspaces were found. Create a workspace first."
        ) from e


def comet_upload_offline_experiments(tracker: Optional[CometArchiveTracker] = None):
    """Upload offline Comet ML experiments using a tracker instance.

    This convenience function uploads and moves offline experiments using either
    a provided tracker or the default global tracker instance.

    Args:
        tracker: A :class:`CometArchiveTracker` instance to use for uploading.
            If ``None``, uses the default global tracker.

    Example:
        >>> # Upload with default tracker
        >>> comet_upload_offline_experiments()

        >>> # Upload with custom tracker
        >>> custom_tracker = CometArchiveTracker(path="/custom/path")
        >>> comet_upload_offline_experiments(custom_tracker)

    See Also:
        :class:`CometArchiveTracker`: For more control over the upload process
    """
    if tracker is None:
        tracker = _default_comet_archive_tracker
    tracker.upload_and_move()


def comet_assure_project_exists(workspace_name: str, project_name: str, project_description: Optional[str] = None):
    """Ensure a Comet ML project exists, creating it if necessary.

    This function checks if a project exists in the specified workspace and creates
    it if it doesn't exist. This is useful for automated experiment setups where
    you want to ensure the target project is available before starting experiments.

    Args:
        workspace_name: The name of the Comet ML workspace.
        project_name: The name of the project to create or verify.
        project_description: Optional description for the project if it needs to be created.

    Example:
        >>> comet_assure_project_exists(
        ...     workspace_name="my-team",
        ...     project_name="ray-tune-experiments",
        ...     project_description="Hyperparameter optimization with Ray Tune",
        ... )

    Note:
        This function requires appropriate permissions to create projects in the
        specified workspace. If the project already exists, no action is taken.
    """
    api = get_comet_api()
    projects = cast("list[str]", api.get(workspace_name))
    if project_name not in projects:
        api.create_project(
            workspace_name,
            project_name,
            project_description=project_description,
        )


class CometArchiveTracker:
    """Track and manage offline Comet ML experiment archives for batch uploading.

    This class provides functionality to track offline Comet ML experiment archives
    (ZIP files) and upload them in batches when internet connectivity is available.
    It's particularly useful for experiments run on compute clusters or offline
    environments where immediate uploading to Comet ML is not possible.

    The tracker can operate in automatic mode (tracking all new archives in a directory)
    or manual mode (explicitly specifying which archives to track).

    Args:
        track: Optional sequence of archive paths to track initially. If provided,
            these archives will be included in upload operations.
        auto: If ``True`` (default), automatically detect new archives in the
            specified path. If ``False``, only manually added archives are tracked.
        path: Directory path where Comet ML offline archives are stored. Defaults
            to the configured :data:`COMET_OFFLINE_DIRECTORY`.

    Attributes:
        path: The directory path being monitored for archives.
        archives: List of archive paths currently being tracked.

    Example:
        Automatic tracking and upload::

        >>> tracker = CometArchiveTracker(auto=True)
        >>> # ... run experiments that create archives ...
        >>> tracker.upload_and_move()  # Upload all new archives

        Manual tracking::

        >>> tracker = CometArchiveTracker(auto=False)
        >>> tracker.update([Path("experiment1.zip"), Path("experiment2.zip")])
        >>> tracker.upload_and_move()

        Custom directory::

        >>> tracker = CometArchiveTracker(path="/custom/comet/archives")
        >>> tracker.upload_and_move()

    Note:
        Archives are moved to an ``uploaded/`` subdirectory after successful upload
        to avoid re-uploading the same experiments and to keep the main directory clean.

    See Also:
        :func:`comet_upload_offline_experiments`: Convenience function using default tracker
        :data:`ray_utilities.constants.COMET_OFFLINE_DIRECTORY`: Default archive directory
    """

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
        if archives is None:
            archives = self.archives
        if not archives:
            _LOGGER.info("No archives to upload - might have already been moved.")
            return [], []
        archives_str = [str(p) for p in self.archives]
        _LOGGER.info("Uploading Archives: %s", archives_str)
        from comet_ml.offline import LOGGER
        import io
        import re

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        LOGGER.addHandler(handler)

        comet_ml.offline.main_upload(archives_str, force_upload=False)

        LOGGER.removeHandler(handler)
        log_contents = log_stream.getvalue()

        failed_uploads = re.findall(r"Upload failed for '([^']+\.zip)'", log_contents)
        successful_uploads = re.findall(
            r"The offline experiment has been uploaded on comet\.com https://www\.comet\.com/([^/]+/[^/]+/([^ \n]+))",
            log_contents,
        )

        if failed_uploads:
            _LOGGER.warning("Comet offline upload failed for: %s", failed_uploads)
        if successful_uploads:
            _LOGGER.info("Comet offline upload succeeded for: %s", successful_uploads)
        if not failed_uploads and not successful_uploads:
            _LOGGER.warning("Comet offline upload may have failed. Log output:\n%s", log_contents)
        return failed_uploads, successful_uploads

    def upload_and_move(self):
        _failed, succeeded = self._upload()
        self.move_archives(succeeded)

    def make_uploaded_dir(self):
        new_dir = self.path / "uploaded"
        new_dir.mkdir(exist_ok=True)
        return new_dir

    def move_archives(self, succeeded: list[tuple[str, str]] | None = None):
        if not self._called_upload:
            _LOGGER.warning("Called move_archives without calling upload first.")
        new_dir = self.make_uploaded_dir()
        zip_names = [name + ".zip" if not name.endswith(".zip") else name for _, name in (succeeded or [])]
        for path in self.archives:
            if succeeded is None or path.name in zip_names:
                _LOGGER.info("Moving uploaded archive %s to %s", path, new_dir)
                path.rename(new_dir / path.name)
            else:
                _LOGGER.info("Skipping archive %s, not uploaded as not reported as upload succeeded", path)


_default_comet_archive_tracker = CometArchiveTracker()
