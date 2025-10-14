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
# pyright: reportPossiblyUnboundVariable=information
from __future__ import annotations

from contextlib import contextmanager
import io
import logging
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Literal, Optional, Sequence, cast, overload

try:
    import comet_ml
except ImportError:
    pass

from ray_utilities.callbacks.upload_helper import UploadHelperMixin
from ray_utilities.constants import COMET_OFFLINE_DIRECTORY

_api: Optional[comet_ml.API] = None
"""
Singleton instance of the Comet API client to make use of caching.

Use :func:`get_comet_api` to access this instance
and initialize it if it first if it is not already created.
"""

__all__ = [
    "COMET_OFFLINE_DIRECTORY",
    "CometArchiveTracker",
    "comet_assure_project_exists",
    "comet_upload_offline_experiments",
    "get_comet_api",
    "get_default_workspace",
]

_LOGGER = logging.getLogger(__name__)
_COMET_OFFLINE_LOGGER = logging.getLogger("comet_ml.offline")

_failed_upload_file_lock = threading.Lock()

COMET_COLOR_STRINGS = {
    "COMET INFO": "\x1b[1;38;5;39mCOMET INFO:\x1b[0m",
    "COMET WARNING": "\x1b[1;38;5;214mCOMET WARNING:\x1b[0m",
    "COMET ERROR": "\x1b[1;38;5;196mCOMET ERROR:\x1b[0m",
}
"""Colored log level strings for Comet ML console output."""


def color_comet_log_strings(log_str: str) -> str:
    """Add ANSI color codes to Comet ML log level strings in the given text.

    This function replaces standard Comet ML log level prefixes with colored
    versions for improved console readability.

    Args:
        log_str: The input log string potentially containing Comet ML log levels.

    Returns:
        The input log string with colored Comet ML log level prefixes.
    """
    for level, color in COMET_COLOR_STRINGS.items():
        log_str = log_str.replace(level, color)
    return log_str


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
        _api = comet_ml.API()  # pyright: ignore[reportPossiblyUnboundVariable]
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


@contextmanager
def _catch_comet_offline_logger():
    """Context manager to temporarily add a stream handler to the comet_ml logger and yield the log stream."""
    from comet_ml.offline import LOGGER as COMET_LOGGER

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    COMET_LOGGER.addHandler(handler)
    try:
        yield log_stream
    finally:
        COMET_LOGGER.removeHandler(handler)


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


class CometArchiveTracker(UploadHelperMixin):
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
        self._wait_and_move_threads: list[threading.Thread] = []

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

    def _check_already_uploaded(self):
        for archive in self.archives.copy():
            if not archive.exists() and (archive.parent / "uploaded" / archive.name).exists():
                _LOGGER.info(
                    "Archive %s has already been uploaded and moved to %s",
                    archive,
                    archive.parent / "uploaded" / archive.name,
                )
                self.archives.remove(archive)

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
        self._check_already_uploaded()
        archives_str = [str(p) for p in self.archives]
        _LOGGER.info("Uploading Archives: %s", archives_str)

        try:
            with _catch_comet_offline_logger() as log_stream:
                comet_ml.offline.main_upload(archives_str, force_upload=False)  # pyright: ignore[reportPossiblyUnboundVariable]
        except comet_ml.exceptions.OfflineExperimentUploadFailed as e:
            _LOGGER.error("Comet offline upload failed with exception: %s", e)
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
        failed_uploads, succeeded = self._upload()
        self._write_failed_upload_file(failed_uploads)
        self.move_archives(succeeded)

    def _write_failed_upload_file(self, failed_uploads: list[str]) -> None:
        """Write details of failed comet uploads to a file in the experiment directory."""
        if not failed_uploads:
            return
        # Use the configured COMET_OFFLINE_DIRECTORY for failed upload file location
        failed_file = Path(COMET_OFFLINE_DIRECTORY) / "failed_comet_uploads.txt"
        # Write failed archive names to file
        with _failed_upload_file_lock:
            with failed_file.open("a") as f:
                for archive_name in failed_uploads:
                    f.write(f"{archive_name}\n")
        _LOGGER.warning("Wrote details of failed comet uploads to %s", failed_file.resolve())

    def make_uploaded_dir(self) -> Path:
        new_dir = self.path / "uploaded"
        new_dir.mkdir(exist_ok=True)
        return new_dir

    def move_archives(self, succeeded: list[tuple[str, str]] | None = None, *, _suppress_upload_warning: bool = False):
        if not self._called_upload and not _suppress_upload_warning:
            _LOGGER.warning("Called move_archives without calling upload first.")
        new_dir = self.make_uploaded_dir().absolute()
        zip_names = [name + ".zip" if not name.endswith(".zip") else name for _, name in (succeeded or [])]
        for path in self.archives:
            if succeeded is None or path.name in zip_names:
                _LOGGER.info("Moving uploaded archive %s to %s", path, new_dir)
                start_time = time.time()
                path.rename(new_dir / path.name)
                _LOGGER.debug("Moved archive %s in %.2f seconds", path, time.time() - start_time)
            else:
                _LOGGER.info("Skipping archive %s, not uploaded as not reported as upload succeeded", path)

    @classmethod
    def _finish_upload_by_process(
        cls,
        process: subprocess.Popen[str] | subprocess.CompletedProcess[str],
        zip_file,
        *,
        move: bool,
        timeout: int | None = None,
    ) -> bool:
        # Note, comet writes its messages to stderr, need to check for errors in the contents.
        if isinstance(process, subprocess.Popen):
            if timeout is not None:
                cls._failure_aware_wait(process, timeout=timeout, report_upload=False)
                if process.poll() is None:
                    _LOGGER.error("Timeout expired while waiting for comet upload process to finish")
                    return False
            done = process.poll() is not None
            if not done:
                _LOGGER.error("Should only call _finish_upload with finished processes", stacklevel=2)
            stdout = process.stdout.read() if process.stdout else "No stdout"
            stderr = process.stderr.read() if process.stderr else ""
        else:
            stdout = process.stdout
            stderr = process.stderr
        success = (
            process.returncode == 0
            and not any(pattern in stderr.lower() for pattern in map(str.lower, cls.error_patterns))
            and not any(pattern in stdout.lower() for pattern in map(str.lower, cls.error_patterns))
        )
        stdout = color_comet_log_strings(stdout)
        stderr = color_comet_log_strings(stderr)
        if success:
            _COMET_OFFLINE_LOGGER.info("Successfully uploaded to comet:\n%s", stdout)
        else:
            _COMET_OFFLINE_LOGGER.error("Error while uploading to comet:\n%s", stdout)
        if process.stderr:
            _COMET_OFFLINE_LOGGER.error("Error while uploading to comet:\n%s", stderr)
        if not move or not success:
            return success
        zip_path = Path(zip_file)
        (zip_path.parent / "uploaded").mkdir(exist_ok=True)
        new_zip_path = zip_path.rename(zip_path.parent / "uploaded" / zip_path.name)
        _LOGGER.debug("Moved uploaded archive %s to %s", zip_path, new_zip_path)
        return success

    @overload
    @classmethod
    def upload_zip_file(cls, zip_file: str, *, blocking: Literal[True] = True, move: bool = True) -> bool: ...

    @overload
    @classmethod
    def upload_zip_file(cls, zip_file: str, *, blocking: Literal[False]) -> subprocess.Popen[str]: ...

    @classmethod
    def upload_zip_file(
        cls, zip_file: str, *, blocking: bool = True, move: bool = True, tracker: Optional[CometArchiveTracker] = None
    ) -> bool | subprocess.Popen[str]:
        """
        Uploads a single Comet ML offline experiment ZIP file.

        Args:
            zip_file: Path to the Comet ML offline experiment ZIP file.
            blocking: If True, creates a `comet upload ...` subprocess and waits for it to finish.
                If False, starts the subprocess and returns it immediately.
            move: If True, moves the ZIP file to an `uploaded/` subdirectory after successful upload.
                If blocking is False and move is True, starts a thread to wait for the process to finish
                and then move the file.
            tracker: Optional CometArchiveTracker instance to track the upload thread if move is True
                and blocking is False. If None, uses the default global tracker.
        """
        if blocking:
            process = subprocess.Popen(
                ["comet", "upload", zip_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            cls._failure_aware_wait(process, timeout=600, report_upload=False)
            return cls._finish_upload_by_process(process, zip_file, move=move, timeout=None)
        # Note, comet writes output to stderr
        process = subprocess.Popen(
            ["comet", "upload", zip_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if move:
            # Start a thread to wait for the process to finish and then move the file
            wait_and_move_thread = threading.Thread(
                target=cls._finish_upload_by_process,
                args=(process, zip_file),
                kwargs={"move": True, "timeout": 1800},
                daemon=False,
                name=f"Thread-_finish_upload_by_process({zip_file})",
            )
            wait_and_move_thread.start()
            if tracker is None:
                tracker = _default_comet_archive_tracker
            tracker._wait_and_move_threads.append(wait_and_move_thread)
        return process

    def __del__(self):
        for thread in self._wait_and_move_threads:
            try:
                if thread.is_alive():
                    _LOGGER.info(
                        "Waiting 5s for Comet offline upload to finish to move files (%s daemon=%s)",
                        thread.name,
                        thread.daemon,
                    )
                    thread.join(timeout=5)
                    if thread.is_alive():
                        _LOGGER.warning(
                            "Comet offline upload thread did not finish in time (%s daemon=%s)",
                            thread.name,
                            thread.daemon,
                        )
            except (RuntimeError, Exception):  # noqa: PERF203
                _LOGGER.exception("Error while waiting for Comet offline upload thread to finish (%s)", thread.name)


_default_comet_archive_tracker = CometArchiveTracker()
