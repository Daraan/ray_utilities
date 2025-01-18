# ruff: noqa: PLC0415

import logging
import os
from pathlib import Path
from typing import Optional, cast

import comet_ml

from interpretable_ddts.runfiles.constants import COMET_OFFLINE_DIRECTORY

_api: Optional[comet_ml.API] = None
"""Singleton instance of the Comet API client to make use of caching."""

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
        return (
            os.environ.get("COMET_DEFAULT_WORKSPACE")
            or get_comet_api().get_default_workspace()
        )
    except IndexError as e:
        raise ValueError(
            "COMET_DEFAULT_WORKSPACE is not set and no comet workspaces were found. Create a workspace first."
        ) from e


def comet_upload_offline_experiments():

    archives = list(map(str, Path(COMET_OFFLINE_DIRECTORY).glob("*.zip")))
    if not archives:
        logging.info("No archives to upload")
        return
    logging.info("Uploading Archives: %s", archives)
    comet_ml.offline.main_upload(archives, force_upload=False)
    new_dir = Path(COMET_OFFLINE_DIRECTORY) / "uploaded"
    new_dir.mkdir(exist_ok=True)
    for path in Path(os.environ["COMET_OFFLINE_DIRECTORY"]).glob("*.zip"):
        path.rename(new_dir / path.name)

def comet_assure_project_exists(workspace_name: str, project_name: str, project_description: Optional[str] = None):
    api = get_comet_api()
    projects = cast(list[str], api.get(workspace_name))
    if project_name not in projects:
        api.create_project(
            workspace_name,
            project_name,
            project_description=project_description,
        )
