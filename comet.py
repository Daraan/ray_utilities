# ruff: noqa: PLC0415

import logging
import os
from pathlib import Path

import comet_ml

from interpretable_ddts.runfiles.constants import COMET_OFFLINE_DIRECTORY


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
