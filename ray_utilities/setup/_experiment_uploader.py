from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generic, Optional, TypeAlias

from typing_extensions import Sentinel, TypeVar

from ray_utilities.callbacks.comet import CometArchiveTracker
from ray_utilities.callbacks.wandb import WandbUploaderMixin

# pyright: enableExperimentalFeatures=true


if TYPE_CHECKING:
    import argparse

    from ray import tune
    from ray.tune import ResultGrid

    from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser


logger = logging.getLogger(__name__)

ParserType_co = TypeVar("ParserType_co", bound="DefaultArgumentParser", covariant=True, default="DefaultArgumentParser")
"""TypeVar for the ArgumentParser type of a Setup, bound and defaults to DefaultArgumentParser."""

NamespaceType: TypeAlias = "argparse.Namespace | ParserType_co"  # Generic, formerly union with , prefer duck-type

_ATTRIBUTE_NOT_FOUND = Sentinel("_ATTRIBUTE_NOT_FOUND")


class CometUploaderMixin(Generic[ParserType_co]):
    comet_tracker: CometArchiveTracker | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setup_args: NamespaceType[ParserType_co] | _ATTRIBUTE_NOT_FOUND = getattr(self, "args", _ATTRIBUTE_NOT_FOUND)
        if setup_args is _ATTRIBUTE_NOT_FOUND:
            logger.info(
                "No args attribute found, likely due to `parse_args=False`, "
                "cannot initialize comet tracker. Need to be setup later manually if desired."
            )
            self.comet_tracker = None
        elif setup_args.comet:
            self.comet_tracker = CometArchiveTracker()
        else:
            self.comet_tracker = None

    def comet_upload_offline_experiments(self):
        """Note this does not check for args.comet"""
        if self.comet_tracker is None:
            if not hasattr(self, "args") or str(self.args.comet).lower() in ("false", "none", "0"):  # pyright: ignore[reportAttributeAccessIssue]
                logger.debug("No comet tracker / args.comet defined. Will not upload offline experiments.")
            else:
                logger.warning(
                    "No comet tracker setup but args.comet=%s. Cannot upload experiments. Upload them manually instead.",
                    self.args.comet,  # pyright: ignore[reportAttributeAccessIssue]
                )
            return
        self.comet_tracker.upload_and_move()


class ExperimentUploader(WandbUploaderMixin, CometUploaderMixin[ParserType_co]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args: NamespaceType[ParserType_co]

    def upload_offline_experiments(self, results: Optional[ResultGrid] = None, tuner: Optional[tune.Tuner] = None):
        unfinished_wandb_uploads = None
        if self.args.wandb and "upload" in self.args.wandb:
            if results is None:
                logger.warning(
                    "Wandb upload requested, but no results provided. This will not upload any offline experiments."
                )
            try:  # if no results (due to a failure) get them in a more hacky way.
                # Do not wait to start uploading to comet.
                unfinished_wandb_uploads = self.wandb_upload_results(results, tuner, wait=False)
            except Exception:
                logger.exception("Error while uploading offline experiments to WandB: %s")
        if self.args.comet and "upload" in self.args.comet:
            logger.info("Uploading offline experiments to Comet")
            try:
                self.comet_upload_offline_experiments()
            except Exception:
                logger.exception("Error while uploading offline experiments to Comet")
        failed_runs = []
        if unfinished_wandb_uploads:
            for process in unfinished_wandb_uploads:
                exit_code = self._failure_aware_wait(process, timeout=900, trial_id="")
                if exit_code != 0:
                    try:
                        failed_runs.append(" ".join(process.args))  # pyright: ignore[reportArgumentType, reportCallIssue]
                    except TypeError:
                        failed_runs.append(str(process.args))
        if failed_runs:
            logger.error("Failed to upload the following wandb runs. Commands to run:\n%s", "\n".join(failed_runs))
