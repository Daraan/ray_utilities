"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional

from ray.air.constants import EXPR_PROGRESS_FILE
from ray.tune.logger import CSVLoggerCallback

from ray_utilities.callbacks.tuner._file_logger_fork_mixin import FileLoggerForkMixin
from ray_utilities.callbacks.tuner.new_style_logger_callback import NewStyleLoggerCallback
from ray_utilities.constants import DEFAULT_EVAL_METRIC
from ray_utilities.misc import resolve_default_eval_metric
from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from pathlib import Path

    from ray_utilities.typing import ForkFromData
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import AnyLogMetricsDict


logger = logging.getLogger(__name__)


class AdvCSVLoggerCallback(NewStyleLoggerCallback, FileLoggerForkMixin, CSVLoggerCallback):
    """Logs trial results in CSV format.

    Prevents logging of videos (keys in :const:`DEFAULT_VIDEO_DICT_KEYS`) even if they are present
    at the first iteration.

    When a trial is forked (has ``FORK_FROM`` in config), creates a new CSV file
    and tries to copy the data from the parent trial.

    Args:
        metric: Core metric to monitor. Adding it here ensures that it is present in the output CSV file.
            as only the columns present in the first logged result are written to the CSV file.
    """

    def __init__(
        self,
        *args,
        metric: Optional[str | DEFAULT_EVAL_METRIC] = DEFAULT_EVAL_METRIC,
        **kwargs,
    ) -> None:
        # cannot pickle DEFAULT_EVAL_METRIC use string
        if metric is DEFAULT_EVAL_METRIC:
            metric = "DEFAULT_EVAL_METRIC"
            # Cannot serialize attribute with DEFAULT_EVAL_METRIC directly
        self.metric: str | Literal["DEFAULT_EVAL_METRIC"] | DEFAULT_EVAL_METRIC | None = metric  # noqa: PYI051
        super().__init__(*args, **kwargs)

    def _get_file_extension(self) -> str:
        return "csv"

    def _get_file_base_name(self) -> str:
        return "progress"

    def _get_default_file_name(self) -> str:
        return EXPR_PROGRESS_FILE

    def _setup_file_handle(self, trial: Trial, local_file_path: Path) -> None:
        """Open the CSV file handle and set CSV-specific state."""
        self._trial_csv[trial] = None  # need to set key # pyright: ignore[reportArgumentType]
        if local_file_path.exists():
            # Check if header needs to be written
            write_header_again = local_file_path.stat().st_size == 0
            # Note: metrics might have changed when loading from checkpoint
            if trial.config.get("cli_args", {}).get("from_checkpoint") or trial.config.get("from_checkpoint"):
                write_header_again = True
            self._trial_continue[trial] = not write_header_again
        else:  # For now this is not called as we enter after a sync from parent
            self._restore_from_remote(local_file_path.name, trial)
            self._trial_continue[trial] = False
        self._trial_files[trial] = local_file_path.open("at")

    def _handle_missing_parent_file(self, trial: Trial, local_file_path: Path) -> None:
        """Handle missing parent file for CSV logger."""
        self._trial_continue[trial] = False
        logger.warning(
            "Trial %s forked but found no logfile for parent, starting fresh .csv log file: %s",
            trial.trial_id,
            local_file_path,
        )

    def _trim_history_back_to_fork_step(self, trial: Trial, copied_file: Path, fork_data: ForkFromData) -> None:
        """When the history file is copied, the parent trial might have already continued,
        need to trim the logged history back to the fork step.
        """
        fork_step = fork_data["parent_training_iteration"]
        # remove all lines after fork step
        temp_file = copied_file.with_suffix(".tmp")
        with copied_file.open("r") as infile, temp_file.open("w") as outfile:
            for i, line in enumerate(infile):
                # we have a csv file here and we cannot be 100% sure about the header.
                # we keep the header + fork_step lines
                if i == 0 or i <= fork_step:
                    outfile.write(line)
                else:
                    break
        temp_file.replace(copied_file)

    def log_trial_result(self, iteration: int, trial: "Trial", result: AnyLogMetricsDict):  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.metric:
            if self.metric is DEFAULT_EVAL_METRIC or self.metric == "DEFAULT_EVAL_METRIC":
                for ema in (True, False):
                    ema_variant = resolve_default_eval_metric(for_logger=True, use_ema=ema)
                    if ema_variant not in result:
                        result[ema_variant] = float("nan")
                # TODO: hardcoded True, but not that important here, we just have to make sure the key is in the header
                self.metric = resolve_default_eval_metric(for_logger=True, use_ema=True)
            if self.metric not in result:
                result[self.metric] = float("nan")
        if trial not in self._trial_csv:
            # Keys are permanently set; remove videos from the first iteration.
            # Therefore also need eval metric in first iteration
            result = remove_videos(result)

        super().log_trial_result(
            iteration,
            trial,
            result,
        )


if TYPE_CHECKING:  # Check ABC
    AdvCSVLoggerCallback()
