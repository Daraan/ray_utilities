"""
Note:
   Creation of loggers is done in _create_default_callbacks which check for inheritance from Callback class.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ray import cloudpickle
from ray.air.constants import EXPR_RESULT_FILE
from ray.tune.experiment.trial import Trial
from ray.tune.logger import JsonLoggerCallback
from ray.tune.utils.util import SafeFallbackEncoder  # pyright: ignore[reportPrivateImportUsage]

from ray_utilities.callbacks.tuner._file_logger_fork_mixin import FileLoggerForkMixin
from ray_utilities.callbacks.tuner.new_style_logger_callback import NewStyleLoggerCallback
from ray_utilities.constants import FORK_FROM
from ray_utilities.postprocessing import remove_videos

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

    from ray_utilities.typing.metrics import AnyLogMetricsDict


logger = logging.getLogger(__name__)


class AdvJsonLoggerCallback(NewStyleLoggerCallback, FileLoggerForkMixin, JsonLoggerCallback):
    """Logs trial results in JSON format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLoggerCallback to be compatible with the ExperimentAnalysis tool.

    This updated class does not log videos stored in the :const:`DEFAULT_VIDEO_DICT_KEYS`.

    When a trial is forked (has ``FORK_FROM`` in config), creates a new JSON file
    and optionally copies data from the parent trial.
    """

    def _get_file_extension(self) -> str:
        return "json"

    def _get_file_base_name(self) -> str:
        return "result"

    def _get_default_file_name(self) -> str:
        return EXPR_RESULT_FILE

    def update_config(self, trial: "Trial", config: dict):
        # Overwrite super method to not use default names.
        # We can call super if this trial is not forked and has never been forked before.
        if not self.is_trial_forked(trial) and FORK_FROM not in trial.config:
            # Use default names
            super().update_config(trial, config)
            return
        self._trial_configs[trial] = config

        trial_local_path = Path(trial.local_path)  # pyright: ignore[reportArgumentType]

        param_file_name_base = f"params-fork-{self.make_forked_trial_id(trial, trial.config[FORK_FROM])}"

        config_out = Path(trial_local_path, param_file_name_base + ".json")
        with config_out.open("w") as f:
            json.dump(
                self._trial_configs[trial],
                f,
                indent=2,
                sort_keys=True,
                cls=SafeFallbackEncoder,
            )

        config_pkl = Path(trial_local_path, param_file_name_base + ".pkl")
        with config_pkl.open("wb") as f:
            cloudpickle.dump(self._trial_configs[trial], f)  # pyright: ignore[reportPrivateImportUsage]

    def _setup_file_handle(self, trial: Trial, local_file_path: Path) -> None:
        """Open the JSON file handle and update config."""
        trial.init_local_path()
        self.update_config(trial, trial.config)
        self._trial_files[trial] = local_file_path.open("at")

    def _handle_missing_parent_file(self, trial: Trial, local_file_path: Path) -> None:
        """Handle missing parent file for JSON logger."""
        logger.warning(
            "Trial %s forked but found no logfile for parent, starting fresh .json log file: %s",
            trial.trial_id,
            local_file_path,
        )

    def log_trial_result(self, iteration: int, trial: Trial, result: dict[str, Any] | AnyLogMetricsDict):
        super().log_trial_result(
            iteration,
            trial,
            remove_videos(result),
        )

    def log_trial_start(self, trial: Trial):
        # NOTE: Because JsonLoggerCallback.log_trial_start is not compatible with forked trials
        # set _call_super_log_trial_start to False to not call it in this chain.
        call_super = True
        if FORK_FROM in trial.config or self.is_trial_forked(trial):
            self._call_super_log_trial_start = call_super = False
        super().log_trial_start(trial)
        assert self._call_super_log_trial_start is True
        if not call_super:
            # As we skip the class after TrackForkedTrialsMixin, continue after the original base.
            super(JsonLoggerCallback, self).log_trial_start(trial)


if TYPE_CHECKING:  # Check ABC
    AdvJsonLoggerCallback()
