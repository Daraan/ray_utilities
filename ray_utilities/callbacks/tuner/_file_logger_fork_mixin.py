"""Common base class for file-based logger callbacks that support forked trials."""

from __future__ import annotations

import logging
import shutil
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TextIO

import pyarrow.fs
from ray.tune.experiment.trial import Trial

from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import ExperimentKey

if TYPE_CHECKING:
    from ray_utilities.typing import ForkFromData


logger = logging.getLogger(__name__)


class FileLoggerForkMixin(TrackForkedTrialsMixin):
    """Mixin for file-based loggers that need to handle forked trials.

    This mixin provides common functionality for CSV and JSON loggers to:
    - Create fork-specific file names
    - Sync parent trial data (local or remote)
    - Handle checkpoint-based loading

    Subclasses must implement:
    - _get_file_extension() - return the file extension (e.g., "csv", "json")
    - _get_file_base_name() - return the base name without fork suffix (e.g., "progress", "result")
    - _get_default_file_name() - return the default file name constant (e.g., EXPR_PROGRESS_FILE)
    - _setup_file_handle() - open the file handle and set any additional state
    """

    @abstractmethod
    def _get_file_extension(self) -> str:
        """Return the file extension (e.g., 'csv', 'json')."""
        ...

    @abstractmethod
    def _get_file_base_name(self) -> str:
        """Return the base file name without fork suffix (e.g., 'progress', 'result')."""
        ...

    @abstractmethod
    def _get_default_file_name(self) -> str:
        """Return the default file name constant (e.g., EXPR_PROGRESS_FILE, EXPR_RESULT_FILE)."""
        ...

    @abstractmethod
    def _setup_file_handle(self, trial: Trial, local_file_path: Path) -> None:
        """Open the file handle and set any additional state for the trial.

        Args:
            trial: The trial to set up
            local_file_path: Path to the file to open
        """
        ...

    @abstractmethod
    def _handle_missing_parent_file(self, trial: Trial, local_file_path: Path) -> None:
        """Handle the case when no parent file was found.

        Args:
            trial: The trial being set up
            local_file_path: Path where the file should be
        """
        ...

    @abstractmethod
    def _trim_history_back_to_fork_step(self, trial: Trial, copied_file: Path, fork_data: ForkFromData) -> None:
        """When the history file is copied, the parent trial might have already continued,
        need to trim the logged history back to the fork step.

        Args:
            trial: The trial being set up
            fork_step: The global step at which the trial was forked
        """
        ...

    _trial_files: dict[Trial, TextIO]
    """Mapping of trials to their open file handles or paths. Used to track if trial is actively logged"""

    def _make_forked_trial_file_name(self, trial: Trial, fork_data: ForkFromData | str) -> str:
        """Create the file name for a forked trial.

        Args:
            trial: The trial being forked
            fork_data: Fork data or fork ID string

        Returns:
            File name for the forked trial (e.g., "progress-fork-{fork_id}.csv")
        """
        fork_info = fork_data if isinstance(fork_data, str) else self.make_forked_trial_id(trial, fork_data)
        return f"{self._get_file_base_name()}-fork-{fork_info}.{self._get_file_extension()}"

    def _get_parent_file_name(self, parent_trial: Trial, trial_id_history: Optional[dict[str, str]] = None) -> str:
        """Get the file name for the parent trial.

        Args:
            parent_trial: The parent trial

        Returns:
            File name of the parent trial (default or fork-specific)
        """
        # NOTE: When the parent_trial went ahead and is terminated we should not clean _current_fork_ids
        # With access to trial.config["trial_id_history"] we also do not need it
        if trial_id_history is not None:
            hist_steps = sorted(int(k) for k in trial_id_history.keys() if k.isdigit())
            if len(hist_steps) < 2:
                logger.error(
                    "trial_id_history should have at least two entries when it should is a fork. "
                    "Using _current_fork_ids fallback"
                )
                parent_fork_id = self._current_fork_ids.get(parent_trial, None)
            else:
                parent_fork_id = trial_id_history[str(hist_steps[-2])]
                parent_fork_id_alt = self._current_fork_ids.get(parent_trial, None)
                if parent_fork_id_alt and parent_fork_id != parent_fork_id_alt:
                    logger.warning(
                        "Mismatch in parent fork_id from trial_id_history (%s) and current_fork_ids (%s) "
                        "for trial %s. Using trial_id_history value.",
                        parent_fork_id,
                        parent_fork_id_alt,
                        parent_trial.trial_id,
                    )
                if ExperimentKey.FORK_SEPARATOR not in parent_fork_id:
                    return self._get_default_file_name()
        else:
            parent_fork_id = self._current_fork_ids.get(parent_trial, None)
        if parent_fork_id is None:
            return self._get_default_file_name()
        return self._make_forked_trial_file_name(parent_trial, parent_fork_id)

    def _sync_parent_file(
        self, trial: Trial, parent_trial: Trial, parent_file_name: str, local_file_path: Path, fork_data: ForkFromData
    ) -> bool:
        """Sync parent trial's file to the forked trial's location.

        Args:
            trial: The forked trial
            parent_trial: The parent trial
            parent_file_name: Name of the parent's file
            local_file_path: Path where to copy the file

        Returns:
            True if file was successfully copied, False otherwise
        """
        parent_local_file_path = Path(parent_trial.local_path, parent_file_name)  # pyright: ignore[reportArgumentType]
        if local_file_path.parent.exists() and local_file_path.exists() and local_file_path.stat().st_size > 0:
            logger.warning(
                "Trial %s forked but log file %s already exists. "
                "This is unexpected, expect when restoring an experiment. "
                "Creating a .parent file and not modifying the existing file.",
                trial.trial_id,
                local_file_path,
            )
            local_parent_copy = local_file_path.with_suffix(".parent.json")
            try:
                shutil.copy2(parent_local_file_path, local_parent_copy)
            except OSError as err:
                logger.error("Error copying parent file to .parent file: %s", err)
            else:
                try:
                    self._trim_history_back_to_fork_step(trial, local_parent_copy, fork_data)
                except Exception:  # noqa: BLE001
                    import traceback  # noqa: PLC0415

                    logger.exception(
                        "Error trimming copied parent file for trial %s from trial %s",
                        trial.trial_id,
                        parent_trial.trial_id,
                    )
                    with Path(local_file_path.parent, "ru_errors.log").open("a") as err_log:
                        err_log.write(
                            f"Error trimming copied parent file for trial {trial.trial_id} "
                            f"from trial {parent_trial.trial_id}\n" + traceback.format_exc()
                        )
            # Did not really copy it but as it already exists, consider it done
            return True

        # Same node - use local copy
        if parent_local_file_path.exists() and local_file_path.parent.exists():
            try:
                shutil.copy2(parent_local_file_path, local_file_path)
            except OSError as err:
                logger.error("Error copying parent file locally: %s", err)
                if err.errno and err.errno == 28:  # No space left on device
                    stat_result = shutil.disk_usage(local_file_path.parent)
                    # Get disk space information
                    logger.error(
                        "No space left on device when copying parent file for trial %s from trial %s. "
                        "Disk space on %s - Total: %s, Used: %s, Free: %s",
                        trial.trial_id,
                        parent_trial.trial_id,
                        local_file_path.parent,
                        stat_result.total,
                        stat_result.used,
                        stat_result.free,
                    )
                return False
            else:
                # When the parent trial has continued, we need to trim the copied file to match the fork step
                try:
                    self._trim_history_back_to_fork_step(trial, local_file_path, fork_data)
                except Exception:  # noqa: BLE001
                    import traceback  # noqa: PLC0415

                    logger.exception(
                        "Error trimming copied parent file for trial %s from trial %s",
                        trial.trial_id,
                        parent_trial.trial_id,
                    )
                    with Path(local_file_path.parent, "ru_errors.log").open("a") as err_log:
                        err_log.write(
                            f"Error trimming copied parent file for trial {trial.trial_id} "
                            f"from trial {parent_trial.trial_id}\n" + traceback.format_exc()
                        )
            return True

        # Different nodes - sync via remote storage
        if trial.storage and parent_trial.storage:
            try:
                parent_remote_file_path = Path(parent_trial.path) / parent_file_name  # pyright: ignore[reportArgumentType]
                logger.debug(
                    "Syncing up parent %s file to %s", self._get_file_extension().upper(), parent_remote_file_path
                )
                # prevent a pyarrow exception
                if parent_remote_file_path.exists():
                    parent_remote_file_path.rename(
                        parent_remote_file_path.parent / (parent_remote_file_path.name + ".old")
                    )
                try:
                    try:
                        pyarrow.fs.copy_files(
                            parent_local_file_path.as_posix(),
                            parent_remote_file_path.as_posix(),
                            source_filesystem=None,
                            destination_filesystem=trial.storage.storage_filesystem,
                            chunk_size=512 * 1024,
                            use_threads=False,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.info("Exception while coppying file: %s. Falling back to different method", e)
                        # sync_up does not support single files, this might create a directory
                        parent_trial.storage.syncer.sync_up(
                            parent_local_file_path.as_posix(),
                            parent_remote_file_path.as_posix(),
                            exclude=["*/checkpoint_*", "*.pkl", "events.out.tfevents.*"],
                        )
                        parent_trial.storage.syncer.wait()
                except FileNotFoundError as fnf_error:
                    # When restoring local file does not exist
                    if parent_remote_file_path.exists() and parent_remote_file_path.is_dir():
                        # Failed sync
                        shutil.rmtree(parent_remote_file_path.as_posix())
                    if not parent_remote_file_path.exists():
                        logger.warning(
                            "file %s does not exist for trial %s. This can happen when restoring from a checkpoint.",
                            fnf_error.filename,
                            trial.trial_id,
                        )
                        # restore old
                        Path(parent_remote_file_path.parent, parent_remote_file_path.name + ".old").resolve().rename(
                            Path(parent_remote_file_path.parent, parent_remote_file_path.name).resolve()
                        )

                logger.debug(
                    "Syncing down parent %s file to %s", self._get_file_extension().upper(), local_file_path.as_posix()
                )
                trial.storage.syncer.sync_down(
                    parent_remote_file_path.as_posix(),
                    local_file_path.as_posix(),
                    exclude=["*/checkpoint_*", "*.pkl", "events.out.tfevents.*"],
                )
                trial.storage.syncer.wait()
            except (RuntimeError, OSError, Exception):
                logger.exception(
                    "Trial %s forked from %s but could not copy parent %s data from remote storage.",
                    trial.trial_id,
                    parent_trial.trial_id,
                    self._get_file_extension().upper(),
                )
                return False
            else:
                return True

        # No storage backend available
        logger.warning(
            "Trial %s forked from %s but could not copy parent %s data, no storage backend.",
            trial.trial_id,
            parent_trial.trial_id,
            self._get_file_extension().upper(),
        )
        return False

    def _handle_checkpoint_loading(self, trial: Trial, local_file_path: Path) -> bool:
        """Handle loading from checkpoint.

        Args:
            trial: The trial being set up
            local_file_path: Path where to copy the file

        Returns:
            True if file was copied from checkpoint, False otherwise
        """
        checkpoint_path = trial.config.get("cli_args", {}).get("from_checkpoint", None) or trial.config.get(
            "from_checkpoint", None
        )
        if checkpoint_path is None:
            return False

        # TODO: Which file to take when trial was forked? We know the file is in the parent folder of the
        # checkpoint. We can take it ONLY if there is one file, AND need to trim it to the step we are loading.
        parent_dir = Path(checkpoint_path).parent
        if not parent_dir.exists():
            return False

        ext = self._get_file_extension()
        base_name = self._get_file_base_name()
        file_pattern = f"{base_name}*.{ext}"
        result_files = list(parent_dir.glob(file_pattern))

        if len(result_files) == 1:
            shutil.copy2(result_files[0], local_file_path)
            # TODO: trim to step that is loaded
            logger.warning(
                "(TODO) Could copy %s file from checkpoint, but it is not trimmed to loaded step yet.",
                ext.upper(),
            )
            return True

        if len(result_files) > 1:
            logger.info(
                "(TODO) Trial %s forked but found multiple %s files in checkpoint. "
                "We do not know which to take for the fork, creating a new one. Files: %s",
                trial.trial_id,
                ext.upper(),
                [f.name for f in result_files],
            )
        return False

    def _setup_forked_trial(self, trial: Trial, fork_data: ForkFromData):
        """Setup trial logging, handling forked trials by creating new files.

        Args:
            trial: The trial being forked
            fork_data: Fork data from trial.config
        """
        # Close current file and clean tracks
        self.log_trial_end(trial)

        # Make sure logdir exists
        file_name = self._make_forked_trial_file_name(trial, fork_data)
        trial.init_local_path()
        local_file_path = Path(trial.local_path, file_name)  # pyright: ignore[reportArgumentType]
        if local_file_path.exists():
            # Might exist if we are restoring from a previous experiment
            logger.warning(
                "Trial %s forked but log file %s already exists. "
                "This is unexpected, except when restoring an experiment",
                trial.trial_id,
                local_file_path,
            )

        # Try to sync from parent trial
        file_copied = None
        if (parent_trial := (fork_data.get("parent_trial") or self.parent_trial_lookup.get(trial))) and isinstance(
            parent_trial, Trial
        ):
            parent_file_name = self._get_parent_file_name(parent_trial, trial.config.get("trial_id_history"))
            file_copied = self._sync_parent_file(trial, parent_trial, parent_file_name, local_file_path, fork_data)
        if False and file_copied is None and (parent_fork_id := fork_data.get("parent_fork_id")):
            # TODO:
            # We do not have parent_trial here, we only need if for the path and could construct it
            # locally from trial
            if ExperimentKey.FORK_SEPARATOR not in parent_fork_id:
                parent_file_name = self._get_default_file_name()
            else:
                parent_file_name = self._make_forked_trial_file_name(parent_trial, parent_fork_id)
            file_copied = self._sync_parent_file(trial, parent_trial, parent_file_name, local_file_path, fork_data)

        # Try to load from checkpoint if parent trial sync didn't work
        if not file_copied:
            file_copied = self._handle_checkpoint_loading(trial, local_file_path)

        # Handle missing parent file
        if not file_copied or not local_file_path.exists():
            self._handle_missing_parent_file(trial, local_file_path)

        # Set up the file handle
        self._setup_file_handle(trial, local_file_path)

    def log_trial_start(self, trial: Trial):
        if trial in self._trial_files and FORK_FROM in trial.config:
            assert self.should_restart_logging(trial)
        if FORK_FROM in trial.config:
            self._setup_forked_trial(trial, trial.config[FORK_FROM])
        return super().log_trial_start(trial)

    def _get_stateXX(self) -> dict:
        """Get the state of the callback for checkpointing.

        Returns:
            Dictionary containing file paths instead of file handles.
            File handles cannot be pickled and will need to be reopened.
        """
        state = super().get_state() if hasattr(super(), "get_state") else {}
        # Store file paths instead of file handles
        state["trial_file_paths"] = {
            trial.trial_id: (file_handle.name if hasattr(file_handle, "name") else str(file_handle))
            for trial, file_handle in self._trial_files.items()
        }
        return state

    def _set_stateXX(self, state: dict) -> None:
        """Set the state of the callback from checkpoint data.

        Args:
            state: State dictionary containing file paths.

        Note:
            File handles will not be restored immediately. They will be recreated
            when trials restart and call log_trial_start.
        """
        if hasattr(super(), "set_state"):
            super().set_state(state)

        # We don't restore file handles here - they will be recreated in log_trial_start
        # when trials are restarted. Just log the information.
        trial_file_paths = state.get("trial_file_paths", {})
        if trial_file_paths:
            logger.info("FileLoggerForkMixin state restored with %d trial file paths", len(trial_file_paths))
            logger.debug("Trial file paths to be reopened: %s", trial_file_paths)
