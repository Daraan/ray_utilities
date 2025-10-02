"""Common base class for file-based logger callbacks that support forked trials."""

from __future__ import annotations

import logging
import shutil
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from ray.tune.experiment.trial import Trial

from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.constants import FORK_FROM

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

    def _get_parent_file_name(self, parent_trial: Trial) -> str:
        """Get the file name for the parent trial.

        Args:
            parent_trial: The parent trial

        Returns:
            File name of the parent trial (default or fork-specific)
        """
        parent_fork_id = self._current_fork_ids.get(parent_trial, None)
        if parent_fork_id is None:
            return self._get_default_file_name()
        return self._make_forked_trial_file_name(parent_trial, parent_fork_id)

    def _sync_parent_file(
        self, trial: Trial, parent_trial: Trial, parent_file_name: str, local_file_path: Path
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

        # Same node - use local copy
        if trial.node_ip == parent_trial.node_ip and parent_local_file_path.exists():
            shutil.copy2(parent_local_file_path, local_file_path)
            return True

        # Different nodes - sync via remote storage
        if trial.storage and parent_trial.storage:
            try:
                parent_remote_file_path = (Path(parent_trial.path) / parent_file_name).as_posix()  # pyright: ignore[reportArgumentType]
                logger.debug(
                    "Syncing up parent %s file to %s", self._get_file_extension().upper(), parent_remote_file_path
                )
                parent_trial.storage.syncer.sync_up(
                    parent_local_file_path.as_posix(),
                    parent_remote_file_path,
                    exclude=["*/checkpoint_*", "*.pkl", "events.out.tfevents.*"],
                )
                parent_trial.storage.syncer.wait()
                logger.debug(
                    "Syncing down parent %s file to %s", self._get_file_extension().upper(), local_file_path.as_posix()
                )
                trial.storage.syncer.sync_down(
                    parent_remote_file_path,
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
        assert not local_file_path.exists(), "File should not exist yet"

        # Try to sync from parent trial
        file_copied = False
        if (parent_trial := fork_data.get("parent_trial")) or isinstance(
            parent_trial := self.parent_trial_lookup.get(trial), Trial
        ):
            parent_file_name = self._get_parent_file_name(parent_trial)
            file_copied = self._sync_parent_file(trial, parent_trial, parent_file_name, local_file_path)

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
