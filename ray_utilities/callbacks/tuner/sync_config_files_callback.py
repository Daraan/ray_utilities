"""
When using config_files with a Setup/Parser, these files are not available on the remote
worker nodes. When loading a checkpoint on the remote those will be missing.
This callback assures that these files are synced to the remote nodes.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import ray
from ray.tune.callback import Callback

from ray.tune.experiment import Trial
from ray.tune.utils.file_transfer import sync_dir_between_nodes


if TYPE_CHECKING:
    from ray.tune.experiment import Trial

_logger = logging.getLogger(__name__)


class SyncConfigFilesCallback(Callback):
    """Syncs config files to remote nodes when trials are restored from checkpoints.

    This callback ensures that config files specified in the experiment setup
    are available on remote worker nodes when trials are restored from checkpoints.
    Config files are needed by the argument parser but may not be present on
    remote nodes in distributed setups.

    The callback works by:
    1. Detecting when trials are restored (`on_trial_restore`)
    2. Checking if the trial has config file sync data in its checkpoint
    3. Using the trial's storage syncer to download config files to the remote node
    4. Restoring config files to the local cache using `ConfigFileSync`

    This is particularly important when using Ray Tune with remote storage
    and distributed execution, where the trial's working directory may be
    on a different node than where the config files were originally located.
    """

    def __init__(self):
        super().__init__()
        self._synced_trials: set[str] = set()

    def on_trial_restore(self, iteration: int, trials: List["Trial"], trial: "Trial", **info) -> None:
        self.on_trial_start(iteration, trials, trial, **info)

    def on_trial_start(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        """Called after restoring a trial instance.

        This method checks if the trial needs config files synced and performs
        the sync operation if necessary.

        Args:
            iteration: Number of iterations of the tuning loop.
            trials: List of trials.
            trial: Trial that just has been restored.
            **info: Kwargs dict for forward compatibility.
        """
        # Avoid syncing the same trial multiple times
        if trial.trial_id in self._synced_trials:
            _logger.debug("Config files already synced for trial %s", trial.trial_id)
            return

        try:
            self._sync_config_files_for_trial(trial)
            self._synced_trials.add(trial.trial_id)
        except Exception:
            _logger.exception(
                "Failed to sync config files for trial %s",
                trial.trial_id,
            )

    def _sync_config_files_for_trial(self, trial: "Trial") -> None:
        """Sync config files for a specific trial.

        Args:
            trial: The trial to sync config files for.
        """
        # Check if trial has storage and syncer available
        if not trial.storage:
            _logger.info("Trial %s has no storage, skipping config file sync", trial.trial_id)
            return

        # Try to get config file sync data from the trial's last result or checkpoint
        config_file_sync_data = self._get_config_file_sync_data(trial)
        if not config_file_sync_data:
            _logger.debug("No config file sync data found for trial %s", trial.trial_id)
            return

        success = 0
        synced = []
        for file in map(Path, config_file_sync_data):
            rel_file = file
            if file.is_absolute():
                rel_file = file.relative_to(Path().absolute())
            dest = Path(trial.storage.trial_working_directory) / rel_file
            synced.append(dest)
            # remote dir is "here", local will be the worker
            good = trial.storage.syncer.sync_down(remote_dir=rel_file.as_posix(), local_dir=dest.as_posix())
            if not good:
                _logger.warning("Failed to sync config file %s for trial %s", file, trial.trial_id)
            success += good
        if success:
            trial.storage.syncer.wait_or_retry()
            _logger.info("Successfully synced %d config files for trial %s to %s", success, trial.trial_id, synced)
        return  # Get the remote path where config files should be stored

    def _get_config_file_sync_data(self, trial: "Trial") -> Optional[Dict[str, Any]]:
        """Extract config file sync data from trial's checkpoint or result.

        Args:
            trial: The trial to extract config file sync data from.

        Returns:
            Config file sync data if available, None otherwise.
        """
        # Try to get from last result first
        return trial.config.get("_config_files", None)

    def _get_remote_config_path(self, trial: "Trial") -> Optional[str]:
        """Get the remote path where config files are stored.

        Args:
            trial: The trial to get remote config path for.

        Returns:
            Remote path string if available, None otherwise.
        """
        # Use trial's remote path as base
        if hasattr(trial, "path") and trial.path:
            return str(Path(trial.path) / "config_files")

        if hasattr(trial, "remote_experiment_path") and trial.remote_experiment_path:
            return str(Path(trial.remote_experiment_path) / trial.trial_id / "config_files")

        return None

    def _sync_down_config_files(self, trial: "Trial", remote_path: str, local_path: Path) -> bool:
        """Sync config files from remote storage to local cache.

        Args:
            trial: The trial whose files to sync.
            remote_path: Remote path where config files are stored.
            local_path: Local path to sync files to.

        Returns:
            True if sync was successful, False otherwise.
        """
        if trial.storage is None:
            return False
        try:
            # Try different sync methods based on what's available on the syncer
            syncer = trial.storage.syncer

            # Method 1: Try sync_down if it exists
            if hasattr(syncer, "sync_down"):
                result = syncer.sync_down(remote_path, str(local_path))
                if result:
                    syncer.wait()
                return bool(result)

            # Method 2: Try sync_up in reverse (some syncers may support bidirectional)
            if hasattr(syncer, "sync_up"):
                # This is a bit hacky, but some syncers may support reverse sync
                result = syncer.sync_up(remote_path, str(local_path))
                if result:
                    syncer.wait()
                return bool(result)

            # Method 3: Use ray.tune.utils.file_transfer if available
            _logger.warning(
                "Syncer for trial %s does not support sync_down, attempting alternative sync method", trial.trial_id
            )
            return self._fallback_sync_method(trial, remote_path, local_path)

        except Exception:
            _logger.exception(
                "Error syncing config files for trial %s",
                trial.trial_id,
            )
            return False

    def _fallback_sync_method(self, trial: "Trial", remote_path: str, local_path: Path) -> bool:
        """Fallback method for syncing when standard syncer methods are unavailable.

        Args:
            trial: The trial whose files to sync.
            remote_path: Remote path where config files are stored.
            local_path: Local path to sync files to.

        Returns:
            True if sync was successful, False otherwise.
        """
        try:
            # Get node information if available
            if hasattr(trial, "node_ip") and trial.node_ip:
                # Try to sync between nodes
                sync_dir_between_nodes(
                    source_ip=trial.node_ip,
                    source_path=remote_path,
                    target_ip="127.0.0.1",  # Current node
                    target_path=str(local_path),
                )
                return True

        except (ImportError, OSError, FileNotFoundError) as e:
            _logger.debug("Fallback sync method failed: %s", e)

        # If all else fails, log the issue but don't crash
        _logger.warning(
            "Unable to sync config files for trial %s - config files may not be available on remote worker",
            trial.trial_id,
        )
        return False

    def _get_remote_cache_dir(self) -> Path:
        """Get the directory for caching config files on remote workers."""
        cache_dir = Path(tempfile.gettempdir()) / "ray_utilities_config_cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file for change detection."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _restore_config_files_from_ray(self, sync_data: Dict[str, Any]) -> List[Path]:
        """Download config files from Ray's object store to local cache."""
        if not sync_data:
            return []

        cache_dir = self._get_remote_cache_dir()
        restored_paths = []

        metadata = sync_data.get("metadata", {})
        content_refs = sync_data.get("content_refs", {})

        for original_path, file_meta in metadata.items():
            # Create cached file path based on hash to avoid conflicts
            file_hash = file_meta["hash"]
            cached_file = cache_dir / f"config_{file_hash[:12]}.txt"

            # Check if file already exists and is up to date
            if cached_file.exists():
                existing_hash = self._compute_file_hash(cached_file)
                if existing_hash == file_hash:
                    _logger.debug("Using cached config file: %s", cached_file)
                    restored_paths.append(cached_file)
                    continue

            # Download and cache the file
            if original_path in content_refs:
                try:
                    content = ray.get(content_refs[original_path])
                    # Ensure content is a string
                    if isinstance(content, (list, bytes)):
                        content = str(content)
                    with open(cached_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    _logger.info("Restored config file to cache: %s -> %s", original_path, cached_file)
                    restored_paths.append(cached_file)
                except (OSError, IOError, KeyError) as e:
                    _logger.error("Failed to restore config file %s: %s", original_path, e)

        return restored_paths

    def get_state(self) -> Optional[Dict]:
        """Get the state of the callback for checkpointing.

        Returns:
            Dictionary containing the set of synced trial IDs.
        """
        return {"synced_trials": list(self._synced_trials)}

    def set_state(self, state: Dict) -> None:
        """Set the state of the callback from checkpoint data.

        Args:
            state: State dictionary containing synced trial IDs.
        """
        if "synced_trials" in state:
            self._synced_trials = set(state["synced_trials"])
