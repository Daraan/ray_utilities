"""Tests for callback upload behavior after trial completion."""
from __future__ import annotations

import os
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ray_utilities.callbacks.tuner.adv_comet_callback import AdvCometLoggerCallback
from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback
from ray_utilities.testing_utils import DisableLoggers, TestHelpers


class MockTrial:
    """Mock trial object for testing."""
    
    def __init__(self, trial_id: str = "test_trial_001"):
        self.trial_id = trial_id


class TestCallbackUploads(DisableLoggers, TestHelpers):
    """Test callback upload behavior after trial completion."""

    def test_comet_callback_on_trial_complete_online(self):
        """Test that on_trial_complete works correctly for online Comet experiments."""
        callback = AdvCometLoggerCallback(online=True)
        trial = MockTrial()
        
        # Mock the parent method
        with patch.object(callback.__class__.__bases__[1], 'on_trial_complete') as mock_parent:
            callback.on_trial_complete(1, [trial], trial)
            # Should call parent method
            mock_parent.assert_called_once_with(1, [trial], trial)
            
            # Should not try to upload for online mode
            # (no upload method should be called since we're online)

    def test_comet_callback_on_trial_complete_offline(self):
        """Test that on_trial_complete triggers upload for offline Comet experiments.""" 
        callback = AdvCometLoggerCallback(online=False)
        trial = MockTrial()
        
        with (
            patch.object(callback.__class__.__bases__[1], 'on_trial_complete') as mock_parent,
            patch.object(callback, '_upload_offline_experiment_if_available') as mock_upload
        ):
            callback.on_trial_complete(1, [trial], trial)
            
            # Should call parent method
            mock_parent.assert_called_once_with(1, [trial], trial)
            # Should call upload method for offline mode
            mock_upload.assert_called_once_with(trial)

    def test_comet_upload_offline_experiment_no_directory(self):
        """Test Comet upload behavior when offline directory doesn't exist."""
        callback = AdvCometLoggerCallback(online=False)
        trial = MockTrial()
        
        with (
            patch('ray_utilities.constants.COMET_OFFLINE_DIRECTORY', '/nonexistent/path'),
            patch('ray_utilities.callbacks.tuner.adv_comet_callback._LOGGER') as mock_logger,
            patch('time.sleep')  # Speed up test
        ):
            callback._upload_offline_experiment_if_available(trial)
            
            mock_logger.debug.assert_called_once()
            self.assertIn("does not exist", mock_logger.debug.call_args[0][0])

    def test_comet_upload_offline_experiment_with_files(self):
        """Test Comet upload behavior when offline files exist."""
        callback = AdvCometLoggerCallback(online=False)
        trial = MockTrial()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock zip file
            zip_file = Path(tmpdir) / "test_experiment.zip" 
            zip_file.touch()
            
            with (
                patch('ray_utilities.constants.COMET_OFFLINE_DIRECTORY', tmpdir),
                patch('ray_utilities.comet.CometArchiveTracker') as mock_tracker_class,
                patch('ray_utilities.callbacks.tuner.adv_comet_callback._LOGGER') as mock_logger,
                patch('time.sleep')  # Speed up test
            ):
                mock_tracker = MagicMock()
                mock_tracker_class.return_value = mock_tracker
                
                callback._upload_offline_experiment_if_available(trial)
                
                # Should create tracker with the zip file
                mock_tracker_class.assert_called_once()
                args = mock_tracker_class.call_args[1]
                self.assertFalse(args['auto'])
                self.assertEqual(len(args['track']), 1)
                self.assertTrue(str(args['track'][0]).endswith('test_experiment.zip'))
                
                # Should call upload_and_move
                mock_tracker.upload_and_move.assert_called_once()
                
                # Should log attempt
                mock_logger.info.assert_called_once()
                self.assertIn("Attempting to upload", mock_logger.info.call_args[0][0])

    def test_wandb_callback_on_trial_complete(self):
        """Test that on_trial_complete triggers sync for WandB runs."""
        callback = AdvWandbLoggerCallback()
        trial = MockTrial()
        
        with (
            patch.object(callback.__class__.__bases__[1], 'on_trial_complete') as mock_parent,
            patch.object(callback, '_sync_offline_run_if_available') as mock_sync
        ):
            callback.on_trial_complete(1, [trial], trial)
            
            # Should call parent method 
            mock_parent.assert_called_once_with(1, [trial], trial)
            # Should call sync method
            mock_sync.assert_called_once_with(trial)

    def test_wandb_sync_offline_run_not_offline_mode(self):
        """Test WandB sync behavior when not in offline mode and no offline runs."""
        callback = AdvWandbLoggerCallback()
        trial = MockTrial()
        
        with (
            patch.dict(os.environ, {'WANDB_MODE': 'online'}, clear=False),
            patch('ray_utilities.callbacks.tuner.adv_wandb_callback.Path') as mock_path_class,
            patch('ray_utilities.callbacks.tuner.adv_wandb_callback._logger') as mock_logger
        ):
            # Mock Path.home().glob to return no offline runs
            mock_home = MagicMock()
            mock_home.glob.return_value = []
            mock_path_class.home.return_value = mock_home
            
            callback._sync_offline_run_if_available(trial)
            
            mock_logger.debug.assert_called_once()
            self.assertIn("No offline runs detected", mock_logger.debug.call_args[0][0])

    def test_wandb_sync_offline_run_with_files(self):
        """Test WandB sync behavior when offline runs exist."""
        callback = AdvWandbLoggerCallback()
        trial = MockTrial()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock offline run directory
            offline_run_dir = Path(tmpdir) / "wandb" / "offline-run-20240101_120000-abcd1234"
            offline_run_dir.mkdir(parents=True)
            
            with (
                patch.dict(os.environ, {'WANDB_MODE': 'offline'}, clear=False),
                patch('ray_utilities.callbacks.tuner.adv_wandb_callback.Path') as mock_path_class,
                patch('ray_utilities.callbacks.tuner.adv_wandb_callback.subprocess.run') as mock_subprocess,
                patch('ray_utilities.callbacks.tuner.adv_wandb_callback._logger') as mock_logger
            ):
                # Mock Path.home() to return our temp directory
                mock_home = MagicMock()
                mock_home.__truediv__ = lambda self, other: Path(tmpdir) / other
                mock_path_class.home.return_value = mock_home
                
                # Mock subprocess.run to return success
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_subprocess.return_value = mock_result
                
                callback._sync_offline_run_if_available(trial)
                
                # Should call wandb sync
                mock_subprocess.assert_called_once()
                args = mock_subprocess.call_args[0][0]
                self.assertEqual(args[0], 'wandb')
                self.assertEqual(args[1], 'sync')
                self.assertTrue(str(args[2]).endswith('offline-run-20240101_120000-abcd1234'))
                
                # Should log success
                mock_logger.info.assert_called()
                log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                self.assertTrue(any("Successfully synced" in msg for msg in log_calls))

    def test_wandb_sync_offline_run_subprocess_error(self):
        """Test WandB sync behavior when subprocess fails."""
        callback = AdvWandbLoggerCallback()
        trial = MockTrial()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock offline run directory
            offline_run_dir = Path(tmpdir) / "wandb" / "offline-run-20240101_120000-abcd1234"
            offline_run_dir.mkdir(parents=True)
            
            with (
                patch.dict(os.environ, {'WANDB_MODE': 'offline'}, clear=False),
                patch('ray_utilities.callbacks.tuner.adv_wandb_callback.Path') as mock_path_class,
                patch('ray_utilities.callbacks.tuner.adv_wandb_callback.subprocess.run') as mock_subprocess,
                patch('ray_utilities.callbacks.tuner.adv_wandb_callback._logger') as mock_logger
            ):
                # Mock Path.home() to return our temp directory
                mock_home = MagicMock()
                mock_home.__truediv__ = lambda self, other: Path(tmpdir) / other
                mock_path_class.home.return_value = mock_home
                
                # Mock subprocess.run to return failure
                mock_result = MagicMock()
                mock_result.returncode = 1
                mock_result.stderr = "Mock error message"
                mock_subprocess.return_value = mock_result
                
                callback._sync_offline_run_if_available(trial)
                
                # Should log failure
                mock_logger.warning.assert_called()
                self.assertIn("Failed to sync", mock_logger.warning.call_args[0][0])

    @pytest.mark.basic
    def test_callbacks_have_upload_methods(self):
        """Test that both callbacks have the expected upload methods."""
        comet_callback = AdvCometLoggerCallback()
        wandb_callback = AdvWandbLoggerCallback()
        
        # Check that methods exist
        self.assertTrue(hasattr(comet_callback, '_upload_offline_experiment_if_available'))
        self.assertTrue(hasattr(wandb_callback, '_sync_offline_run_if_available'))
        self.assertTrue(callable(comet_callback._upload_offline_experiment_if_available))
        self.assertTrue(callable(wandb_callback._sync_offline_run_if_available))