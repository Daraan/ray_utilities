"""Tests for restarting experiments with the Comet callback logger.

This module tests the experiment restarting functionality of AdvCometLoggerCallback,
including forked trials, experiment recreation, and proper handling of offline/online
modes during restarts.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ray_utilities.callbacks.tuner.adv_comet_callback import AdvCometLoggerCallback
from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import make_experiment_key
from ray_utilities.testing_utils import DisableLoggers, TestHelpers, patch_args
from ray_utilities.testing_utils import MockTrial


@pytest.mark.basic
class TestCometRestartExperiments(DisableLoggers, TestHelpers):
    """Test Comet callback experiment restart functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Create base callback for testing - disable workspace checking and use valid keys
        with (
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key") as mock_make_key,
            patch.dict("os.environ", {"COMET_API_KEY": "fake_api_key"}, clear=False),
        ):
            mock_make_key.return_value = "a" * 32  # Valid 32-char key
            self.online_callback = AdvCometLoggerCallback(online=True)
            self.offline_callback = AdvCometLoggerCallback(online=False)

    def _create_forked_trial(self, trial_id: str, fork_from: str, config: dict | None = None) -> MockTrial:
        """Create a mock forked trial with proper configuration."""
        base_config = {FORK_FROM: fork_from}
        if config:
            base_config.update(config)
        trial = MockTrial(trial_id, config=base_config)
        # Set a proper trial_id format that matches expected pattern: <1-8alphanumeric>_<0000#>
        if trial_id.isdigit():
            trial.trial_id = f"abc123de_{int(trial_id):05d}"
        else:
            # Extract number from trial_id if it contains digits
            match = re.search(r"(\d+)", trial_id)
            number = int(match.group(1)) if match else 1
            trial.trial_id = f"abc123de_{number:05d}"
        return trial

    def _create_mock_trial(self, trial_number: int, config: dict | None = None) -> MockTrial:
        """Create a mock trial with proper trial ID format."""
        trial = MockTrial(str(trial_number), config=config)
        trial.trial_id = f"abc123de_{trial_number:05d}"
        return trial

    def _create_online_experiment(self) -> Mock:
        """Create a mock online comet experiment."""
        mock_experiment = Mock()
        mock_experiment.end = Mock()
        mock_experiment.set_name = Mock()
        mock_experiment.add_tags = Mock()
        mock_experiment.log_other = Mock()
        mock_experiment.log_parameters = Mock()
        mock_experiment.log_metrics = Mock()
        mock_experiment.set_filename = Mock()
        mock_experiment.set_pip_packages = Mock()
        return mock_experiment

    def _create_offline_experiment(self) -> Mock:
        """Create a mock offline comet experiment."""
        mock_experiment = self._create_online_experiment()
        mock_experiment.offline_directory = "/tmp/comet"
        return mock_experiment

    def test_restart_experiment_for_forked_trial_basic(self):
        """Test basic experiment restart functionality for forked trials."""
        callback = self.online_callback
        parent_trial_id = "parent_trial_001"
        fork_step = 100
        forked_trial = self._create_forked_trial("forked_trial_001", f"{parent_trial_id}?_step={fork_step}")

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "trial_is_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="b" * 32),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_experiment_class.return_value = mock_new_experiment

            # Call the method under test
            result = callback._restart_experiment_for_forked_trial(forked_trial, (parent_trial_id, fork_step))

            # Verify parent experiment was ended
            mock_parent_experiment.end.assert_called_once()

            # Verify new experiment was created
            mock_experiment_class.assert_called_once()

            # Verify the new experiment is returned and stored
            self.assertEqual(result, mock_new_experiment)
            self.assertEqual(callback._trial_experiments[forked_trial], mock_new_experiment)

    def test_restart_experiment_offline_mode(self):
        """Test experiment restart in offline mode."""
        callback = self.offline_callback
        parent_trial_id = "parent_trial_001"
        fork_step = 50
        forked_trial = self._create_forked_trial("forked_trial_002", f"{parent_trial_id}?_step={fork_step}")

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("comet_ml.OfflineExperiment") as mock_offline_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "trial_is_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="c" * 32),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_offline_experiment_class.return_value = mock_new_experiment

            # Call the method under test
            result = callback._restart_experiment_for_forked_trial(forked_trial, (parent_trial_id, fork_step))

            # Verify parent experiment was ended
            mock_parent_experiment.end.assert_called_once()

            # Verify offline experiment was created (not online Experiment)
            mock_offline_experiment_class.assert_called_once()

            # Verify the result is the new experiment
            self.assertEqual(result, mock_new_experiment)

    def test_restart_experiment_with_experiment_key_override(self):
        """Test that experiment keys are properly modified for forked trials."""
        callback = self.online_callback
        callback.experiment_kwargs = {"experiment_key": "original_key", "project_name": "test_project"}

        parent_trial_id = "parent_trial_001"
        fork_step = 200
        forked_trial = self._create_forked_trial("forked_trial_003", f"{parent_trial_id}?_step={fork_step}")

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "trial_is_forked", return_value=True),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key") as mock_make_key,
            patch("ray_utilities.callbacks.tuner.adv_comet_callback._LOGGER") as mock_logger,
            patch.object(callback, "_check_workspaces", return_value=0),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_experiment_class.return_value = mock_new_experiment
            mock_make_key.return_value = "new_forked_experiment_key_with_proper_length"  # 40 chars

            # Call the method under test
            callback._restart_experiment_for_forked_trial(forked_trial, (parent_trial_id, fork_step))

            # Verify warning about overriding experiment key was logged
            mock_logger.warning.assert_called_once()
            self.assertIn("Need to modify experiment key", mock_logger.warning.call_args[0][0])

            # Verify make_experiment_key was called with correct parameters
            # It's called twice: once for fork data and once in _start_experiment
            self.assertEqual(mock_make_key.call_count, 2)
            # First call should be with fork data
            mock_make_key.assert_any_call(forked_trial, (parent_trial_id, fork_step))
            # Second call should be with just the trial (in _start_experiment as setdefault)
            mock_make_key.assert_any_call(forked_trial)

            # Verify new experiment was created with modified key
            call_kwargs = mock_experiment_class.call_args[1]
            self.assertEqual(call_kwargs["experiment_key"], "new_forked_experiment_key_with_proper_length")
            self.assertEqual(call_kwargs["project_name"], "test_project")  # Other kwargs preserved

    def test_restart_experiment_forked_trial_not_started(self):
        """Test restart for forked trial that hasn't been started yet (e.g., loaded from checkpoint)."""
        callback = self.online_callback
        parent_trial_id = "parent_trial_001"
        fork_step = 75
        forked_trial = self._create_forked_trial("forked_trial_004", f"{parent_trial_id}?_step={fork_step}")

        # Forked trial is not in _trial_experiments (not started yet)

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "log_trial_end", side_effect=KeyError("trial not found")),
            patch.object(callback, "get_forked_trial_info") as mock_get_info,
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="e" * 40),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_experiment_class.return_value = mock_new_experiment

            # Mock forked trial info where parent is not present
            mock_info = Mock()
            mock_info.parent_is_present = False
            mock_get_info.return_value = (mock_info,)  # Return tuple as expected

            # Call the method under test
            result = callback._restart_experiment_for_forked_trial(forked_trial, (parent_trial_id, fork_step))

            # Verify KeyError was caught and handled
            mock_get_info.assert_called_once_with(forked_trial)

            # Verify new experiment was still created
            mock_experiment_class.assert_called_once()
            self.assertEqual(result, mock_new_experiment)

    def test_log_trial_start_with_forked_trial(self):
        """Test log_trial_start with a forked trial triggers experiment restart."""
        callback = self.online_callback
        callback.tags = ["test_tag"]

        parent_trial_id = "parent_trial_001"
        fork_step = 150
        forked_trial = self._create_forked_trial("forked_trial_005", f"{parent_trial_id}?_step={fork_step}")

        with (
            patch.object(callback, "_restart_experiment_for_forked_trial") as mock_restart,
            patch.object(callback, "get_forked_trial_info") as mock_get_info,
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.parse_fork_from") as mock_parse,
        ):
            mock_experiment = self._create_online_experiment()
            mock_restart.return_value = mock_experiment
            mock_parse.return_value = (parent_trial_id, fork_step)
            mock_get_info.return_value = True  # Trial is forked

            # Mock the side effect of _restart_experiment_for_forked_trial
            def mock_restart_side_effect(trial, fork_data):
                callback._trial_experiments[trial] = mock_experiment
                return mock_experiment

            mock_restart.side_effect = mock_restart_side_effect

            # Call log_trial_start
            callback.log_trial_start(forked_trial)

            # Verify fork_from was parsed
            assert forked_trial.config is not None
            mock_parse.assert_called_once_with(forked_trial.config[FORK_FROM])

            # Verify restart was called
            mock_restart.assert_called_once_with(forked_trial, (parent_trial_id, fork_step))

            # Verify experiment was configured
            mock_experiment.set_name.assert_called_once_with(str(forked_trial))
            mock_experiment.add_tags.assert_called_once_with(["test_tag"])
            # Verify both log_other calls: "Created from" and command line args
            self.assertEqual(mock_experiment.log_other.call_count, 2)
            mock_experiment.log_other.assert_any_call("Created from", "Ray")
            # The second call should be for CLI args if they exist

    def test_log_trial_start_non_forked_trial(self):
        """Test log_trial_start with non-forked trial uses normal flow."""
        callback = self.online_callback
        callback.tags = ["test_tag"]

        normal_trial = self._create_mock_trial(1, config={"param1": "value1"})

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.flatten_dict") as mock_flatten,
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="f" * 40),
        ):
            mock_experiment = self._create_online_experiment()
            mock_experiment_class.return_value = mock_experiment
            mock_flatten.return_value = {"param1": "value1"}

            # Call log_trial_start
            callback.log_trial_start(normal_trial)

            # Verify experiment was created
            mock_experiment_class.assert_called_once()

            # Verify experiment was configured
            mock_experiment.set_name.assert_called_once_with(str(normal_trial))
            mock_experiment.add_tags.assert_called_once_with(["test_tag"])

    def test_restart_experiment_invalid_fork_data(self):
        """Test restart fails gracefully with invalid fork data."""
        callback = self.online_callback
        forked_trial = self._create_forked_trial("forked_trial_006", "invalid_fork_format")

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.parse_fork_from", return_value=None),
            patch.object(callback, "trial_is_forked", return_value=True),
        ):
            # Should raise ValueError for invalid fork data
            with self.assertRaises(ValueError) as cm:
                callback._restart_experiment_for_forked_trial(forked_trial, None)

            self.assertIn("Could not parse", str(cm.exception))
            assert forked_trial.config is not None
            self.assertIn(forked_trial.config[FORK_FROM], str(cm.exception))

    def test_start_experiment_with_custom_kwargs(self):
        """Test _start_experiment respects custom experiment_kwargs."""
        callback = self.online_callback
        callback.experiment_kwargs = {
            "project_name": "test_project",
            "workspace": "test_workspace",
            "experiment_key": "custom_key",
        }

        trial = self._create_mock_trial(1)
        custom_kwargs = {
            "project_name": "override_project",
            "experiment_key": make_experiment_key(trial),
            "api_key": "test_api_key",
        }

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "_check_workspaces", return_value=0),
        ):
            mock_experiment = self._create_online_experiment()
            mock_experiment_class.return_value = mock_experiment

            # Call with custom kwargs
            result = callback._start_experiment(trial, custom_kwargs)

            # Verify experiment was created with custom kwargs
            mock_experiment_class.assert_called_once()
            call_kwargs = mock_experiment_class.call_args[1]
            self.assertEqual(call_kwargs["project_name"], "override_project")
            self.assertEqual(call_kwargs["experiment_key"], make_experiment_key(trial))
            self.assertEqual(call_kwargs["api_key"], "test_api_key")
            # workspace should not be in custom_kwargs
            self.assertNotIn("workspace", call_kwargs)

            # Verify trial is stored and experiment returned
            self.assertEqual(callback._trial_experiments[trial], mock_experiment)
            self.assertEqual(result, mock_experiment)

    def test_restart_preserves_offline_directory_structure(self):
        """Test that restart in offline mode preserves directory structure."""
        callback = self.offline_callback
        callback.experiment_kwargs = {"offline_directory": "/custom/offline/dir"}

        parent_trial_id = "parent_trial_001"
        fork_step = 300
        forked_trial = self._create_forked_trial("forked_trial_007", f"{parent_trial_id}?_step={fork_step}")

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("comet_ml.OfflineExperiment") as mock_offline_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "trial_is_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="g" * 40),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_offline_experiment_class.return_value = mock_new_experiment

            # Call the method under test
            callback._restart_experiment_for_forked_trial(forked_trial, (parent_trial_id, fork_step))

            # Verify offline experiment was created with custom directory
            call_kwargs = mock_offline_experiment_class.call_args[1]
            self.assertEqual(call_kwargs["offline_directory"], "/custom/offline/dir")

    def test_multiple_restarts_same_trial(self):
        """Test that multiple restarts of the same trial work correctly."""
        callback = self.online_callback
        parent_trial_id = "parent_trial_001"
        fork_step = 400
        forked_trial = self._create_forked_trial("forked_trial_008", f"{parent_trial_id}?_step={fork_step}")

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "trial_is_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="h" * 40),
        ):
            # First restart
            mock_experiment1 = self._create_online_experiment()
            mock_experiment_class.return_value = mock_experiment1

            callback._trial_experiments[forked_trial] = Mock()  # Previous experiment
            result1 = callback._restart_experiment_for_forked_trial(forked_trial, (parent_trial_id, fork_step))

            # Second restart
            mock_experiment2 = self._create_online_experiment()
            mock_experiment_class.return_value = mock_experiment2

            result2 = callback._restart_experiment_for_forked_trial(forked_trial, (parent_trial_id, fork_step))

            # Verify both experiments were created and the latest one is stored
            self.assertEqual(mock_experiment_class.call_count, 2)
            self.assertEqual(result1, mock_experiment1)
            self.assertEqual(result2, mock_experiment2)
            self.assertEqual(callback._trial_experiments[forked_trial], mock_experiment2)

    @patch_args("--comet", "offline")
    def test_integration_with_offline_upload_on_restart(self):
        """Test integration between restart and offline upload functionality."""
        callback = AdvCometLoggerCallback(online=False, upload_offline_experiments=True)
        parent_trial_id = "parent_trial_001"
        fork_step = 500
        forked_trial = self._create_forked_trial("forked_trial_009", f"{parent_trial_id}?_step={fork_step}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock zip file for upload
            zip_file = Path(tmpdir) / "forked_trial_009.zip".replace("_", "xx")
            zip_file.touch()

            with (
                patch("comet_ml.OfflineExperiment") as mock_offline_experiment_class,
                patch("comet_ml.config.set_global_experiment"),
                patch.object(callback, "trial_is_forked", return_value=True),
                patch("ray_utilities.callbacks.tuner.adv_comet_callback.COMET_OFFLINE_DIRECTORY", tmpdir),
                patch.object(callback, "_upload_offline_experiment_if_available") as mock_upload,
                patch.object(callback, "_check_workspaces", return_value=0),
                patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="d" * 40),
            ):
                mock_new_experiment = self._create_online_experiment()
                mock_offline_experiment_class.return_value = mock_new_experiment

                # Create initial experiment
                callback._trial_experiments[forked_trial] = Mock()

                # Restart experiment
                callback._restart_experiment_for_forked_trial(forked_trial, (parent_trial_id, fork_step))

                # Simulate trial completion
                callback.on_trial_complete(1, [forked_trial], forked_trial)

                # Verify upload was attempted
                mock_upload.assert_called_once_with(forked_trial)


class TestCometRestartEdgeCases(DisableLoggers, TestHelpers):
    """Test edge cases in Comet callback restart functionality."""

    def _create_mock_trial(self, trial_number: int, config: dict | None = None) -> MockTrial:
        """Create a mock trial with proper trial ID format."""
        trial = MockTrial(str(trial_number), config=config)
        trial.trial_id = f"abc123de_{trial_number:05d}"
        return trial

    def test_restart_concurrent_experiments(self):
        """Test restart behavior with concurrent experiments."""
        callback = AdvCometLoggerCallback(online=True)

        # Create multiple forked trials
        forked_trial_1 = self._create_mock_trial(1, config={FORK_FROM: "parent_001?_step=100"})
        forked_trial_2 = self._create_mock_trial(2, config={FORK_FROM: "parent_002?_step=200"})

        # Mock parent experiments
        mock_parent_1 = Mock()
        mock_parent_2 = Mock()
        callback._trial_experiments[forked_trial_1] = mock_parent_1
        callback._trial_experiments[forked_trial_2] = mock_parent_2

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "trial_is_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
        ):
            mock_new_exp_1 = Mock()
            mock_new_exp_2 = Mock()
            mock_experiment_class.side_effect = [mock_new_exp_1, mock_new_exp_2]

            # Restart both experiments
            result_1 = callback._restart_experiment_for_forked_trial(forked_trial_1, ("parent_001", 100))
            result_2 = callback._restart_experiment_for_forked_trial(forked_trial_2, ("parent_002", 200))

            # Verify both parent experiments were ended
            mock_parent_1.end.assert_called_once()
            mock_parent_2.end.assert_called_once()

            # Verify both new experiments were created and stored correctly
            self.assertEqual(result_1, mock_new_exp_1)
            self.assertEqual(result_2, mock_new_exp_2)
            self.assertEqual(callback._trial_experiments[forked_trial_1], mock_new_exp_1)
            self.assertEqual(callback._trial_experiments[forked_trial_2], mock_new_exp_2)
            self.assertEqual(mock_experiment_class.call_count, 2)
