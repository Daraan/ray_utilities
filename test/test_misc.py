from __future__ import annotations

import os
import random
import re
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
import ray.tune.logger
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray_utilities.callbacks.tuner.adv_comet_callback import AdvCometLoggerCallback
from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.misc import RE_GET_TRIAL_ID
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import (
    Cases,
    DisableLoggers,
    TestHelpers,
    check_args,
    iter_cases,
    mock_trainable_algorithm,
    no_parallel_envs,
    patch_args,
)
from ray_utilities.testing_utils import _MockTrial as MockTrial
from ray_utilities.training.helpers import make_divisible

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


@pytest.mark.basic
class TestMeta(TestCase):
    @Cases([1, 2, 3])
    def test_test_cases(self, cases):
        tested = []
        for i, r in enumerate(iter_cases(cases), start=1):
            self.assertEqual(r, i)
            tested.append(r)
        self.assertEqual(tested, [1, 2, 3])
        # test list
        tested = []
        for i, r in enumerate(iter_cases([1, 2, 3]), start=1):
            self.assertEqual(r, i)
            tested.append(r)
        self.assertEqual(tested, [1, 2, 3])
        # Test iterator
        tested = []
        for i, r in enumerate(iter_cases(v for v in [1, 2, 3]), start=1):
            self.assertEqual(r, i)
            tested.append(r)
        self.assertEqual(tested, [1, 2, 3])

    def test_env_config_merge(self):
        config = AlgorithmConfig()
        config.environment(env_config={"a": 1, "b": 2})
        config.evaluation(evaluation_config=AlgorithmConfig.overrides(env_config={"a": 5, "d": 5}))
        override = AlgorithmConfig.overrides(env_config={"b": 3, "c": 4})
        config.update_from_dict(override)
        self.assertDictEqual(config.env_config, {"a": 1, "b": 3, "c": 4})
        eval_config = config.get_evaluation_config_object()
        assert eval_config
        self.assertDictEqual(eval_config.env_config, {"a": 5, "b": 3, "c": 4, "d": 5})

    @no_parallel_envs
    @mock_trainable_algorithm
    def test_no_parallel_envs(self):
        self.assertEqual(DefaultArgumentParser.num_envs_per_env_runner, 1)
        self.assertEqual(
            AlgorithmSetup(init_param_space=False).trainable_class().algorithm_config.num_envs_per_env_runner, 1
        )


class TestNoLoggers(DisableLoggers):
    def test_no_loggers(self):
        # This test is just to ensure that the DisableLoggers context manager works.
        # It does not need to do anything, as the context manager will disable loggers.
        self.assertEqual(ray.tune.logger.DEFAULT_LOGGERS, ())
        setup = AlgorithmSetup()
        trainable = setup.trainable_class()
        if isinstance(trainable._result_logger, ray.tune.logger.UnifiedLogger):
            self.assertEqual(trainable._result_logger._logger_cls_list, ())
            self.assertEqual(len(trainable._result_logger._loggers), 0)
        if isinstance(trainable.algorithm._result_logger, ray.tune.logger.UnifiedLogger):
            self.assertEqual(trainable.algorithm._result_logger._logger_cls_list, ())
            self.assertEqual(len(trainable.algorithm._result_logger._loggers), 0)


@pytest.mark.basic
class TestMisc(TestCase):
    def test_re_find_id(self):
        match = RE_GET_TRIAL_ID.search("sdf_sdgsg_12:12:id=52e65_00002_sdfgf")
        assert match is not None
        self.assertEqual(match.groups(), ("52e65_00002", "52e65", "00002"))
        self.assertEqual(match.group(), "id=52e65_00002")
        self.assertEqual(match.group(1), "52e65_00002")
        self.assertEqual(match.group("trial_id"), "52e65_00002")
        self.assertEqual(
            match.groupdict(), {"trial_id": "52e65_00002", "trial_id_part1": "52e65", "trial_number": "00002"}
        )

        match = RE_GET_TRIAL_ID.search("sdf_sdgsg_12:12:id=52e65_sdfgf")
        assert match is not None
        self.assertEqual(match.groups(), ("52e65", "52e65", None))
        self.assertEqual(match.group(), "id=52e65")
        self.assertEqual(match.group(1), "52e65")
        self.assertEqual(match.group("trial_id"), "52e65")
        self.assertEqual(match.groupdict(), {"trial_id": "52e65", "trial_id_part1": "52e65", "trial_number": None})

    def test_make_divisible(self):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        a_div = make_divisible(a, b)
        self.assertEqual(a_div % b, 0)
        self.assertGreaterEqual(a_div, a)

    def test_check_valid_args_decorator(self):
        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):

            @check_args
            @patch_args("--it", 10, check_for_errors=False)
            def f():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            f()

        with self.assertRaisesRegex(ExceptionGroup, "Unexpected unrecognized args"):

            @check_args
            @patch_args("--it", 10, check_for_errors=False)
            def f2():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)
                raise ValueError("Some other error")

            f2()

        # Test exception

        @check_args(exceptions=["--it", "10"])
        @patch_args("--it", 10, check_for_errors=False)
        def h():
            AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

        h()

        # Exception order matters
        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):

            @check_args(exceptions=["10", "--it"])
            @patch_args("--it", 10, check_for_errors=False)
            def g():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            g()

        # Exception order matters
        with self.assertRaisesRegex(
            ValueError, re.escape("Unexpected unrecognized args: ['--foo', '10', '--it', '10', '--bar', '10']")
        ):

            @check_args(exceptions=["--it", "10"])
            @patch_args("--foo", "10", "--it", "10", "--bar", "10", check_for_errors=False)
            def g():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            g()

    def test_parse_args_with_check(self):
        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):

            @patch_args("--it", 10)
            def f():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            f()

        with self.assertRaisesRegex(ExceptionGroup, "Unexpected unrecognized args"):

            @patch_args("--it", 10)
            def f2():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)
                raise ValueError("Some other error")

            f2()

        # Test exception;  OK
        @patch_args("--it", 10, except_parser_errors=["--it", "10"])
        def h():
            AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

        h()

        # Exception order matters

        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):

            @patch_args("--it", 10, except_parser_errors=["10", "--it"])
            def g():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            g()

        # Exception order matters
        with self.assertRaisesRegex(
            ValueError, re.escape("Unexpected unrecognized args: ['--foo', '12', '--bar', '13']")
        ):

            @patch_args("--foo", "10", "--it", "12", "--bar", "13", except_parser_errors=["10", "--it"])
            def g():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            g()

    def test_parse_args_as_with(self):
        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):
            with patch_args("--it", 10):
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

        with self.assertRaisesRegex(ExceptionGroup, "Unexpected unrecognized args"):
            with patch_args("--it", 10):
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)
                raise ValueError("Some other error")

        # Test exception;  OK
        with patch_args("--it", 10, except_parser_errors=["--it", "10"]):
            AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

    def test_can_import_default_arguments(self):
        # This test is just to ensure that the default_arguments module can be imported
        # without errors. It does not need to do anything else.
        import default_arguments.PYTHON_ARGCOMPLETE_OK  # noqa: PLC0415


class TestCallbackUploads(DisableLoggers, TestHelpers):
    """Test callback upload behavior after trial completion."""

    def test_comet_callback_on_trial_complete_online(self):
        """Test that on_trial_complete works correctly for online Comet experiments."""
        callback = AdvCometLoggerCallback(online=True)
        trial = MockTrial("test_trial_001")

        # Mock the parent method
        with patch.object(callback.__class__.__bases__[1], "on_trial_complete") as mock_parent:
            callback.on_trial_complete(1, [trial], trial)
            # Should call parent method
            mock_parent.assert_called_once_with(1, [trial], trial)

            # Should not try to upload for online mode
            # (no upload method should be called since we're online)

    def test_comet_callback_on_trial_complete_offline(self):
        """Test that on_trial_complete triggers upload for offline Comet experiments."""
        callback = AdvCometLoggerCallback(online=False, upload_offline_experiments=True)
        trial = MockTrial("test_trial_001")

        with (
            patch.object(callback.__class__.__bases__[1], "on_trial_complete") as mock_parent,
            patch.object(callback, "_upload_offline_experiment_if_available") as mock_upload,
        ):
            callback.on_trial_complete(1, [trial], trial)

            # Should call parent method
            mock_parent.assert_called_once_with(1, [trial], trial)
            # Should call upload method for offline mode
            mock_upload.assert_called_once_with(trial)

    def test_comet_upload_offline_experiment_no_directory(self):
        """Test Comet upload behavior when offline directory doesn't exist."""
        callback = AdvCometLoggerCallback(online=False, upload_offline_experiments=True)
        trial = MockTrial("test_trial_001")

        with (
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.COMET_OFFLINE_DIRECTORY", "/non/existent/dir"),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback._LOGGER") as mock_logger,
            patch("time.sleep"),  # Speed up test
        ):
            callback._upload_offline_experiment_if_available(trial)

            mock_logger.debug.assert_called_once()
            self.assertIn("does not exist", mock_logger.debug.call_args[0][0])

    def test_comet_upload_offline_experiment_with_files(self):
        """Test Comet upload behavior when offline files exist."""
        callback = AdvCometLoggerCallback(online=False, upload_offline_experiments=True)
        trial = MockTrial("test_trial_001")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock zip file
            zip_file = Path(tmpdir) / "test_trial_001.zip".replace("_", "xx")
            zip_file.touch()

            with (
                patch("ray_utilities.callbacks.tuner.adv_comet_callback.COMET_OFFLINE_DIRECTORY", tmpdir),
                patch("ray_utilities.callbacks.tuner.adv_comet_callback.CometArchiveTracker") as mock_tracker_class,
                patch("ray_utilities.callbacks.tuner.adv_comet_callback._LOGGER") as mock_logger,
                patch("time.sleep"),  # Speed up test
                patch("comet_ml.offline"),
            ):
                mock_tracker = MagicMock()
                mock_tracker_class.return_value = mock_tracker

                callback._upload_offline_experiment_if_available(trial)

                # Should create tracker with the zip file
                mock_tracker_class.assert_called_once()
                args = mock_tracker_class.call_args[1]
                self.assertFalse(args["auto"])
                self.assertEqual(len(args["track"]), 1)
                self.assertTrue(str(args["track"][0]).endswith("test_trial_001.zip".replace("_", "xx")))

                # Should call upload_and_move
                mock_tracker.upload_and_move.assert_called_once()

                # Should log attempt
                mock_logger.info.assert_called_once()
                self.assertIn("Attempting to upload", mock_logger.info.call_args[0][0])

    def test_wandb_callback_on_trial_complete(self):
        """Test that on_trial_complete triggers sync for WandB runs."""
        callback = AdvWandbLoggerCallback(mode="offline+upload", upload_offline_experiments=True)
        trial = MockTrial("test_trial_001")

        with (
            patch.object(callback.__class__.__bases__[1], "on_trial_complete") as mock_parent,
            patch.object(callback, "_sync_offline_run_if_available") as mock_sync,
        ):
            callback.on_trial_complete(1, [trial], trial)

            # Should call parent method
            mock_parent.assert_called_once_with(1, [trial], trial)
            # Should call sync method
            mock_sync.assert_called_once_with(trial)

    def test_wandb_sync_offline_run_not_offline_mode(self):
        """Test WandB sync behavior when not in offline mode and no offline runs."""
        callback = AdvWandbLoggerCallback(mode="offline+upload", upload_offline_experiments=True)
        trial = MockTrial("test_trial_001", storage=MagicMock())

        with (
            patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=False),
            patch("ray_utilities.callbacks.tuner.adv_wandb_callback.Path") as mock_path_class,
            patch("ray_utilities.callbacks.tuner.adv_wandb_callback._logger") as mock_logger,
        ):
            # Mock Path.home().glob to return no offline runs
            mock_path_class.__truediv__ = lambda self, other: self
            # Ensure both the class and its instances return the mocked glob result
            mock_path_class.return_value = mock_path_class
            mock_path_class.glob.return_value = []

            callback._sync_offline_run_if_available(trial)

            mock_logger.error.assert_called_once()
            self.assertIn("No wandb offline experiments found", mock_logger.error.call_args[0][0])

    def test_wandb_sync_offline_run_with_files(self):
        """Test WandB sync behavior when offline runs exist."""
        callback = AdvWandbLoggerCallback(mode="offline+upload", upload_offline_experiments=True)
        trial = MockTrial("test_trial_001", storage=MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock offline run directory
            offline_run_dir = Path(tmpdir) / "wandb" / "offline-run-20240101_120000-abcd1234"
            offline_run_dir.mkdir(parents=True)

            with (
                patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False),
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback.Path") as mock_path_class,
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback.subprocess.run") as mock_subprocess,
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback._logger") as mock_logger,
            ):
                # Mock Path.home() to return our temp directory
                mock_path_class.home.return_value = offline_run_dir
                mock_path_class.__truediv__ = lambda self, other: self
                mock_path_class.return_value = mock_path_class
                mock_path_class.glob.return_value = [offline_run_dir]

                # Mock subprocess.run to return success
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_subprocess.return_value = mock_result

                callback._sync_offline_run_if_available(trial)

                # Should call wandb sync
                mock_subprocess.assert_called_once()
                args = mock_subprocess.call_args[0][0]
                self.assertEqual(args[0], "wandb")
                self.assertEqual(args[1], "sync")
                self.assertTrue(str(args[2]).endswith("offline-run-20240101_120000-abcd1234"))

                # Should log success
                mock_logger.info.assert_called()
                log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                self.assertTrue(any("Successfully synced" in msg for msg in log_calls))

    def test_wandb_sync_offline_run_subprocess_error(self):
        """Test WandB sync behavior when subprocess fails."""
        callback = AdvWandbLoggerCallback(mode="offline+upload", upload_offline_experiments=True)
        trial = MockTrial("test_trial_001", storage=MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock offline run directory
            offline_run_dir = Path(tmpdir) / "wandb" / "offline-run-20240101_120000-abcd1234"
            offline_run_dir.mkdir(parents=True)

            with (
                patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False),
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback.Path") as mock_path_class,
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback.subprocess.run") as mock_subprocess,
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback._logger") as mock_logger,
            ):
                # Mock Path.home() to return our temp directory
                mock_path_class.home.return_value = offline_run_dir
                mock_path_class.__truediv__ = lambda self, other: self
                mock_path_class.return_value = mock_path_class
                mock_path_class.glob.return_value = [offline_run_dir]

                # Mock subprocess.run to return failure
                mock_result = MagicMock()
                mock_result.returncode = 1
                mock_result.stderr = "Mock error message"
                mock_subprocess.return_value = mock_result

                callback._sync_offline_run_if_available(trial)

                # Should log failure
                mock_logger.warning.assert_called()
                self.assertIn("Failed to sync", mock_logger.warning.call_args[0][0])
