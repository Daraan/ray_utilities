"""Tests for Comet and WandB callback loggers, including experiment restarting and upload functionality.

This module contains tests for both AdvCometLoggerCallback and AdvWandbLoggerCallback,
covering experiment restarting, forked trials, experiment recreation, upload handling,
and proper handling of offline/online modes for both Comet and WandB loggers.
"""

from __future__ import annotations

import io
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from collections import defaultdict
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, cast
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import pytest
from ray.tune.experiment import Trial

from ray_utilities.callbacks.comet import COMET_FAILED_UPLOAD_FILE, CometArchiveTracker
from ray_utilities.callbacks.tuner.adv_comet_callback import AdvCometLoggerCallback
from ray_utilities.callbacks.tuner.adv_wandb_callback import AdvWandbLoggerCallback
from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin
from ray_utilities.callbacks.upload_helper import UploadHelperMixin
from ray_utilities.callbacks.wandb import WandbUploaderMixin
from ray_utilities.constants import FORK_DATA_KEYS, FORK_FROM, get_run_id
from ray_utilities.misc import ExperimentKey, make_experiment_key, make_fork_from_csv_header
from ray_utilities.nice_logger import ImportantLogger
from ray_utilities.runfiles.run_tune import run_tune
from ray_utilities.setup.experiment_base import ExperimentSetupBase
from ray_utilities.setup.ppo_mlp_setup import MLPSetup
from ray_utilities.testing_utils import DisableLoggers, MockPopenClass, MockTrial, TestHelpers, patch_args
from ray_utilities.typing import ForkFromData, Forktime


class DummyWandbUploader(WandbUploaderMixin):
    def __init__(self):
        pass

    @classmethod
    def _failure_aware_wait(cls, process, timeout=1, trial_id=None, *, terminate_on_timeout=False, report_upload=True):  # pyright: ignore[reportIncompatibleMethodOverride]
        return 1  # Simulate failure

    def upload_paths(self, wandb_paths, trial_runs=None, *, wait=True, parallel_uploads=5) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        # Simulate a failed upload
        class DummyProcess:
            def __init__(self, wandb_path):
                self.args = ["wandb", "sync", str(wandb_path)]
                self.stdout = None

            def poll(self):
                return 1

        DummyProcess(wandb_paths[0])  # Simulate instantiation, but do not use
        # Write failed file
        grand_path = Path(wandb_paths[0]).parent.parent
        failed_file = grand_path / f"failed_wandb_uploads-{get_run_id()}.txt"
        with open(failed_file, "w") as f:
            f.write("trial_1 : wandb sync /fake/path\n")


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
        callback = AdvCometLoggerCallback(online=False, upload_intermediate=True)
        trial = MockTrial("test_trial_001")

        with (
            patch.object(callback, "_upload_offline_experiment_if_available") as mock_upload,
            patch.object(callback, "_start_experiment") as mock_start,
        ):

            def start_experiment_side_effect(trial):
                callback._trial_experiments[trial] = MagicMock()
                return callback._trial_experiments[trial]

            mock_start.side_effect = start_experiment_side_effect
            callback.on_trial_start(1, [trial], trial)
            callback.on_trial_complete(1, [trial], trial)

            # Should call upload method for offline mode
            mock_upload.assert_called_once_with(trial, upload_command=None, blocking=False)

    @pytest.mark.flaky(max_runs=2, min_passes=1)
    def test_comet_upload_offline_experiment_no_directory(self):
        """Test Comet upload behavior when offline directory doesn't exist."""
        callback = AdvCometLoggerCallback(online=False, upload_intermediate=True)
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
        callback = AdvCometLoggerCallback(online=False, upload_intermediate=True)
        trial = MockTrial("test_trial_001")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock zip file
            filename = "test_trial_001.zip".replace("_", ExperimentKey.REPLACE_UNDERSCORE)
            zip_file = Path(tmpdir) / filename
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

                mock_logger.reset_mock()  # Side-effect in __del__ might log data; potential race condition
                callback._upload_offline_experiment_if_available(trial)

                # Should create tracker with the zip file
                mock_tracker_class.assert_called_once()
                args = mock_tracker_class.call_args[1]
                self.assertFalse(args["auto"])
                self.assertEqual(len(args["track"]), 1)
                self.assertTrue(
                    str(args["track"][0]).endswith("test_trial_001.zip".replace("_", ExperimentKey.REPLACE_UNDERSCORE))
                )

                # Should call upload_and_move
                mock_tracker.upload_and_move.assert_called_once()

                # Should log attempt
                mock_logger.info.assert_called_once()
                self.assertIn("Attempting to upload", mock_logger.info.call_args[0][0])

    def test_wandb_callback_on_trial_complete(self):
        """Test that on_trial_complete triggers sync for WandB runs."""
        callback = AdvWandbLoggerCallback(mode="offline", upload_intermediate=True)
        trial = MockTrial("test_trial_001", status="PAUSED")

        with (
            patch("ray.wait") as ray_wait,
            patch("ray.remote"),
            patch("ray.get"),
            patch("ray.kill"),
            patch("ray.get_runtime_context"),
            patch.object(callback, "_sync_offline_run_if_available") as mock_subprocess_run,
            patch.object(callback, "_wait_for_trial_actor"),
        ):
            ray_wait.side_effect = lambda futures, *args, **kwargs: (futures, [])
            callback.on_trial_start(1, [trial], trial)
            callback.on_trial_complete(1, [trial], trial)

            # mock_thread.assert_called()
            # Should call sync method
            mock_subprocess_run.assert_called_once_with(trial)

    def test_wandb_sync_offline_run_not_offline_mode(self):
        """Test WandB sync behavior when not in offline mode and no offline runs."""
        callback = AdvWandbLoggerCallback(mode="offline+upload", upload_intermediate=True)
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
        callback = AdvWandbLoggerCallback(mode="offline+upload", upload_intermediate=True)
        trial = MockTrial("test_trial_001", storage=MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock offline run directory
            offline_run_dir = Path(tmpdir) / "wandb" / "offline-run-20240101_120000-abcd1234"
            offline_run_dir.mkdir(parents=True)

            with (
                patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False),
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback.Path") as mock_path_class,
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback.subprocess.run") as mock_subprocess,
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback._logger") as _mock_logger,
                patch("ray_utilities.callbacks.upload_helper.logger") as _mock_logger_upload,
                patch.object(ImportantLogger, "important_info") as important_logger,
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
                important_logger.assert_called()
                log_calls = [call[0][1] for call in important_logger.call_args_list]
                self.assertTrue(any("Successfully synced" in msg for msg in log_calls))

    def test_wandb_sync_offline_run_subprocess_error(self):
        """Test WandB sync behavior when subprocess fails."""
        callback = AdvWandbLoggerCallback(mode="offline+upload", upload_intermediate=True)
        trial = MockTrial("test_trial_001", storage=MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock offline run directory
            offline_run_dir = Path(tmpdir) / "wandb" / "offline-run-20240101_120000-abcd1234"
            offline_run_dir.mkdir(parents=True)

            with (
                patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False),
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback.Path") as mock_path_class,
                patch("ray_utilities.callbacks.tuner.adv_wandb_callback.subprocess.run") as mock_subprocess,
                patch("ray_utilities.callbacks.upload_helper.logger") as mock_logger,
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
                mock_logger.error.assert_called()
                self.assertIn("Failed to sync offline run", mock_logger.error.call_args[0][0])


class TestWandbFailedUpload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wandb_dir = Path(self.temp_dir) / "wandb"
        self.wandb_dir.mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def test_failed_upload_file_created(self):
        uploader = DummyWandbUploader()
        uploader.upload_paths([self.wandb_dir])
        grand_path = self.wandb_dir.parent.parent
        failed_file = grand_path / f"failed_wandb_uploads-{get_run_id()}.txt"
        self.assertTrue(failed_file.exists())
        with open(failed_file) as f:
            content = f.read()
        self.assertIn("wandb sync", content)


class TestCometFailedUpload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.comet_dir = Path(self.temp_dir)
        self.archive = self.comet_dir / "exp1.zip"
        self.archive.touch()
        self.orig_offline_dir = os.environ.get("COMET_OFFLINE_DIRECTORY")
        os.environ["COMET_OFFLINE_DIRECTORY"] = str(self.comet_dir)
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def tearDown(self):
        if self.orig_offline_dir is not None:
            os.environ["COMET_OFFLINE_DIRECTORY"] = self.orig_offline_dir

    def test_failed_upload_file_created(self):
        with patch("ray_utilities.callbacks.comet.COMET_OFFLINE_DIRECTORY", str(self.comet_dir)):
            tracker = CometArchiveTracker(path=self.comet_dir)
            # Simulate failed upload
            tracker._write_failed_upload_file([str(self.archive)])
            failed_file = self.comet_dir / COMET_FAILED_UPLOAD_FILE
            self.assertTrue(failed_file.exists())
            with open(failed_file) as f:
                content = f.read()
            self.assertIn("exp1.zip", content)


class TestLoggerIntegration(TestHelpers):
    @MockPopenClass.mock
    def test_wandb_upload(self, mock_popen_class, mock_popen):
        # NOTE: This test is flaky, there are instances of no wandb folder copied
        # Does the actor die silently?
        self.no_pbar_updates()

        # use more iterations here for it to be more likely that the files are synced.
        with (
            patch_args(
                "--wandb", "offline+upload",
                "--num_jobs", 1,
                "--iterations", 5,
                "--batch_size", 32,
                "--minibatch_size", 32,
                "--num_envs_per_env_runner", 1,
                "--fcnet_hiddens", "[8]",
            ),
            #mock.patch("ray_utilities.callbacks.wandb.wandb_api", new=MagicMock()),
            #mock.patch("wandb.init"),
            mock.patch.object(ExperimentSetupBase, "_stop_monitor", return_value=None),
            mock.patch.object(ExperimentSetupBase, "verify_wandb_uploads"),
            mock.patch.object(UploadHelperMixin, "_failure_aware_wait") as _mock_aware_wait,
            mock.patch("subprocess.run") as mock_run,  # upload on_trial_complete
            mock.patch("time.sleep"),
            mock.patch("select.select", side_effect= lambda a, b, c, timeout, **kwargs: (a, b, c)),
            #mock.patch("wandb._sentry")
            #mock.patch.object(wandb.sdk.wandb_setup._WandbSetup, "ensure_service") as _,
        ):  # fmt: skip
            setup = MLPSetup()
            setup.trainable_class.use_pbar = False
            _results = run_tune(setup, raise_errors=True)
        threads = threading.enumerate()
        for t in threads:
            if "upload" in t.name.lower():
                t.join()
                self.assertFalse(t.is_alive(), "Upload thread did not finish")
        mock_run.assert_called()
        self.assertListEqual(mock_run.call_args.args[0][:2], ["wandb", "sync"])
        # self.assertFalse(mock_aware_wait.called, "Should not gather and therefore not wait.")

    @MockPopenClass.mock
    @patch.multiple("ray", remote=MagicMock(), get=MagicMock(), get_actor=MagicMock())
    @patch("time.sleep", new=MagicMock())
    @patch("select.select")
    @patch("ray.wait")
    def test_wandb_upload_dependency_ordering(self, mock_popen_class, mock_popen, ray_wait, select_mock):
        # randomize this test
        ray_wait.side_effect = lambda futures, *args, **kwargs: ([1], [])
        select_mock.side_effect = lambda a, b, c, timeout, **kwargs: (a, b, c)
        random.seed(22)
        id_counter = 1

        # Use a large prime multiplier and modulus to generate deterministic, pseudo-random IDs.
        # This ensures unique, reproducible IDs for testing without collisions.
        def generate_id():
            nonlocal id_counter
            # Deterministic but random-like: use a counter and hash it
            val = (id_counter * 982451653) % (2**32)
            id_counter += 1
            # Format as 8 hex digits
            return f"{val:08x}"

        base_trial_ids = [generate_id() for _ in range(3)]
        # Add parallel trial ids
        base_trial_ids.extend(base_trial_ids[0] + "_" + f"{i:05}" for i in range(5))
        trials = {tid: MockTrial(tid) for tid in base_trial_ids}
        # randomize order
        trial_id_to_trial: dict[str, MockTrial] = trials.copy()
        random.shuffle(base_trial_ids)
        # Fork 6 trials
        possible_parents = base_trial_ids.copy()
        generations = {0: possible_parents}
        child_graph = {tid: [] for tid in possible_parents}
        parent_lookup = dict.fromkeys(base_trial_ids, None)
        assert possible_parents
        for gen in range(1, 5):
            children = []
            possible_parents_this_generation = random.sample(possible_parents, len(possible_parents) // 2)
            parent_trials = [trial_id_to_trial[p] for p in possible_parents_this_generation]
            possible_child_trials = [t for t in trials.values() if t not in parent_trials]
            self.assertGreater(len(possible_parents_this_generation), 0)
            for i in range(min(6, len(possible_child_trials))):
                if i == 0 and gen % 2 == 0:
                    # assure at least one re-forking each second generation, use last child
                    parent_id = child_id  # pyright: ignore[reportPossiblyUnboundVariable]  # noqa: F821
                else:
                    parent_id = random.choice(possible_parents_this_generation)
                fork_data: ForkFromData = {
                    "parent_trial_id": trial_id_to_trial[parent_id].trial_id,
                    "parent_training_iteration": gen * 10,
                    "parent_time": Forktime("current_step", gen * 100),
                }
                forked_trial = possible_child_trials.pop()
                child_id: str = make_experiment_key(forked_trial, fork_data)
                self.assertTrue(32 <= len(child_id) <= 50)
                trial_id_to_trial[child_id] = forked_trial
                children.append(child_id)
                child_graph[parent_id].append(child_id)
                assert child_id not in child_graph
                assert child_id not in parent_lookup
                parent_lookup[child_id] = parent_id
                child_graph[child_id] = []
            generations[gen] = children
            possible_parents.extend(children)
        # Create dummy offline paths for each trial
        with tempfile.TemporaryDirectory() as tmpdir:
            # Result paths:
            trial_paths = {tid: Path(tmpdir) / tid for tid in trials}

            class MockResults(list):
                experiment_path = tmpdir

            mock_results = MockResults(SimpleNamespace(path=p.as_posix()) for p in trial_paths.values())
            # create wandb_folders
            track_file_contents = dict.fromkeys(trials, "")
            # create child dirs and tracking files
            trial_paths: dict[str, Path]
            for trial, children in child_graph.items():
                base_trial = trial_id_to_trial[trial]
                base_path = trial_paths[base_trial.trial_id]
                for child_id in children:
                    track_file_contents[base_trial.trial_id] += f"{child_id}, {trial}, 100, _step\n"
                    child_path = base_path / "wandb" / f"offline-run-20250101_123030-{child_id}"
                    child_path.mkdir(parents=True, exist_ok=True)
                    trial_paths[child_id] = child_path
            for base_tid in base_trial_ids:
                base_path = trial_paths[base_tid]
                base_run_dir = base_path / "wandb" / f"offline-run-20250101_123030-{base_tid}"
                base_run_dir.mkdir(parents=True, exist_ok=True)
                trial_paths[base_tid] = base_run_dir
                self.assertEqual(str(base_path.parent), tmpdir)
                if len((track := track_file_contents[base_tid]).split("\n")) > 0:
                    # only write if base was forked once
                    csv_file = base_path.parent / "wandb_fork_from.csv"
                    if not csv_file.exists():
                        with open(csv_file, "w") as f:
                            f.write(make_fork_from_csv_header())
                            # f.write(f"f{FORK_DATA_KEYS[0]}, {FORK_DATA_KEYS[1]}, parent_step, step_metric\n")
                    with open(csv_file, "a") as f:
                        f.write(track)

            # possible upload order, traverse child graph
            uploader = WandbUploaderMixin()
            mock_fork_called = False

            self.maxDiff = None

            def mock_fork_relationships(wandb_paths):
                result = WandbUploaderMixin._parse_wandb_fork_relationships(wandb_paths)
                self.assertDictEqual(
                    {child: parent for child, parent in parent_lookup.items() if parent is not None},
                    {child: parent_data[0] for child, parent_data in result.items()},
                )
                nonlocal mock_fork_called
                mock_fork_called = True
                return result

            mock_graph_build_called = False

            def mock_graph_build(trial_runs: list[tuple[str, Path]], fork_relationships):
                upload_groups = WandbUploaderMixin._build_upload_dependency_graph(
                    uploader, trial_runs, fork_relationships
                )
                self.assertSetEqual(set(trial_runs), set(trial_paths.items()))
                uploaded_trials = set()
                for group in upload_groups:
                    for trial_id, _ in group:
                        parent_id = parent_lookup[trial_id]
                        if parent_id is not None:
                            self.assertIn(
                                parent_id, uploaded_trials, f"Parent {parent_id} of {trial_id} not uploaded first"
                            )
                        uploaded_trials.add(trial_id)
                nonlocal mock_graph_build_called
                mock_graph_build_called = True
                return upload_groups

            uploader._parse_wandb_fork_relationships = mock_fork_relationships
            uploader._build_upload_dependency_graph = mock_graph_build
            mock_popen.stderr = None
            mock_popen.stdout = io.StringIO("MOCK: wandb: Syncing files...")
            mock_popen.returncode = 0
            mock_popen.args = ["wandb", "sync", mock_results[0].path]
            uploader.project = "testing"
            uploader.wandb_upload_results(mock_results)  # pyright: ignore[reportArgumentType]
            self.assertTrue(mock_fork_called, "wandb fork relationships mock was not called")
            self.assertTrue(mock_graph_build_called, "wandb graph build mock was not called")
            self.assertEqual(mock_popen_class.call_count, len(trial_id_to_trial))

    def test_trial_id_parsing(self):
        uploader = WandbUploaderMixin()
        tempdir = Path("tmp")
        for dirname, expected in (
            (tid1 := Trial.generate_id(), tid1),
            (tid2 := f"{Trial.generate_id()}_00000", tid2),
            # (f"offline-run-{tid1}", tid1),  # not supported without a timestamp
            # (f"offline-run-{tid2}", tid2),
            (f"offline-run-20231225_143022-{tid1}", tid1),
            (f"offline-run-20231225_143022-{tid2}", tid2),
            ("offline-run-20231225_143022-trial_789-10", "trial_789-10"),
            ("invalid-format", "invalid-format"),
            ("PPO-experiment-456-forked", "PPO-experiment-456-forked"),
        ):
            run_dir = tempdir / dirname
            trial_id = uploader._extract_trial_id_from_wandb_run(run_dir)
            self.assertEqual(trial_id, expected, f"Failed to parse {dirname}, expected {expected}")

    def _create_fork_info_file(self, wandb_dir: Path, fork_data: list[tuple[str, str, Optional[int]]]):
        """Create a wandb_fork_from.txt file with fork relationship data."""
        fork_file = wandb_dir / "wandb_fork_from.csv"
        exists = fork_file.exists()
        with fork_file.open("a" if exists else "w") as f:
            if not exists:
                # Could upgrade to make_fork_from_csv_header()
                f.write(f"{FORK_DATA_KEYS[0]}, {FORK_DATA_KEYS[1]}, parent_step, step_metric\n")
            for trial_id, parent_id, parent_step in fork_data:
                step_str = str(parent_step) if parent_step is not None else ""
                f.write(f"{trial_id}, {parent_id}, {step_str}, _step\n")

    def test_parse_wandb_fork_relationships_variants(self):
        cases = [
            {
                "desc": "simple fork relationships",
                "dirs": ["wandb"],
                "fork_data": [
                    ("child_1", "parent_1", 100),
                    ("child_2", "parent_1", 150),
                    ("parent_1", "root", 50),
                ],
                "expected": {
                    "child_1": ("parent_1", 100),
                    "child_2": ("parent_1", 150),
                    "parent_1": ("root", 50),
                },
            },
            {
                "desc": "fork relationships without step numbers",
                "dirs": ["wandb"],
                "fork_data": [
                    ("child_1", "parent_1", None),
                ],
                "expected": {
                    "child_1": ("parent_1", None),
                },
            },
            {
                "desc": "no fork info file exists",
                "dirs": ["wandb"],
                "fork_data": [],
                "expected": {},
            },
            {
                "desc": "multiple wandb directories",
                "dirs": ["wandb1", "wandb2"],
                "fork_data": [
                    ("child_1", "parent_1", 100),
                    ("child_2", "parent_2", 200),
                ],
                "expected": {
                    "child_1": ("parent_1", 100),
                    "child_2": ("parent_2", 200),
                },
                "split": True,
            },
        ]
        uploader = WandbUploaderMixin()
        for case in cases:
            with self.subTest(case=case["desc"]), tempfile.TemporaryDirectory() as tmpdir:
                dirs = [Path(tmpdir) / d / "wandb" for d in case["dirs"]]
                if case["fork_data"]:
                    self._create_fork_info_file(Path(tmpdir), case["fork_data"])
                relationships = uploader._parse_wandb_fork_relationships(dirs)
                self.assertEqual(relationships, case["expected"])

    def test_build_upload_dependency_graph_complex_tree(self):
        """Test building dependency graph for complex fork tree."""
        trial_runs = [
            ("root", Path("root_run")),
            ("parent_1", Path("parent_1_run")),
            ("parent_2", Path("parent_2_run")),
            ("child_1_1", Path("child_1_1_run")),
            ("child_1_2", Path("child_1_2_run")),
            ("child_2_1", Path("child_2_1_run")),
            ("independent", Path("independent_run")),
            ("grandchild", Path("grandchild_run")),
        ]

        fork_relationships = {
            "parent_1": ("root", 100),
            "parent_2": ("root", 150),
            "child_1_1": ("parent_1", 200),
            "child_1_2": ("parent_1", 250),
            "child_2_1": ("parent_2", 300),
            "independent": ("missing_parent", 1000),  # Should be treated as independent
            "grandchild": ("child_1_1", 400),
        }

        uploader = WandbUploaderMixin()
        groups = uploader._build_upload_dependency_graph(trial_runs, fork_relationships)

        # Should have 4 levels
        self.assertEqual(len(groups), 4)

        # Check level structure
        group_trial_ids = [[trial_id for trial_id, _ in group] for group in groups]

        # Level 0: root
        self.assertSetEqual(set(group_trial_ids[0]), {"root", "independent"})

        # Level 1: parent_1, parent_2 (parallel)
        self.assertCountEqual(group_trial_ids[1], ["parent_1", "parent_2"])

        # Level 2: child_1_1, child_1_2, child_2_1 (parallel)
        self.assertCountEqual(group_trial_ids[2], ["child_1_1", "child_1_2", "child_2_1"])

        # Level 3: grandchild
        self.assertEqual(group_trial_ids[3], ["grandchild"])

    def test_gather_uploads_mechanism(self):
        """Test the gather_uploads mechanism in AdvWandbLoggerCallback."""

        # Create callback with offline+upload mode
        callback = AdvWandbLoggerCallback(
            project="test_project",
            upload_intermediate=True,
            mode="offline",
        )

        # Create mock trials
        trial1 = MagicMock(spec=Trial)
        trial1.trial_id = "trial_1"
        trial1.status = "RUNNING"
        trial1.local_path = None  # Will be set in test

        trial2 = MagicMock(spec=Trial)
        trial2.trial_id = "trial_2"
        trial2.status = "RUNNING"
        trial2.local_path = None

        trial3 = MagicMock(spec=Trial)
        trial3.trial_id = "trial_3"
        trial3.status = "RUNNING"
        trial3.local_path = None

        # Initialize callback state
        callback._trials = [trial1, trial2, trial3]  # pyright: ignore[reportAttributeAccessIssue]
        callback._active_trials_count = 3

        # Mock parent log_trial_end to avoid KeyError
        with patch.object(callback.__class__.__bases__[-1], "log_trial_end"):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Setup trial paths
                for i, trial in enumerate([trial1, trial2, trial3], 1):
                    trial_path = Path(tmpdir) / f"trial_{i}"
                    trial_path.mkdir()
                    trial.local_path = trial_path.as_posix()

                    # Create wandb offline run directory
                    wandb_dir = trial_path / "wandb"
                    wandb_dir.mkdir()
                    offline_run_dir = wandb_dir / f"offline-run-20250101_120000-trial_{i}"
                    offline_run_dir.mkdir()

                # Create fork relationships - trial_2 is forked from trial_1
                fork_file = Path(tmpdir) / "wandb_fork_from.csv"
                with fork_file.open("w") as f:
                    f.write("trial_id, parent_id, parent_step, step_metric\n")
                    f.write("trial_2, trial_1, 100, _step\n")

                # Mock subprocess.run to avoid actual uploads
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="Success")

                    # Call _gather_and_upload_trials directly to test the mechanism
                    callback._gather_and_upload_trials(trial1)
                    callback._gather_and_upload_trials(trial2)
                    callback._gather_and_upload_trials(trial3)

                    # All trials should trigger immediate processing since we have all active trials
                    time.sleep(1)

                    # Verify that trials were gathered and processed
                    self.assertEqual(len(callback._trials_ending), 0)  # Should be cleared after processing

            # Test that timer is properly managed
            self.assertIsNone(callback._gather_timer)
        threads = threading.enumerate()
        for thread in threads:
            if "upload" in thread.name.lower():
                thread.join()

    @mock.patch("wandb.Api", new=MagicMock())
    def test_restart_logging_actor_with_resume(self):
        """Test that _restart_logging_actor properly sets resume parameter."""

        # Create callback
        callback = AdvWandbLoggerCallback(
            project="test_project",
            upload_intermediate=True,
            mode="offline",
        )

        # Create mock trial
        trial = MagicMock(spec=Trial)
        trial.trial_id = "trial_123"
        trial.status = "RUNNING"

        # Mock the necessary methods
        callback._cleanup_logging_actors = MagicMock()
        callback._start_logging_actor = MagicMock()
        callback._trial_queues = {trial: MagicMock()}
        callback._trial_logging_futures = {trial: MagicMock()}
        callback._trial_logging_actors = {trial: MagicMock()}
        callback._past_trial_ids = defaultdict(list, {cast("Trial", trial): ["id_before"]})

        # Mock log_trial_end to not actually end the trial
        with patch.object(callback, "log_trial_end"):
            # Test 1: Resume same trial (not forked)
            wandb_kwargs = {"id": "trial_123"}
            callback._restart_logging_actor(trial, **wandb_kwargs)

            # Check that resume was set
            start_call_kwargs = callback._start_logging_actor.call_args[1]
            self.assertEqual(start_call_kwargs.get("resume"), "must")
            self.assertEqual(start_call_kwargs.get("id"), "trial_123")

            # Reset mocks
            callback._start_logging_actor.reset_mock()

            # Test 2: Fork (not resume)
            wandb_kwargs = {"id": "trial_123_fork", "fork_from": "trial_123?_step=100"}
            callback._restart_logging_actor(trial, **wandb_kwargs)

            # Check that resume was NOT set for fork
            start_call_kwargs = callback._start_logging_actor.call_args[1]
            self.assertNotIn("resume", start_call_kwargs)
            self.assertEqual(start_call_kwargs.get("fork_from"), "trial_123?_step=100")

    @mock.patch.object(AdvWandbLoggerCallback, "_start_monitor", new=MagicMock())
    def test_restart_logging_actor_with_forked_trial_resume(self):
        """Test that _restart_logging_actor correctly handles resuming a trial that was previously forked.

        In the normal workflow, the trial ID is set in TrackForkedTrialsMixin.on_trial_start
        before AdvWandbLoggerCallback.log_trial_start is called. This means that new_trial_id
        == previous_trial_id in normal scenarios. This test simulates that behavior.
        """

        # Create callback
        callback = AdvWandbLoggerCallback(
            project="test_project",
            upload_intermediate=True,
            mode="offline",
        )

        # Create mock trial
        trial = MagicMock(spec=Trial)
        trial.trial_id = "trial_123"
        trial.status = "RUNNING"

        # Simulate that this trial was forked before and has a custom trial ID
        # (experiment_key different from trial.trial_id)
        # In normal flow, this would be set by on_trial_start
        forked_trial_id = "trial_123_forkof_parent_456_step_100"
        callback._trial_ids = {trial: forked_trial_id}

        # Mock the necessary methods
        callback._cleanup_logging_actors = MagicMock()
        callback._start_logging_actor = MagicMock()
        callback._trial_queues = {trial: MagicMock()}
        callback._trial_logging_futures = {trial: MagicMock()}
        callback._trial_logging_actors = {trial: MagicMock()}
        callback._past_trial_ids = defaultdict(list, {cast("Trial", trial): ["id_before_fork"]})

        # Mock log_trial_end to not actually end the trial
        with patch.object(callback, "log_trial_end"):
            # Test 1: Resume a previously forked trial without creating a new fork
            # In normal flow, the ID passed in wandb_kwargs matches the tracked ID
            # because both were set in on_trial_start
            wandb_kwargs = {"id": forked_trial_id}
            callback._restart_logging_actor(trial, **wandb_kwargs)

            # Check that resume was set because new_id == previous_id and no fork_from
            start_call_kwargs = callback._start_logging_actor.call_args[1]
            self.assertEqual(start_call_kwargs.get("resume"), "must")
            self.assertEqual(start_call_kwargs.get("id"), forked_trial_id)

            # Reset mocks
            callback._start_logging_actor.reset_mock()

            # Test 2: Fork from the previously forked trial
            # When forking, we pass fork_from which indicates this is a new fork
            # In normal flow, the trial would have been forked in on_trial_start first
            # and the ID would still match, but fork_from would be present
            wandb_kwargs = {"id": forked_trial_id, "fork_from": f"{forked_trial_id}?_step=200"}
            callback._restart_logging_actor(trial, **wandb_kwargs)

            # Check that resume was NOT set because fork_from is present
            start_call_kwargs = callback._start_logging_actor.call_args[1]
            self.assertNotIn("resume", start_call_kwargs)
            self.assertEqual(start_call_kwargs.get("fork_from"), f"{forked_trial_id}?_step=200")

    def test_track_forked_trials_get_trial_id(self):
        """Test that get_trial_id returns the correct trial ID for both forked and non-forked trials."""

        # Create mixin instance
        mixin = TrackForkedTrialsMixin()

        # Test 1: Non-forked trial - should return trial.trial_id
        trial1 = MagicMock(spec=Trial)
        trial1.trial_id = "trial_abc"

        # Simulate on_trial_start for non-forked trial
        mixin._trial_ids[trial1] = trial1.trial_id
        self.assertEqual(mixin.get_trial_id(trial1), "trial_abc")

        # Test 2: Forked trial - should return custom experiment_key
        trial2 = MagicMock(spec=Trial)
        trial2.trial_id = "trial_def"
        forked_id = "trial_def_forkof_parent_123_step_50"

        # Simulate on_trial_start for forked trial
        mixin._trial_ids[trial2] = forked_id
        self.assertEqual(mixin.get_trial_id(trial2), forked_id)

        # Test 3: Trial not tracked yet - should return trial.trial_id as fallback
        trial3 = MagicMock(spec=Trial)
        trial3.trial_id = "trial_ghi"
        self.assertEqual(mixin.get_trial_id(trial3), "trial_ghi")

    def test_track_forked_trials_cleanup_on_complete(self):
        """Test that on_trial_complete properly cleans up all tracking data."""
        from ray_utilities.callbacks.tuner.track_forked_trials import TrackForkedTrialsMixin  # noqa: PLC0415

        # Create mixin instance
        mixin = TrackForkedTrialsMixin()

        # Create mock trial
        trial = MagicMock(spec=Trial)
        trial.trial_id = "trial_123"

        # Simulate tracking data for a forked trial
        forked_id = "trial_123_forkof_parent_456_step_100"
        mixin._trial_ids[trial] = forked_id
        mixin._current_fork_ids[trial] = forked_id
        mixin._forked_trials[trial] = [{"parent_trial_id": "parent_456"}]
        mixin._currently_not_forked_trials.add(trial)
        mixin.parent_trial_lookup[trial] = "parent_456"

        # Verify data is present
        self.assertIn(trial, mixin._trial_ids)
        self.assertIn(trial, mixin._current_fork_ids)
        self.assertIn(trial, mixin._forked_trials)
        self.assertIn(trial, mixin._currently_not_forked_trials)
        self.assertIn(trial, mixin.parent_trial_lookup)

        # Call on_trial_complete
        mixin.on_trial_complete(iteration=1, trials=[trial], trial=trial)

        # Verify all data is cleaned up
        self.assertNotIn(trial, mixin._trial_ids)
        self.assertNotIn(trial, mixin._forked_trials)
        self.assertNotIn(trial, mixin._currently_not_forked_trials)
        self.assertNotIn(trial, mixin.parent_trial_lookup)
        # We still keep it around for childs that still might need it
        self.assertIn(trial, mixin._current_fork_ids)


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

    def tearDown(self):
        for thread in chain(self.online_callback._threads, self.offline_callback._threads):
            if isinstance(thread, subprocess.Popen):
                thread.terminate()
            elif isinstance(thread, threading.Thread):
                thread.join(timeout=1)
            else:
                # Catch-all for unexpected thread types
                try:
                    if hasattr(thread, "terminate"):
                        thread.terminate()
                    elif hasattr(thread, "join"):
                        thread.join(timeout=1)
                except Exception:  # noqa: BLE001
                    pass
        super().tearDown()

    def _create_forked_trial(self, trial_id: str, fork_data: ForkFromData, config: dict | None = None) -> MockTrial:
        """Create a mock forked trial with proper configuration."""
        base_config = {FORK_FROM: fork_data}
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
        parent_trial_id = "p_trial_0001"
        fork_step = 100
        fork_data: ForkFromData = {
            "parent_trial_id": parent_trial_id,
            "parent_training_iteration": fork_step,
            "parent_time": Forktime("training_iteration", fork_step),
        }
        forked_trial = self._create_forked_trial("forked_trial_001", fork_data)

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "is_trial_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="b" * 32),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_experiment_class.return_value = mock_new_experiment

            # Call the method under test
            fork_data: ForkFromData = {
                "parent_trial_id": parent_trial_id,
                "parent_training_iteration": fork_step,
                "parent_time": Forktime("training_iteration", fork_step),
            }
            result = callback._restart_experiment_for_forked_trial(forked_trial, fork_data)

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
        parent_trial_id = "p_trial_0001"
        fork_step = 50
        fork_data: ForkFromData = {
            "parent_trial_id": parent_trial_id,
            "parent_training_iteration": fork_step,
            "parent_time": Forktime("training_iteration", fork_step),
        }
        forked_trial = self._create_forked_trial("forked_trial_002", fork_data)

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("comet_ml.OfflineExperiment") as mock_offline_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "is_trial_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="c" * 32),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_offline_experiment_class.return_value = mock_new_experiment

            # Call the method under test
            fork_data: ForkFromData = {
                "parent_trial_id": parent_trial_id,
                "parent_training_iteration": fork_step,
                "parent_time": Forktime("training_iteration", fork_step),
            }
            result = callback._restart_experiment_for_forked_trial(forked_trial, fork_data)

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

        parent_trial_id = "p_trial_0001"
        fork_step = 200
        fork_data: ForkFromData = {
            "parent_trial_id": parent_trial_id,
            "parent_training_iteration": fork_step,
            "parent_time": Forktime("training_iteration", fork_step),
        }
        forked_trial = self._create_forked_trial("forked_trial_003", fork_data)

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "is_trial_forked", return_value=True),
            patch(
                "ray_utilities.callbacks.tuner.adv_comet_callback.AdvCometLoggerCallback.make_forked_trial_id"
            ) as mock_make_key,
            patch("ray_utilities.callbacks.tuner.adv_comet_callback._LOGGER") as mock_logger,
            patch.object(callback, "_check_workspaces", return_value=0),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_experiment_class.return_value = mock_new_experiment
            mock_make_key.return_value = "new_forked_experiment_key_with_proper_length"  # 40 chars

            # Call the method under test
            fork_data: ForkFromData = {
                "parent_trial_id": parent_trial_id,
                "parent_training_iteration": fork_step,
                "parent_time": Forktime("training_iteration", fork_step),
            }
            callback._restart_experiment_for_forked_trial(forked_trial, fork_data)

            # Verify warning about overriding experiment key was logged
            mock_logger.warning.assert_called_once()
            self.assertIn("Need to modify experiment key", mock_logger.warning.call_args[0][0])

            # Verify make_experiment_key was called with correct parameters
            # It's called twice: once for fork data and once in _start_experiment
            self.assertEqual(mock_make_key.call_count, 1)
            # Should be called with trial and fork_data
            mock_make_key.assert_called_with(forked_trial, fork_data)

            # Verify new experiment was created with modified key
            call_kwargs = mock_experiment_class.call_args[1]
            self.assertEqual(call_kwargs["experiment_key"], "new_forked_experiment_key_with_proper_length")
            self.assertEqual(call_kwargs["project_name"], "test_project")  # Other kwargs preserved

    def test_restart_experiment_forked_trial_not_started(self):
        """Test restart for forked trial that hasn't been started yet (e.g., loaded from checkpoint)."""
        callback = self.online_callback
        parent_trial_id = "p_trial_0001"
        fork_step = 75
        fork_data: ForkFromData = {
            "parent_trial_id": parent_trial_id,
            "parent_training_iteration": fork_step,
            "parent_time": Forktime("training_iteration", fork_step),
        }
        forked_trial = self._create_forked_trial("forked_trial_004", fork_data)

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
            class MockWithGet(Mock):
                def __getitem__(self, key):
                    return {-1: fork_data}[key]

            mock_info = MockWithGet()
            # cannot mock magic methods
            mock_get_info.return_value = mock_info  # Return tuple as expected

            # Call the method under test
            fork_data: ForkFromData = {
                "parent_trial_id": parent_trial_id,
                "parent_training_iteration": fork_step,
                "parent_time": Forktime("training_iteration", fork_step),
            }
            result = callback._restart_experiment_for_forked_trial(forked_trial, fork_data)

            # Verify KeyError was caught and handled
            # mock_get_info.assert_called_once_with(forked_trial)

            # Verify new experiment was still created
            mock_experiment_class.assert_called_once()
            self.assertEqual(result, mock_new_experiment)

    def test_log_trial_start_with_forked_trial(self):
        """Test log_trial_start with a forked trial triggers experiment restart."""
        callback = self.online_callback
        callback.tags = ["test_tag"]

        parent_trial_id = "p_trial_0001"
        fork_step = 150
        fork_data: ForkFromData = {
            "parent_trial_id": parent_trial_id,
            "parent_training_iteration": fork_step,
            "parent_time": Forktime("training_iteration", fork_step),
        }

        forked_trial = self._create_forked_trial("2", fork_data)

        with (
            patch.object(callback, "_restart_experiment_for_forked_trial") as mock_restart,
            patch.object(callback, "get_forked_trial_info") as mock_get_info,
        ):
            mock_experiment = self._create_online_experiment()
            mock_restart.return_value = mock_experiment
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

            # Verify restart was called
            mock_restart.assert_called_once_with(forked_trial, fork_data)

            # Verify experiment was configured
            mock_experiment.set_name.assert_called_once_with(
                f"{forked_trial!s}_forkof_{parent_trial_id}_training_iteration={fork_step}"
            )
            mock_experiment.add_tags.assert_called_once_with(["test_tag", "forked"])
            # Verify both log_other calls: "Created from" and command line args
            mock_experiment.log_other.assert_any_call("Created from", "Ray")
            expected_calls = 1
            if callback._cli_args:
                expected_calls += 1
            if FORK_FROM in forked_trial.config:
                expected_calls += 1
            # Will be called with sys args if present
            self.assertEqual(mock_experiment.log_other.call_count, expected_calls, mock_experiment.log_other.call_args)
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
        forked_trial = self._create_forked_trial("forked_trial_006", "invalid_fork_format")  # pyright: ignore[reportArgumentType]

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with patch.object(callback, "is_trial_forked", return_value=True):
            # Should raise ValueError for invalid fork data
            with self.assertRaises(TypeError) as cm:
                callback._restart_experiment_for_forked_trial(forked_trial, None)

            self.assertIn("string indices must be integers", str(cm.exception))
            assert forked_trial.config is not None

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

        parent_trial_id = "p_trial_0001"
        fork_step = 300
        fork_data: ForkFromData = {
            "parent_trial_id": parent_trial_id,
            "parent_training_iteration": fork_step,
            "parent_time": Forktime("training_iteration", fork_step),
        }
        forked_trial = self._create_forked_trial("forked_trial_007", fork_data)

        # Create mock parent experiment
        mock_parent_experiment = self._create_online_experiment()
        callback._trial_experiments[forked_trial] = mock_parent_experiment

        with (
            patch("comet_ml.OfflineExperiment") as mock_offline_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "is_trial_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="g" * 40),
        ):
            mock_new_experiment = self._create_online_experiment()
            mock_offline_experiment_class.return_value = mock_new_experiment

            # Call the method under test
            fork_data: ForkFromData = {
                "parent_trial_id": parent_trial_id,
                "parent_training_iteration": fork_step,
                "parent_time": Forktime("training_iteration", fork_step),
            }
            callback._restart_experiment_for_forked_trial(forked_trial, fork_data)

            # Verify offline experiment was created with custom directory
            call_kwargs = mock_offline_experiment_class.call_args[1]
            self.assertEqual(call_kwargs["offline_directory"], "/custom/offline/dir")

    def test_multiple_restarts_same_trial(self):
        """Test that multiple restarts of the same trial work correctly."""
        callback = self.online_callback
        parent_trial_id = "p_trial_0001"
        fork_step = 400
        fork_data = {
            "parent_trial_id": parent_trial_id,
            "parent_training_iteration": fork_step,
            "parent_time": Forktime("training_iteration", fork_step),
        }
        forked_trial = self._create_forked_trial("forked_trial_008", fork_data)

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "is_trial_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
            patch("ray_utilities.callbacks.tuner.adv_comet_callback.make_experiment_key", return_value="h" * 40),
        ):
            # First restart
            mock_experiment1 = self._create_online_experiment()
            mock_experiment_class.return_value = mock_experiment1

            callback._trial_experiments[forked_trial] = Mock()  # Previous experiment
            fork_data: ForkFromData = {
                "parent_trial_id": parent_trial_id,
                "parent_training_iteration": fork_step,
                "parent_time": Forktime("training_iteration", fork_step),
            }
            result1 = callback._restart_experiment_for_forked_trial(forked_trial, fork_data)

            # Second restart
            mock_experiment2 = self._create_online_experiment()
            mock_experiment_class.return_value = mock_experiment2

            result2 = callback._restart_experiment_for_forked_trial(forked_trial, fork_data)

            # Verify both experiments were created and the latest one is stored
            self.assertEqual(mock_experiment_class.call_count, 2)
            self.assertEqual(result1, mock_experiment1)
            self.assertEqual(result2, mock_experiment2)
            self.assertEqual(callback._trial_experiments[forked_trial], mock_experiment2)

    @patch_args("--comet", "offline")
    def test_integration_with_offline_upload_on_restart(self):
        """Test integration between restart and offline upload functionality."""
        callback = AdvCometLoggerCallback(online=False, upload_intermediate=True)
        parent_trial_id = "p_trial_0001"
        fork_step = 500
        fork_data = {
            "parent_trial_id": parent_trial_id,
            "parent_training_iteration": fork_step,
            "parent_time": Forktime("training_iteration", fork_step),
        }
        forked_trial = self._create_forked_trial("forked_trial_009", fork_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock zip file for upload
            zip_file = Path(tmpdir) / "forked_trial_009.zip".replace("_", "xx")
            zip_file.touch()

            with (
                patch("comet_ml.OfflineExperiment") as mock_offline_experiment_class,
                patch("comet_ml.config.set_global_experiment"),
                patch.object(callback, "is_trial_forked", return_value=True),
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
                fork_data: ForkFromData = {
                    "parent_trial_id": parent_trial_id,
                    "parent_training_iteration": fork_step,
                    "parent_time": Forktime("training_iteration", fork_step),
                }
                callback._restart_experiment_for_forked_trial(forked_trial, fork_data)

                # Simulate trial completion
                callback.on_trial_complete(1, [forked_trial], forked_trial)

                # Verify upload was attempted
                self.assertEqual(mock_upload.call_count, 2)
                # trial, no command, not blocking
                mock_upload.assert_called_with(forked_trial, upload_command=None, blocking=False)

    def test_restart_concurrent_experiments(self):
        """Test restart behavior with concurrent experiments."""
        callback = AdvCometLoggerCallback(online=True)

        # Create multiple forked trials
        fork_data_1: ForkFromData = {
            "parent_trial_id": "parent_001",
            "parent_training_iteration": 100,
            "parent_time": Forktime("training_iteration", 100),
        }
        fork_data_2: ForkFromData = {
            "parent_trial_id": "parent_002",
            "parent_training_iteration": 200,
            "parent_time": Forktime("training_iteration", 200),
        }
        forked_trial_1 = self._create_mock_trial(1, config={FORK_FROM: fork_data_1})
        forked_trial_2 = self._create_mock_trial(2, config={FORK_FROM: fork_data_2})

        # Mock parent experiments
        mock_parent_1 = Mock()
        mock_parent_2 = Mock()
        callback._trial_experiments[forked_trial_1] = mock_parent_1
        callback._trial_experiments[forked_trial_2] = mock_parent_2

        with (
            patch("comet_ml.Experiment") as mock_experiment_class,
            patch("comet_ml.config.set_global_experiment"),
            patch.object(callback, "is_trial_forked", return_value=True),
            patch.object(callback, "_check_workspaces", return_value=0),
        ):
            mock_new_exp_1 = Mock()
            mock_new_exp_2 = Mock()
            mock_experiment_class.side_effect = [mock_new_exp_1, mock_new_exp_2]

            # Restart both experiments
            result_1 = callback._restart_experiment_for_forked_trial(forked_trial_1, fork_data_1)
            result_2 = callback._restart_experiment_for_forked_trial(forked_trial_2, fork_data_2)

            # Verify both parent experiments were ended
            mock_parent_1.end.assert_called_once()
            mock_parent_2.end.assert_called_once()

            # Verify both new experiments were created and stored correctly
            self.assertEqual(result_1, mock_new_exp_1)
            self.assertEqual(result_2, mock_new_exp_2)
            self.assertEqual(callback._trial_experiments[forked_trial_1], mock_new_exp_1)
            self.assertEqual(callback._trial_experiments[forked_trial_2], mock_new_exp_2)
            self.assertEqual(mock_experiment_class.call_count, 2)
