from __future__ import annotations

import logging
import os
import pickle
import random
import signal
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ray import tune
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.train._internal.storage import StorageContext
from ray.tune import CheckpointConfig
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.logger import (
    CSVLoggerCallback,
    JsonLoggerCallback,
    TBXLoggerCallback,
)
from ray.tune.result import SHOULD_CHECKPOINT, TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.schedulers.pb2 import PB2
from ray.tune.schedulers.pbt import PopulationBasedTraining
from ray.tune.schedulers.pbt import logger as ray_pbt_logger
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import CombinedStopper
from ray.tune.stopper.maximum_iteration import MaximumIterationStopper
from ray.tune.utils.mock_trainable import MOCK_TRAINABLE_NAME, register_mock_trainable  # noqa: PLC0415

from ray_utilities.callbacks.algorithm import exact_sampling_callback
from ray_utilities.callbacks.tuner.metric_checkpointer import StepCheckpointer  # pyright: ignore[reportDeprecated]
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.constants import (
    CURRENT_STEP,
    EVAL_METRIC_RETURN_MEAN,
    NUM_ENV_STEPS_PASSED_TO_LEARNER,
    NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
)
from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations
from ray_utilities.misc import is_pbar, raise_tune_errors
from ray_utilities.runfiles import run_tune
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.ppo_mlp_setup import MLPSetup
from ray_utilities.setup.scheduled_tuner_setup import PPOMLPWithPBTSetup
from ray_utilities.testing_utils import (
    ENV_RUNNER_CASES,
    Cases,
    DisableLoggers,
    InitRay,
    MockTrial,
    SetupWithCheck,
    TestHelpers,
    TrainableWithChecks,
    _MockTrialRunner,
    format_result_errors,
    iter_cases,
    mock_result,
    patch_args,
)
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.helpers import make_divisible
from ray_utilities.tune.scheduler.re_tune_scheduler import ReTuneScheduler
from ray_utilities.tune.scheduler.top_pbt_scheduler import CyclicMutation, KeepMutation, TopPBTTrialScheduler
from ray_utilities.tune.searcher.optuna_searcher import OptunaSearchWithPruner
from ray_utilities.tune.searcher.optuna_searcher import _logger as optuna_logger

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.tune.result_grid import ResultGrid
    from ray.tune.search.sample import Integer

    from ray_utilities.tune.scheduler.top_pbt_scheduler import _PBTTrialState2
    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData
    from ray_utilities.typing.metrics import LogMetricsDict
    from ray_utilities.typing.trainable_return import TrainableReturnData


try:
    # Needed for PB2 init
    import GPy  # pyright: ignore[reportMissingImports] # noqa: F401
except ImportError:
    sys.modules["GPy"] = MagicMock()

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
MINIBATCH_SIZE = 32


@pytest.mark.tuner
class TestTuner(InitRay, TestHelpers, DisableLoggers, num_cpus=4):
    def test_optuna_search_added(self):
        with patch_args("--tune", "batch_size", "--num_samples", "1"):
            optuna_setup = AlgorithmSetup()
            self.assertTrue(optuna_setup.args.optimize_config)
            tuner = optuna_setup.create_tuner()
            assert tuner._local_tuner and tuner._local_tuner._tune_config
            self.assertIsInstance(tuner._local_tuner._tune_config.search_alg, OptunaSearch)
            # verify metrics key
            assert tuner._local_tuner._tune_config.search_alg
            self.assertEqual(
                tuner._local_tuner._tune_config.search_alg.metric,
                EVAL_METRIC_RETURN_MEAN,
            )
        with patch_args("--num_samples", "1"):
            setup2 = AlgorithmSetup()
            self.assertFalse(setup2.args.optimize_config)
            tuner2 = setup2.create_tuner()
            assert tuner2._local_tuner and tuner2._local_tuner._tune_config
            self.assertNotIsInstance(tuner2._local_tuner._tune_config.search_alg, OptunaSearch)

    @pytest.mark.basic
    def test_max_iteration_stopper_added(self):
        # Check max iteration stopper added
        with patch_args("-it", "10"):
            setup = AlgorithmSetup()
            self.assertFalse(setup.args.optimize_config)
            tuner = setup.create_tuner()
            assert tuner._local_tuner and tuner._local_tuner._tune_config

            def check_stopper(stopper: MaximumIterationStopper | Any, _):
                self.assertEqual(stopper._max_iter, 10)
                return True

            self.assertIsInstance(
                self.check_stopper_added(tuner, MaximumIterationStopper, check=check_stopper), MaximumIterationStopper
            )

        # Check default behavior
        with patch_args():
            setup = AlgorithmSetup()
            tuner = setup.create_tuner()
            self.assertFalse(setup.args.optimize_config)
            assert tuner._local_tuner and tuner._local_tuner._tune_config

            def check_stopper(stopper: MaximumIterationStopper | Any, _):
                self.assertEqual(
                    stopper._max_iter,
                    second=calculate_iterations(
                        dynamic_buffer=False,
                        batch_size=setup.config.train_batch_size_per_learner,
                        total_steps=DefaultArgumentParser.total_steps,
                    ),
                )
                return True

            self.assertIsInstance(
                self.check_stopper_added(tuner, MaximumIterationStopper, check=check_stopper), MaximumIterationStopper
            )

        # Check Stopper not added when batch_size / iterations is adjusted dynamically after start
        with patch_args("--tune", "batch_size"):
            setup = AlgorithmSetup()
            tuner = setup.create_tuner()
            self.assertTrue(setup.args.optimize_config)
            assert tuner._local_tuner and tuner._local_tuner._tune_config

            def check_stopper(stopper: MaximumIterationStopper | Any, _):
                self.assertNotIsInstance(stopper, MaximumIterationStopper)
                return True

            self.assertNotIsInstance(
                self.check_stopper_added(tuner, MaximumIterationStopper, check=check_stopper), MaximumIterationStopper
            )

        # But added when manually adding iterations
        with patch_args("--tune", "batch_size", "-it", 3):
            setup = AlgorithmSetup()
            tuner = setup.create_tuner()
            self.assertTrue(setup.args.optimize_config)
            assert tuner._local_tuner and tuner._local_tuner._tune_config

            def check_stopper(stopper: MaximumIterationStopper | Any, _):
                self.assertEqual(stopper._max_iter, 3)
                return True

            self.assertIsInstance(
                self.check_stopper_added(tuner, MaximumIterationStopper, check=check_stopper), MaximumIterationStopper
            )

    def test_run_tune_function(self):
        batch_size = make_divisible(BATCH_SIZE, DefaultArgumentParser.num_envs_per_env_runner)
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "3",
            "--batch_size", batch_size,
            "--iterations", "3",
            "--fcnet_hiddens", "[4]",
            "--num_envs_per_env_runner", "4",
        ):  # fmt: skip
            with MLPSetup(init_trainable=False) as setup:
                setup.config.training(num_epochs=2, minibatch_size=batch_size)
                setup.config.evaluation(evaluation_interval=1)  # else eval metric not in dict
            results = run_tune(setup)
            assert not isinstance(results, dict)
            # NOTE: This can be OK even if runs fail!
            for result in results:
                assert result.metrics
                self.assertEqual(result.metrics["current_step"], 3 * batch_size)
                self.assertEqual(result.metrics[TRAINING_ITERATION], 3)
            raise_tune_errors(results)

    @pytest.mark.xfail(reason="Currently not added StepCheckpointer")
    def test_step_checkpointing(self):
        batch_size = make_divisible(BATCH_SIZE, DefaultArgumentParser.num_envs_per_env_runner)
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size", batch_size,
            "--minibatch_size", MINIBATCH_SIZE,
            "--iterations", "5",
            "--total_steps", batch_size * 5,
            "--use_exact_total_steps",  # do not adjust total_steps
            "--checkpoint_frequency_unit", "steps",
            "--checkpoint_frequency", batch_size * 2,
            "--fcnet_hiddens", "[4]",
        ):  # fmt: skip
            setup = MLPSetup()
            # Workaround for NOT working StepCheckpointer as it works on a copy of the result dict.
            # AlgoStepCheckpointer is not added, hardcoded checkpointing!
            # is_algorithm_callback_added(
            #    setup.config,
            #    AlgoStepCheckpointer.make_callback_class(checkpoint_frequency=setup.args.checkpoint_frequency),
            # )
            tuner = setup.create_tuner()
            assert tuner._local_tuner
            run_config = tuner._local_tuner.get_run_config()
            run_config.checkpoint_config.checkpoint_at_end = False  # pyright: ignore[reportOptionalMemberAccess]
            tune_callbacks = run_config.callbacks
            assert tune_callbacks
            self.assertTrue(any(isinstance(cb, StepCheckpointer) for cb in tune_callbacks))
            results = tuner.fit()
            path, checkpoints = self.get_checkpoint_dirs(results[0])
            # restore latest checkpoint
            self.assertEqual(
                len(checkpoints),
                2,  # + 1 for save at end?
                "Expected 2 checkpoints, got: " + str(checkpoints) + " found " + str(os.listdir(path)),
            )
            checkpoints = sorted(checkpoints)
            with patch_args(
                "--from_checkpoint",
                checkpoints[-1],
                "--num_jobs",
                "1",
            ):
                t2 = setup.trainable_class.from_checkpoint(checkpoints[-1])
                assert t2._current_step == batch_size * 4, (
                    f"Expected current_step to be {batch_size * 4}, got {t2._current_step}"
                )

    def test_checkpoint_force_by_trial_callback(self):
        """
        Test that cloud syncing is forced if one of the trials has made more
        than num_to_keep checkpoints since last sync.
        Legacy test: test_trial_runner_3.py::TrialRunnerTest::
            testCloudCheckpointForceWithNumToKeep
        """
        import tempfile  # noqa: PLC0415
        from typing import Optional  # noqa: PLC0415

        from ray.air.execution import PlacementGroupResourceManager  # noqa: PLC0415
        from ray.train import SyncConfig  # noqa: PLC0415, TC002
        from ray.tune import Callback, CheckpointConfig  # noqa: PLC0415
        from ray.tune.execution.tune_controller import TuneController  # noqa: PLC0415

        register_mock_trainable()

        def mock_storage_context(
            exp_name: str = "exp_name",
            storage_path: Optional[str] = None,
            storage_context_cls: type = StorageContext,
            sync_config: Optional[SyncConfig] = None,
        ) -> StorageContext:
            trial_name = "trial_name"

            storage = storage_context_cls(
                storage_path=storage_path,
                experiment_dir_name=exp_name,
                trial_dir_name=trial_name,
                sync_config=sync_config,
            )
            # Patch the default /tmp/ray/session_* so we don't require ray
            # to be initialized in unit tests.
            session_path = tempfile.mkdtemp()
            storage._get_session_path = lambda: session_path

            os.makedirs(storage.trial_fs_path, exist_ok=True)
            os.makedirs(storage.trial_driver_staging_path, exist_ok=True)

            return storage

        class CheckpointCallback(Callback):
            num_checkpoints = 0

            def on_trial_result(self, iteration, trials, trial: Trial, result, **info):
                # Checkpoint every two iterations
                if result[TRAINING_ITERATION] % 2 == 0:
                    self.num_checkpoints += 1

        with tempfile.TemporaryDirectory() as local_dir:
            storage = mock_storage_context(storage_path=local_dir)

            # disable automatic checkpointing
            checkpoint_config = CheckpointConfig(checkpoint_frequency=0)
            callback = CheckpointCallback()
            runner = TuneController(
                resource_manager_factory=PlacementGroupResourceManager,
                storage=storage,
                callbacks=[callback],
                trial_checkpoint_config=checkpoint_config,
            )

            trial = Trial(
                MOCK_TRAINABLE_NAME,
                checkpoint_config=checkpoint_config,
                stopping_criterion={"training_iteration": 6},
                storage=storage,
            )
            runner.add_trial(trial)

            while not runner.is_finished():
                runner.step()

            assert callback.num_checkpoints == 3

    @mock.patch("ray.tune.impl.tuner_internal.StorageContext", new=MagicMock())
    def test_loggers_added(self):
        def fake_trainable():
            pass

        with patch_args("--offline_loggers", 0):
            setup = MLPSetup(init_trainable=False)
            setup.trainable = fake_trainable
            self.assertIn("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", os.environ)
            self.assertEqual(os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"], "1")
            tuner = setup.create_tuner()
            callbacks = tuner._local_tuner.get_run_config().callbacks  # pyright: ignore[reportOptionalMemberAccess]
            if callbacks:
                assert not any(c for c in callbacks if isinstance(c, CSVLoggerCallback)), callbacks
                assert not any(c for c in callbacks if isinstance(c, JsonLoggerCallback)), callbacks
                assert not any(c for c in callbacks if isinstance(c, TBXLoggerCallback)), callbacks
            del os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"]
        with patch_args("--offline_loggers", "all"):
            setup = MLPSetup(init_trainable=False)
            self.assertTrue(setup.args.offline_loggers)
            setup.trainable = fake_trainable
            if "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS" in os.environ:
                self.assertEqual(os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"], "0")
            tuner = setup.create_tuner()

            callbacks = tuner._local_tuner.get_run_config().callbacks  # pyright: ignore[reportOptionalMemberAccess]
            if callbacks:
                assert any(c for c in callbacks if isinstance(c, CSVLoggerCallback)), "csv missing"
                assert any(c for c in callbacks if isinstance(c, JsonLoggerCallback)), "json missing"
                assert any(c for c in callbacks if isinstance(c, TBXLoggerCallback)), "tbx missing"
            if "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS" in os.environ:
                del os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"]

        # No online loggers -> all offline
        with patch_args("--wandb", "0", "--comet", "0"):
            setup = MLPSetup(init_trainable=False)
            setup.trainable = fake_trainable
            tuner = setup.create_tuner()
            self.assertEqual(setup.args.offline_loggers, True)
            self.assertEqual(os.environ.get("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "0"), "0")

            callbacks = tuner._local_tuner.get_run_config().callbacks  # pyright: ignore[reportOptionalMemberAccess]
            if callbacks:
                assert any(c for c in callbacks if isinstance(c, CSVLoggerCallback)), "no csv"
                assert any(c for c in callbacks if isinstance(c, JsonLoggerCallback)), "no json"
                assert any(c for c in callbacks if isinstance(c, TBXLoggerCallback)), "no tbx"
                # default
            if "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS" in os.environ:
                del os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"]
        with patch_args("--wandb", "offline"):
            setup = MLPSetup(init_trainable=False)
            setup.trainable = fake_trainable
            tuner = setup.create_tuner()
            self.assertEqual(os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"], "1")
            callbacks = tuner._local_tuner.get_run_config().callbacks  # pyright: ignore[reportOptionalMemberAccess]
            if callbacks:
                assert not any(c for c in callbacks if isinstance(c, CSVLoggerCallback))
                assert any(c for c in callbacks if isinstance(c, JsonLoggerCallback))
                assert not any(c for c in callbacks if isinstance(c, TBXLoggerCallback))
            if "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS" in os.environ:
                del os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"]

    def test_restore_after_sigusr1(self):
        """Test that experiments can be restored after SIGUSR1 signal interruption."""
        batch_size = make_divisible(BATCH_SIZE, DefaultArgumentParser.num_envs_per_env_runner)
        self._setup_backup_mock.stop()

        def send_signal_after_delay(delay_seconds: float):
            """Send SIGUSR1 to current process after delay."""
            time.sleep(delay_seconds)
            logger.info("Sending SIGUSR1 signal to abort tuning")
            os.kill(os.getpid(), signal.SIGUSR1)

        ITERATIONS = 110

        with patch_args(
            "--num_samples", "2",
            "--num_jobs", "2",
            "--batch_size", batch_size,
            "--iterations", ITERATIONS,
            "--fcnet_hiddens", "[4]",
            "--num_envs_per_env_runner", "1",
            "--checkpoint_frequency_unit", "iterations",
            "--checkpoint_frequency", "50",
            "--offline_loggers", "json",
            "--log_stats", "more",
            "--buffer_length", "1",
        ):  # fmt: skip
            with MLPSetup(init_trainable=False) as setup1:
                setup1.config.training(num_epochs=2, minibatch_size=batch_size)
                setup1.config.evaluation(evaluation_interval=1)

        tuner1 = setup1.create_tuner()
        run_config = tuner1._local_tuner.get_run_config()  # pyright: ignore[reportOptionalMemberAccess]
        experiment_path = Path(run_config.storage_path) / run_config.name  # pyright: ignore[reportArgumentType, reportOperatorIssue]

        # Start signal thread - interrupt after some trials start
        signal_thread = threading.Thread(target=send_signal_after_delay, args=(20.0,))
        signal_thread.daemon = True
        signal_thread.start()

        results1 = tuner1.fit()

        logger.warning("Tuning interrupted, results so far: %s", results1)

        # Check if we were interrupted
        try:
            with self.assertRaisesRegex(RuntimeError, "no trial has reported this metric"):
                results1.get_best_result()
        except AssertionError:
            # check that results are not completed
            best_result: tune.Result = results1.get_best_result()
            if TRAINING_ITERATION not in best_result.metrics:  # pyright: ignore[reportOperatorIssue]
                # happens if trial is still in pending state
                self.fail("training_iterations not in metrics: keys: " + str(best_result.metrics.keys()))  # pyright: ignore[reportOptionalMemberAccess]
            self.assertLess(best_result.metrics[TRAINING_ITERATION], ITERATIONS)  # pyright: ignore[reportOptionalSubscript]
        # If we get here without interruption, still check results

        signal_thread.join(timeout=2.0)

        # Verify experiment directory exists
        self.assertTrue(
            os.path.exists(experiment_path) or str(experiment_path).startswith("s3:/"),
            f"Experiment path should exist: {experiment_path}",
        )

        # Now restore from checkpoint
        with patch_args(
            "--restore_path",
            str(experiment_path),
        ):
            setup2 = MLPSetup()
        del os.environ["RAY_UTILITIES_RESTORED"]
        self.maxDiff = None
        state1 = setup1.get_state()
        state2 = setup2.get_state()
        config1_state = cast("AlgorithmConfig", state1.pop("config"))
        config2_state = cast("AlgorithmConfig", state2.pop("config"))
        seed1 = cast("Integer", state1["param_space"].pop("env_seed"))
        seed2 = cast("Integer", state2["param_space"].pop("env_seed"))
        # Only RestoreOverride keys, like restore_path are allowed to differ
        state2["args"].restore_path = None
        self.compare_param_space(state1.pop("tune_parameters"), state2.pop("tune_parameters"))  # pyright: ignore[reportArgumentType]
        self.assertDictEqual(state1, state2)
        self.compare_configs(config1_state, config2_state)
        self.assertEqual((seed1.lower, seed1.upper), (seed2.lower, seed2.upper))

        # Second run should complete all trials
        results2 = cast("ResultGrid", run_tune(setup2))
        raise_tune_errors(results2)  # pyright: ignore[reportArgumentType]

        # Verify all trials completed
        self.assertEqual(len(results2), 2, "Should have 2 completed trials")
        for result in results2:
            assert result.metrics
            self.assertEqual(
                result.metrics[TRAINING_ITERATION],
                ITERATIONS,
                f"Trial should complete 10 iterations, got {result.metrics[TRAINING_ITERATION]}",
            )
            self.assertEqual(result.metrics[CURRENT_STEP], ITERATIONS * batch_size)
        self.assertIsNotNone(results2.get_best_result())

        self._setup_backup_mock.start()


@pytest.mark.tuner
class TestTunerCheckpointing(InitRay, TestHelpers, DisableLoggers):
    def test_checkpoint_auto(self):
        # self.enable_loggers()
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size", BATCH_SIZE,
            "--minibatch_size", MINIBATCH_SIZE,
            "--iterations", "4",
            "--fcnet_hiddens", "[4]",
        ):  # fmt: skip
            setup = MLPSetup()
        tuner = setup.create_tuner()
        tuner._local_tuner.get_run_config().checkpoint_config = (  # pyright: ignore[reportOptionalMemberAccess]
            tune.CheckpointConfig(
                checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                checkpoint_score_order="max",
                checkpoint_frequency=2,  # Save every two iterations
                # NOTE: num_keep does not appear to work here
            )
        )
        results = tuner.fit()
        raise_tune_errors(results)
        checkpoint_dir, checkpoints = self.get_checkpoint_dirs(results[0])
        self.assertEqual(
            len(checkpoints),
            2,  # 4 iterations / 2 checkpoint frequency = 2 checkpoints
            f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir)} in {checkpoint_dir}",
        )

    def test_checkpoint_manually(self):
        # self.enable_loggers()
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size", BATCH_SIZE,
            "--minibatch_size", MINIBATCH_SIZE,
            "--iterations", "2",
            "--fcnet_hiddens", "[4]",
        ):  # fmt: skip

            class CheckpointSetup(MLPSetup):
                def _create_trainable(self):
                    class DefaultTrainableWithCheckpoint(DefaultTrainable):
                        def step(self):
                            result = super().step()
                            result[SHOULD_CHECKPOINT] = True
                            if is_pbar(self._pbar):
                                self._pbar.update(1)
                            return result

                        def save_checkpoint(self, checkpoint_dir: str):
                            saved = super().save_checkpoint(checkpoint_dir)
                            logger.info("Checkpoint saved to %s", checkpoint_dir)
                            return saved

                    return DefaultTrainableWithCheckpoint.define(self)

            setup = CheckpointSetup()
        tuner = setup.create_tuner()
        results = tuner.fit()
        self.assertEqual(results.num_errors, 0, "Encountered errors: " + format_result_errors(results.errors))  # pyright: ignore[reportAttributeAccessIssue,reportOptionalSubscript]
        checkpoint_dir, checkpoints = self.get_checkpoint_dirs(results[0])
        self.assertEqual(
            len(checkpoints),
            2,
            f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir)} in {checkpoint_dir}",
        )


@pytest.mark.tuner
@pytest.mark.length(speed="medium")
class TestReTuning(InitRay, TestHelpers, DisableLoggers, num_cpus=4):
    @Cases(ENV_RUNNER_CASES)
    def test_retune_with_different_config(self, cases):
        # self.enable_loggers()
        NUM_ITERS_2 = 3
        batch_size = make_divisible(BATCH_SIZE, DefaultArgumentParser.num_envs_per_env_runner)

        class TrainableWithChecksA(TrainableWithChecks):
            def setup_check(self, *args, **kwargs):
                assert self._iteration == 1, "Trainable should be setup with iteration 1"

            def step_pre_check(self):
                assert self.algorithm_config.train_batch_size_per_learner == batch_size * 2, (
                    "Batch size should be 2x the original batch size, "
                    f"not {self.algorithm_config.train_batch_size_per_learner}"
                )

            def step_post_check(self, result: StrictAlgorithmReturnData, metrics: LogMetricsDict, rewards):
                assert result["training_iteration"] >= 2, (
                    f"Expected training_iteration to be at least 2, got {result['training_iteration']}"
                )
                expected = 2 * batch_size
                expected_lifetime = (
                    expected * (result["training_iteration"] - 1) + batch_size
                )  # first step + 2 iterations
                # Do not compare ENV_STEPS_SAMPLED when using multiple envs per env runner
                assert NUM_ENV_STEPS_PASSED_TO_LEARNER in result[ENV_RUNNER_RESULTS]
                value = result[ENV_RUNNER_RESULTS].get(NUM_ENV_STEPS_PASSED_TO_LEARNER, None)
                assert value == expected, f"Expected {expected} env steps passed to learner, got {value}"
                assert (
                    value := result[ENV_RUNNER_RESULTS].get(NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME, None)
                ) == expected_lifetime, (
                    f"Expected {expected_lifetime} env steps passed to learner lifetime, got {value}"
                )

        Setup = SetupWithCheck(TrainableWithChecksA)

        for num_env_runners in iter_cases(cases):
            with self.subTest(num_env_runners=num_env_runners):
                with (
                    patch_args(
                        "--num_samples", "1",
                        "--num_jobs", "1",
                        "--batch_size", batch_size,  # overwrite
                        "--use_exact_total_steps",  # do not adjust total_steps
                        "--minibatch_size", MINIBATCH_SIZE,  # keep
                        "--iterations", "1",  # overwrite
                    ),
                     AlgorithmSetup() as setup1
                ):  # fmt: skip
                    setup1.config.env_runners(num_env_runners=num_env_runners)
                tuner1 = setup1.create_tuner()
                assert tuner1._local_tuner
                tuner1._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(
                    checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                    checkpoint_score_order="max",
                    checkpoint_frequency=1,
                )
                results1 = tuner1.fit()
                del tuner1
                self.assertEqual(
                    results1.num_errors, 0, "Encountered errors: " + format_result_errors((results1.errors))
                )
                # Check metrics:
                result1 = results1[0]
                assert result1.metrics and result1.config
                self.assertEqual(result1.metrics[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME], batch_size)
                self.assertEqual(result1.metrics[TRAINING_ITERATION], 1)
                checkpoint_dir, checkpoints = self.get_checkpoint_dirs(results1[0])
                self.assertEqual(
                    len(checkpoints),
                    1,
                    f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir)} in {checkpoint_dir}",
                )
                self.assertTrue(os.path.exists(checkpoints[0]), "Checkpoint file does not exist: " + checkpoints[0])

                with patch_args(
                    "--num_samples", "1",
                    "--num_jobs", "1",
                    "--batch_size", batch_size * 2,
                    "--minibatch_size", MINIBATCH_SIZE,
                    "--total_steps", batch_size * 2 * NUM_ITERS_2 + batch_size,  # 1 + NUM_ITERS_2 iterations
                    "--use_exact_total_steps",  # do not adjust total_steps
                    "--from_checkpoint", checkpoints[0],
                    "--log_stats", "most",
                ):  # fmt: skip
                    with (
                        Setup() as setup2,
                        Setup() as setup2b,
                    ):  # second setup to make sure no side-effects are tested
                        setup2.config.env_runners(num_env_runners=num_env_runners)
                        setup2b.config.env_runners(num_env_runners=num_env_runners)
                    self.assertEqual(setup2.args.total_steps, batch_size * 2 * NUM_ITERS_2 + batch_size)
                    # Auto iteration will be 4; but only 3 new should be done.
                    self.assertEqual(setup2.args.train_batch_size_per_learner, batch_size * 2)
                Trainable2 = setup2b.create_trainable()
                if TYPE_CHECKING:
                    Trainable2 = setup2b.trainable_class
                trainable2_local = Trainable2(setup2b.sample_params())
                if trainable2_local.algorithm_config.callbacks_on_sample_end and isinstance(
                    trainable2_local.algorithm_config.callbacks_on_sample_end, Iterable
                ):
                    # check that exactly one callback is an exact_sampling_cb
                    self.assertEqual(
                        len(
                            {
                                cb
                                for cb in trainable2_local.algorithm_config.callbacks_on_sample_end
                                if cb.__name__ == exact_sampling_callback.__name__.split(".")[-1]
                            }
                        ),
                        1,
                    )
                self.maxDiff = None
                # DynamicEvalInterval can change evaluation_interval; ignore that key
                self.compare_configs(
                    trainable2_local.algorithm_config.to_dict(),
                    setup2.config.to_dict(),
                    ignore=("evaluation_interval",),
                )
                self.assertEqual(trainable2_local.algorithm_config.train_batch_size_per_learner, batch_size * 2)
                trainable2_local.stop()

                tuner2 = setup2.create_tuner()
                assert tuner2._local_tuner
                tuner2._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(
                    checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                    checkpoint_score_order="max",
                    checkpoint_frequency=1,
                )
                assert tuner2._local_tuner

                # assert that the stopper stops not too early, e.g. because parser.args.iterations was not updated.
                def check_stopper(stopper: MaximumIterationStopper | Any, _):
                    self.assertEqual(stopper._max_iter, NUM_ITERS_2 + 1)
                    return True

                self.check_stopper_added(tuner2, MaximumIterationStopper, check=check_stopper)
                results2 = tuner2.fit()

                self.assertEqual(results2.num_errors, 0, "Encountered errors: " + format_result_errors(results2.errors))  # pyright: ignore[reportAttributeAccessIssue,reportOptionalSubscript]
                result2 = results2[0]
                assert result2.metrics
                self.assertIn(
                    "_checking_class_",
                    result2.metrics,
                    "Metrics should contain '_checking_class_'. Custom class was likely not used",
                )
                # Check iterations change
                self.assertEqual(result2.metrics["current_step"], batch_size * 2 * NUM_ITERS_2 + batch_size)
                self.assertEqual(result2.metrics[TRAINING_ITERATION], NUM_ITERS_2 + 1)
                self.assertEqual(result2.metrics["iterations_since_restore"], NUM_ITERS_2)

                # Change batch size change:
                # do not check NUM_ENV_STEPS_SAMPLED_LIFETIME when using multiple envs per env runner
                self.assertEqual(
                    result2.metrics[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
                    batch_size * 2 * NUM_ITERS_2 + batch_size,
                )
                checkpoint_dir2, checkpoints2 = self.get_checkpoint_dirs(results2[0])
                self.assertEqual(
                    len(checkpoints2),
                    NUM_ITERS_2,  # 2 checkpoints as in total 3 steps; or does it save ?
                    f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir2)} in {checkpoint_dir2}",
                )
                self.assertTrue(os.path.exists(checkpoints2[0]), "Checkpoint file does not exist: " + checkpoints2[0])

    @Cases([0])  # more env runners should have no influence here
    def test_retune_with_tune_argument(self, cases):
        batch_size = make_divisible(BATCH_SIZE, DefaultArgumentParser.num_envs_per_env_runner)

        class TrainableWithChecksB(TrainableWithChecks):
            debug_step = False

        Setup = SetupWithCheck(TrainableWithChecksB)

        for num_env_runners in iter_cases(cases):
            with self.subTest(num_env_runners=num_env_runners):
                with (
                    patch_args(
                        "--num_samples", "1",
                        "--num_jobs", 1,
                        "--batch_size", batch_size,  # overwrite
                        "--minibatch_size", MINIBATCH_SIZE,  # keep
                        "--iterations", "1",  # overwrite
                    ),
                    AlgorithmSetup() as setup1,
                ):  # fmt: skip
                    setup1.config.env_runners(num_env_runners=num_env_runners)
                tuner1 = setup1.create_tuner()
                assert tuner1._local_tuner
                tuner1._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(
                    checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                    checkpoint_score_order="max",
                    checkpoint_frequency=1,
                )
                results1 = tuner1.fit()
                self.assertEqual(
                    results1.num_errors, 0, "Encountered errors: " + format_result_errors((results1.errors))
                )
                checkpoint_dir, checkpoints = self.get_checkpoint_dirs(results1[0])
                self.assertEqual(
                    len(checkpoints),
                    1,
                    f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir)} in {checkpoint_dir}",
                )

                NUM_RUNS = 5
                # Define multipliers for sample space; all values will be divisible by num_envs_per_env_runner
                SAMPLE_SPACE_MULTIPLIERS = [4, 15, 19]
                assert setup1.config.num_envs_per_env_runner
                SAMPLE_SPACE = [max(setup1.config.num_envs_per_env_runner * m, 16) for m in SAMPLE_SPACE_MULTIPLIERS]
                with patch_args(
                    "--num_samples", NUM_RUNS,
                    "--num_jobs", 2 if num_env_runners > 0 else 4,
                    "--from_checkpoint", checkpoints[0],
                    "--log_stats", "most",
                    "--tune", "batch_size", "minibatch_size",
                    "--iterations", "3",
                ):  # fmt: skip
                    with Setup() as setup2:
                        setup2.config.env_runners(num_env_runners=num_env_runners)
                    setup2.param_space["minibatch_size"] = {"grid_search": SAMPLE_SPACE}
                    setup2.param_space["train_batch_size_per_learner"] = {"grid_search": SAMPLE_SPACE}
                assert setup2.config.num_envs_per_env_runner is not None
                self.assertTrue(all(s % setup2.config.num_envs_per_env_runner == 0 for s in SAMPLE_SPACE))
                tuner2 = setup2.create_tuner()
                assert tuner2._local_tuner
                tuner2._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(
                    checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                    checkpoint_score_order="max",
                    checkpoint_frequency=1,
                )
                # Tuner will sample parameter and feed is as input arg to the trainable
                results2 = tuner2.fit()
                raise_tune_errors(results2)
                self.assertEqual(
                    results2.num_terminated, NUM_RUNS, f"Expected num_samples={NUM_RUNS} runs to be terminated"
                )
                self.assertEqual(len(results2), NUM_RUNS)
                batch_sizes = set()
                minibatch_sizes = set()

                checkpoints = {}
                for result in results2:
                    assert result.config
                    batch_sizes.add(result.config["train_batch_size_per_learner"])
                    minibatch_sizes.add(result.config["minibatch_size"])
                    checkpoints[result.checkpoint] = result
                # check that different values were added
                if NUM_RUNS >= 7:
                    self.assertEqual(batch_sizes, set(SAMPLE_SPACE), f"{batch_sizes} != {set(SAMPLE_SPACE)}")
                else:
                    self.assertLessEqual(
                        batch_sizes, set(SAMPLE_SPACE), f"{batch_sizes} not a subset of {set(SAMPLE_SPACE)}"
                    )
                # check if values match expectations
                for result in results2:
                    assert result.metrics and result.config
                    self.assertEqual(result.metrics[TRAINING_ITERATION], 3)
                    self.assertEqual(result.metrics["iterations_since_restore"], 2)
                    # Check that new batch_size was used
                    self.assertEqual(
                        result.metrics["current_step"],
                        result.config["train_batch_size_per_learner"] * 2 + batch_size,
                        "Expected current_step to be 2x the batch size + initial step",
                    )


class TestOptunaTuner(TestHelpers, DisableLoggers):
    def test_optuna_tuner_setup(self):
        with patch_args("--tune", "batch_size", "--num_samples", "1"):
            optuna_setup = AlgorithmSetup()
            self.assertTrue(optuna_setup.args.optimize_config)
            tuner = optuna_setup.create_tuner()
            assert tuner._local_tuner and tuner._local_tuner._tune_config
            self.assertIsInstance(tuner._local_tuner._tune_config.search_alg, OptunaSearch)
            stopper = tuner._local_tuner.get_run_config().stop
            if isinstance(stopper, CombinedStopper):
                self.assertTrue(any(isinstance(s, OptunaSearchWithPruner) for s in stopper._stoppers))
                optuna_stoppers = [s for s in stopper._stoppers if isinstance(s, OptunaSearchWithPruner)]
                self.assertEqual(len(optuna_stoppers), 1)
                optuna_stopper = optuna_stoppers[0]
            else:
                self.assertIsInstance(stopper, OptunaSearchWithPruner)
                optuna_stopper = stopper
                # stopper and search are the same
            self.assertIs(optuna_stopper, tuner._local_tuner._tune_config.search_alg)
        with patch_args("--num_samples", "1"):
            setup2 = AlgorithmSetup()
            self.assertFalse(setup2.args.optimize_config)
            tuner2 = setup2.create_tuner()
            assert tuner2._local_tuner and tuner2._local_tuner._tune_config
            self.assertNotIsInstance(tuner2._local_tuner._tune_config.search_alg, OptunaSearch)
            self.assertNotIsInstance(tuner2._local_tuner.get_run_config().stop, OptunaSearchWithPruner)

    @pytest.mark.tuner
    @pytest.mark.flaky(max_runs=2, min_passes=1)
    def test_pruning(self):
        """
        Test might fail due to bad luck, low numbers first then high.

        Note:
            Remember Optuna might not prune the first 10 trials.
            Reduce num_jobs or adjust seed and test again.
        """
        MAX_STEP = 15
        with patch_args(
            "--optimize_config", "--pruner_warmup_steps", 3, "--num_samples", "15", "--num_jobs", "4", "--seed", "42"
        ):

            def trainable(params: dict[str, Any]) -> TrainableReturnData:
                logger.info("Running trainable with value: %s", params["fake_result"])
                for i in range(MAX_STEP):
                    tune.report(
                        {
                            "current_step": i,
                            EVALUATION_RESULTS: {ENV_RUNNER_RESULTS: {EPISODE_RETURN_MEAN: params["fake_result"]}},
                        }
                    )
                return {
                    "done": True,
                    "current_step": MAX_STEP,
                    EVALUATION_RESULTS: {ENV_RUNNER_RESULTS: {EPISODE_RETURN_MEAN: params["fake_result"]}},
                }

            class RandomParamsSetup(AlgorithmSetup):
                PROJECT = "OptunaTest"

                def create_param_space(self):
                    return {
                        "fake_result": tune.grid_search([*[1] * 3, *[2] * 3, *[3] * 3, *[4] * 3, *[5] * 3]),
                        # Need non-trivial search space for OptunaSearch & Pruner to be added
                        "random_feature": tune.uniform(0, 1),
                        "module": "OptunaTest",
                        "env": "CartPole-v1",
                    }

                def _create_trainable(self):  # type: ignore[override]
                    return trainable

            with RandomParamsSetup() as setup:
                setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=64)
                setup.config.evaluation(evaluation_interval=1)

            # Removed the Optuna Pruner for grid search cases
            with self.assertLogs(optuna_logger, level="INFO") as log:
                _results = run_tune(setup)

            self.assertTrue(
                any("Optuna pruning trial" in out for out in log.output),
                "Logger did not report a pruned trial, searching for 'Optuna pruning trial' in:\n"
                + "\n".join(log.output),
            )
            self.assertGreaterEqual(
                len([result for result in _results if result.metrics["current_step"] < MAX_STEP]),  # pyright: ignore[reportOptionalSubscript,reportAttributeAccessIssue]
                3,
            )
            # Check that all results have at least 3 iterations (pruned trials may have less)
            for result in _results:
                assert result.metrics is not None  # pyright: ignore[reportAttributeAccessIssue]
                self.assertGreaterEqual(result.metrics.get(TRAINING_ITERATION, 0), 3)  # pyright: ignore[reportAttributeAccessIssue]


class TestReTuneScheduler(TestHelpers, DisableLoggers, InitRay, num_cpus=4):
    # Some tests taken from ray's testing suite

    def setUp(self):
        super().setUp()
        self.NUM_TRIALS = 5
        self.batch_size_mutations: dict[str, Any] = {"train_batch_size_per_learner": {"grid_search": [32, 64, 128]}}
        self.setup = AlgorithmSetup(init_trainable=False)

    def setup_scheduler(self, *, num_trials=None, step_once=True, tmpdir, trial_config=None, hyperparam_mutations=None):
        num_trials = self.NUM_TRIALS if num_trials is None else num_trials
        hyperparam_mutations = hyperparam_mutations or self.batch_size_mutations
        trial_config = trial_config or self.setup.sample_params()
        if "train_batch_size_per_learner" not in trial_config:
            trial_config["train_batch_size_per_learner"] = 32
        self.storage = StorageContext(storage_path=tmpdir, experiment_dir_name="test_re_scheduler")
        SYNCH = False
        scheduler = ReTuneScheduler(
            perturbation_interval=10,
            mode="max",
            hyperparam_mutations=hyperparam_mutations,
            synch=SYNCH,
            metric="episode_reward_mean",
            quantile_fraction=0.99,
        )
        runner = _MockTrialRunner(scheduler)
        for i in range(num_trials):
            trial = MockTrial(i, trial_config, self.storage)
            trial.init_local_path()
            # runner calls add_trial on step
            runner.add_trial(trial)  # calls ReTuner.add_trial
            trial.status = Trial.RUNNING
        for i in range(num_trials):
            trial = runner.trials[i]
            if step_once:
                if SYNCH:
                    self.check_on_trial_result(
                        scheduler,
                        runner,
                        trial,
                        mock_result(10, 50 * i),
                        expected_decision=ReTuneScheduler.PAUSE,
                    )
                else:
                    self.check_on_trial_result(
                        scheduler,
                        runner,
                        trial,
                        mock_result(10, 50 * i),
                        expected_decision=ReTuneScheduler.CONTINUE,
                    )
        # num_checkpoint increases if trial is in upper_quantile
        try:
            scheduler.reset_stats()  # set checkpoints to 0
        except AttributeError:
            logger.exception("Failed to reset scheduler stats. Likely due to new interface")
            scheduler._num_checkpoints = 0
        return scheduler, runner

    # Test based on ray's testing suite
    def check_on_trial_result(self, pbt: ReTuneScheduler, runner, trial: Trial, result, expected_decision=None):
        trial.status = Trial.RUNNING
        decision = pbt.on_trial_result(runner, trial, result)
        if expected_decision is None:
            pass
        elif expected_decision == ReTuneScheduler.PAUSE:
            self.assertTrue(
                trial.status == Trial.PAUSED or decision == expected_decision  # pyright: ignore[reportUnnecessaryComparison]
            )
        elif expected_decision == ReTuneScheduler.CONTINUE:
            self.assertEqual(decision, expected_decision)
        return decision

    def test_retuner_basics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler, runner = self.setup_scheduler(tmpdir=tmpdir)

    def testPerturbsLowPerformingTrials(self):  # noqa: N802
        with tempfile.TemporaryDirectory() as tmpdir:
            pbt, runner = self.setup_scheduler(tmpdir=tmpdir)
            trials: list[Trial] = runner.get_trials()
            lower_quantile, upper_quantile = pbt._quantiles()
            # Assume only one top trial
            self.assertEqual(lower_quantile, trials[:-1], "lower_quantile mismatch")
            self.assertEqual(upper_quantile, trials[-1:], "upper_quantile mismatch")
            # self.assertNotIn(trials[2], lower_quantile, "trial 2 should not be in lower quantile")  # if 0.25
            # self.assertNotIn(trials[2], upper_quantile, "trial 2 should not be in upper quantile")  # if 0.25
            self.assertTrue(trials, "trials should not be empty")

            # no perturbation: haven't hit next perturbation interval
            self.check_on_trial_result(pbt, runner, trials[0], mock_result(15, -100), ReTuneScheduler.CONTINUE)
            # Assumes self.NUM_TRIALS = 5
            # Trial 0, score 0, trial 4 score 200
            self.assertEqual(pbt.last_scores(trials), [0, 50, 100, 150, 200])
            self.assertEqual(pbt._num_perturbations, 0)
            self.assertNotIn("@perturbed", trials[0].experiment_tag)

            # Perturbation only happens in lower quantile (0.5) by default (max with ray implementation)

            # perturb since it's lower quantile
            self.check_on_trial_result(pbt, runner, trials[0], mock_result(20, -100), ReTuneScheduler.PAUSE)
            self.assertEqual(pbt.last_scores(trials), [-100, 50, 100, 150, 200])
            self.assertIn("@perturbed", trials[0].experiment_tag)
            self.assertIn(trials[0].restored_checkpoint, ["trial_3", "trial_4"])  # pyright: ignore[reportAttributeAccessIssue] # from mock
            self.assertEqual(pbt._num_perturbations, 1)

            # also perturbed as trial[2] now in lower quantile
            self.check_on_trial_result(pbt, runner, trials[2], mock_result(20, 40), ReTuneScheduler.PAUSE)
            self.assertEqual(pbt.last_scores(trials), [-100, 50, 40, 150, 200])
            self.assertEqual(pbt._num_perturbations, 2)
            self.assertIn(trials[2].restored_checkpoint, ["trial_3", "trial_4"])  # pyright: ignore[reportAttributeAccessIssue] # from mock
            self.assertIn("@perturbed", trials[2].experiment_tag)

            # trial 1 is in neither quantile if quantile 0.25 is used (ray default)
            pbt._quantile_fraction = 0.25
            self.check_on_trial_result(pbt, runner, trials[1], mock_result(20, 100), ReTuneScheduler.PAUSE)
            self.assertEqual(pbt.last_scores(trials), [-100, 100, 40, 150, 200])
            self.assertEqual(pbt._num_perturbations, 2)
            self.assertIsNone(trials[1].restored_checkpoint)  # pyright: ignore[reportAttributeAccessIssue] # from mock
            self.assertNotIn("@perturbed", trials[1].experiment_tag)

    def testCheckpointsMostPromisingTrials(self):  # noqa: N802
        # taken from ray's testing suite
        with tempfile.TemporaryDirectory() as tmpdir:
            pbt, runner = self.setup_scheduler(tmpdir=tmpdir)
            trials = runner.get_trials()
            self.assertEqual(pbt.last_scores(trials), [0, 50, 100, 150, 200])

            pbt._quantile_fraction = 3 / 5
            # As 200=200 but trial will not be in 99% quantile, decision is NOOP
            # NOOP means the trial is paused, and new checkpoint can be loaded
            # TODO: Should continue trial if nearly good
            # no checkpoint: haven't hit next perturbation interval yet
            self.check_on_trial_result(pbt, runner, trials[0], mock_result(15, 200), ReTuneScheduler.CONTINUE)
            self.assertEqual(pbt.last_scores(trials), [0, 50, 100, 150, 200])
            self.assertEqual(pbt._num_checkpoints, 0)

            # checkpoint: both past interval and upper quantile
            self.check_on_trial_result(pbt, runner, trials[0], mock_result(20, 200), ReTuneScheduler.CONTINUE)
            self.assertEqual(pbt.last_scores(trials), [200, 50, 100, 150, 200])
            self.assertEqual(pbt._num_checkpoints, 1)
            self.check_on_trial_result(pbt, runner, trials[1], mock_result(30, 201), ReTuneScheduler.CONTINUE)
            self.assertEqual(pbt.last_scores(trials), [200, 201, 100, 150, 200])
            self.assertEqual(pbt._num_checkpoints, 2)

            # not upper quantile any more, only top 2 are kept -> Pause
            self.check_on_trial_result(pbt, runner, trials[4], mock_result(30, 199), ReTuneScheduler.PAUSE)
            self.assertEqual(pbt.last_scores(trials), [200, 201, 100, 150, 199])
            self.assertEqual(pbt._num_checkpoints, 2)
            self.assertEqual(pbt._num_perturbations, 1)
            self.assertIn(trials[4].restored_checkpoint, ["trial_0", "trial_1"])

    def testCheckpointing(self):  # noqa: N802
        # taken from ray's testing suite
        with tempfile.TemporaryDirectory() as tmpdir:
            pbt, runner = self.setup_scheduler(tmpdir=tmpdir)

            class Experiment(tune.Trainable):
                def step(self):
                    return {"episode_reward_mean": self.training_iteration, "current_step": self.training_iteration * 5}

                def save_checkpoint(self, checkpoint_dir):
                    checkpoint = os.path.join(checkpoint_dir, "checkpoint")
                    self._ckpt = checkpoint
                    with open(checkpoint, "w") as f:
                        f.write("OK")
                    print("Checkpoint saved to", checkpoint)

                def reset_config(self, config) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
                    return True

                def load_checkpoint(self, checkpoint):
                    pass

                def restore(self, checkpoint_path):
                    # does not receive correct path
                    return

            trial_hyperparams = {"train_batch_size_per_learner": 32}

            analysis = tune.run(
                Experiment,
                num_samples=3,
                scheduler=pbt,
                checkpoint_config=CheckpointConfig(checkpoint_frequency=3, num_to_keep=None),
                config=trial_hyperparams,
                stop={"training_iteration": 30},
                time_budget_s=100_000,
                checkpoint_score_attr="episode_reward_mean",
            )

            for trial in analysis.trials:
                self.assertEqual(trial.status, Trial.TERMINATED)
                self.assertTrue(trial.has_checkpoint())

    @pytest.mark.flaky(max_runs=3, min_passes=1)
    def testPermutationContinuation(self):  # noqa: N802
        # taken from ray's testing suite
        # self.enable_loggers()

        os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "500"  # timeout for save checkpoint debugging
        # suppress warnings about excessive checkpoint syncs
        os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0.25"

        class MockTrainable(tune.Trainable):
            def setup(self, config):
                self.iter = 0
                self.a = config["a"]
                self.b = config["b"]
                self.c = config["c"]

            def step(self):
                time.sleep(1)
                self.iter += 1
                return {"mean_accuracy": (self.a - self.iter) * self.b, "a": self.a, "b": self.b, "c": self.c}

            def save_checkpoint(self, checkpoint_dir):
                checkpoint_path = os.path.join(checkpoint_dir, "model.mock")
                with open(checkpoint_path, "wb") as fp:
                    pickle.dump((self.a, self.b, self.iter), fp)

            def load_checkpoint(self, checkpoint: str):  # pyright: ignore[reportIncompatibleMethodOverride]
                # NOTE: loading a checkpoint changes training_iteration when synch=False
                checkpoint_path = os.path.join(checkpoint, "model.mock")
                with open(checkpoint_path, "rb") as fp:
                    self.a, self.b, self.iter = pickle.load(fp)
                # This resets the training iteration stop criterion to a bad value
                print("Training iteration after reload", self.training_iteration, "@ iter", self.iter)

        scheduler = ReTuneScheduler(
            time_attr="training_iteration",
            metric="mean_accuracy",
            mode="max",
            perturbation_interval=1,
            perturbation_factors=(1, 1),
            log_config=True,
            hyperparam_mutations={"c": lambda: 1},  # c always mutates to 1
            synch=True,
        )

        class MockParam(object):
            def __init__(self, params):
                self._params = params
                self._index = 0

            def __call__(self, *args, **kwargs):
                val = self._params[self._index % len(self._params)]
                self._index += 1
                return val

        param_a = MockParam([10, 20, 30, 40])
        param_b = MockParam([1.2, 0.9, 1.1, 0.8])

        random.seed(100)
        np.random.seed(1000)
        checkpoint_config = CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="training_iteration",
            checkpoint_score_order="min",
            checkpoint_frequency=1,
            checkpoint_at_end=True,
        )
        results = tune.run(
            MockTrainable,
            config={
                "a": tune.sample_from(lambda _: param_a()),
                "b": tune.sample_from(lambda _: param_b()),
                "c": 1,
            },
            fail_fast=True,
            num_samples=4,
            checkpoint_config=checkpoint_config,
            scheduler=scheduler,
            name="testPermutationContinuation",
            stop={"training_iteration": 3},
        )
        # in the end all should have the same hyperparameters of trial_3:
        # Trial 4 (40-3) * 0.8 = 29.6
        # Trial 3 (30-3) * 1.1 = 29.7
        trial3 = results.trials[2]
        assert trial3.checkpoint
        with open(os.path.join(trial3.checkpoint.path, "model.mock"), "rb") as fp:
            trial3_ckpt = pickle.load(fp)
        # assert all configs are equal in the end
        for trial in results.trials[1:]:
            self.assertDictEqual(trial.config, trial3.config)
        for i, trial in enumerate(results.trials):
            if i != 3:
                self.assertEqual(trial.metric_analysis["a"]["min"], param_a._params[i])
            else:
                self.assertEqual(trial.metric_analysis["a"]["max"], param_a._params[i])
            self.assertEqual(trial.metric_analysis["a"]["last"], 30)
            self.assertEqual(trial.metric_analysis["b"]["last"], 1.1)
            assert trial.checkpoint
            with open(os.path.join(trial.checkpoint.path, "model.mock"), "rb") as fp:
                self.assertEqual(trial3_ckpt, pickle.load(fp))
            if i != 2:
                with open(os.path.join(trial.checkpoint.path[:-1] + "0", "model.mock"), "rb") as fp:
                    self.assertNotEqual(trial3_ckpt, pickle.load(fp))


class TestTuneWithTopTrialScheduler(TestHelpers, DisableLoggers, InitRay, num_cpus=4):
    # Some tests taken from ray's testing suite

    @pytest.mark.length(speed="medium")
    @mock.patch("wandb.Api", new=MagicMock())
    @mock.patch("ray_utilities.callbacks.wandb.wandb_api", new=MagicMock())
    def test_run_tune_with_top_trial_scheduler(self):
        original_exploit = TopPBTTrialScheduler._exploit
        perturbation_interval = 100
        best_step_size_idx = 0
        best_value_idx = 1
        batch_sizes = (25, 50, 100)

        num_exploits = 0

        fake_results: dict[int, dict[int, float]] = {
            batch_sizes[0]: {
                v: v // batch_sizes[0] for v in range(batch_sizes[0], 401, batch_sizes[0])
            },  # 1, 2, ..., 8
            # exploit this:
            batch_sizes[1]: {
                v: v // batch_sizes[1] + 20 for v in range(batch_sizes[1], 401, batch_sizes[1])
            },  # 21, 22, ..., 24
            batch_sizes[2]: {v: v // batch_sizes[2] + 5 for v in range(batch_sizes[2], 401, batch_sizes[2])},  # 6, 7
        }
        best_step_size = batch_sizes[1]
        best_step_size_for_step = {}
        for batch_size, values in fake_results.items():
            for step, step_value in values.items():
                if step % perturbation_interval != 0:
                    # not at perturbation interval
                    continue
                if step not in best_step_size_for_step or best_step_size_for_step[step][1] < step_value:
                    best_step_size_for_step[step] = (batch_size, step_value)
        # TODO: Currently 100 is always best, change later to 50 (this would mean) @ 200, the 50 steps is better.
        assert all(
            v > not_best_v
            for v in fake_results[best_step_size].values()
            for not_best_k, not_best_values in fake_results.items()
            if not_best_k != best_step_size
            for not_best_v in not_best_values.values()
        )

        race_conditions = 0

        def test_exploit_function(
            self: TopPBTTrialScheduler, tune_controller: TuneController, trial: Trial, trial_to_clone: Trial
        ) -> None:
            # check that best_step_size is used
            # NOTE: trial.last_result has NOT been updated for the LAST (non-best) trial
            # ALSO: If after perturbation and loading a checkpoint it is also off
            nonlocal num_exploits
            num_exploits += 1
            print("Exploit number", num_exploits, "called at step", self._trial_state[trial].last_train_time)
            if self._trial_state[trial].last_perturbation_time % perturbation_interval != 0:
                # We can end up here due to a race condition as Trial pause reached too slow and another step was taken
                # ignore most of the asserts but check that we did not end up here more than once.
                nonlocal race_conditions
                race_conditions += 1
                logger.warning(
                    "Exploit called at step %s not at perturbation interval for trial. Likely due to race condition.",
                    self._trial_state[trial].last_perturbation_time,
                )
            else:
                current_step = self._trial_state[trial].last_train_time
                # When a trial is loaded, it can make a step
                # assert current_step % perturbation_interval == 0
                assert (
                    self._trial_state[trial].last_score
                    == fake_results[trial.config["train_batch_size_per_learner"]][current_step]
                )
                assert current_step == self._trial_state[trial].last_perturbation_time
                assert (
                    best_step_size_for_step[current_step][best_step_size_idx]
                    == trial_to_clone.config["train_batch_size_per_learner"]
                    == batch_sizes[1]
                )
                # check that value is the expected value
                # trial_to_clone.last_result should already be updated.
                assert (
                    trial_to_clone.last_result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
                    == best_step_size_for_step[current_step][best_value_idx]
                )
                # call original exploit function
            original_exploit(self, tune_controller, trial, trial_to_clone)
            # TODO: Check updated trial
            # Do not perturb to best trials
            assert trial.config["train_batch_size_per_learner"] != batch_sizes[1]

        TopPBTTrialScheduler._exploit = test_exploit_function

        class CheckTrainableForTop(TrainableWithChecks):
            debug_step = False
            use_pbar = False

            def step(self) -> LogMetricsDict:
                self._current_step += self.algorithm_config.train_batch_size_per_learner
                result = {ENV_RUNNER_RESULTS: {}, EVALUATION_RESULTS: {ENV_RUNNER_RESULTS: {}}}
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME] = self._current_step
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER] = (
                    self.algorithm_config.train_batch_size_per_learner
                )
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME] = self._current_step + 2
                result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] = fake_results[
                    self.algorithm_config.train_batch_size_per_learner
                ][self._current_step]
                result["_checking_class_"] = "CheckTrainableForTop"  # pyright: ignore[reportGeneralTypeIssues]
                logger.info(
                    "Batch size: %s, step %s, result: %s",
                    self.algorithm_config.train_batch_size_per_learner,
                    self._current_step,
                    result,
                )
                result["current_step"] = self._current_step
                if is_pbar(self._pbar):
                    self._pbar.update(1)
                    self._pbar.set_description(
                        f"Step: {self._current_step} batch_size={self.algorithm_config.train_batch_size_per_learner} "
                        f"result={result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}"
                    )
                time.sleep(2)  # Avoid race conditions by pause being too slow.
                return result  # pyright: ignore[reportReturnType]

        ray_pbt_logger.setLevel(logging.DEBUG)
        with patch_args(
            # main args for this experiment
            "--tune", "batch_size",
            # Meta / less influential arguments for the experiment.
            # NOTE: Num samples is multiplides by grid_search values. So effectively num_samples=3
            "--num_samples", 1,
            "--num_jobs", 3,
            "--max_step_size", max(batch_sizes) * 2,
            "--min_step_size", min(batch_sizes),
            "--minibatch_size", min(batch_sizes),
            "--env_seeding_strategy", "same",
            # constant
            "--seed", "42",
            "--log_level", "DEBUG",
            "--log_stats", "most",
            "--total_steps", max(batch_sizes) * 3,
            "--use_exact_total_steps",
            "--no_dynamic_eval_interval",
            "--fcnet_hiddens", "[4]",
            "--num_envs_per_env_runner", 5,
            "--test",
            "pbt",
            "--quantile_fraction", "0.1",
            "--perturbation_interval", perturbation_interval,
        ):  # fmt: skip
            Setup = SetupWithCheck(CheckTrainableForTop, PPOMLPWithPBTSetup)
            setup = Setup(
                config_files=["experiments/models/mlp/default.cfg"],
                # TODO: Trials are reused, trial name might be wrong then
            )
            setup.param_space["train_batch_size_per_learner"] = tune.grid_search(batch_sizes)
            assert setup.args.command
            setup.args.command.set_hyperparam_mutations(
                {
                    "train_batch_size_per_learner": CyclicMutation(batch_sizes),
                    "fcnet_hiddens": KeepMutation([2]),
                }
            )
            results = run_tune(setup)
            print("Num exploits:", num_exploits)
            raise_tune_errors(results)  # pyright: ignore[reportArgumentType]
            self.assertTrue(all(result.metrics["_checking_class_"] == "CheckTrainableForTop" for result in results))  # pyright: ignore[reportOptionalSubscript, reportAttributeAccessIssue]
            # At final step (if batch_size % perturbation_interval) there is no exploitation
            # There should be (3-1) x 2 = 4 exploitations
            self.assertEqual(num_exploits, max(batch_sizes) * (3 - 1) // perturbation_interval * 2)
            # Check that at most one race condition happened
            self.assertLessEqual(race_conditions, 1)
            self.assertTrue(all(r.config["fcnet_hiddens"] == [2] for r in results))  # pyright: ignore[reportAttributeAccessIssue, reportOptionalSubscript]


class DummyTrial:
    def __init__(self, trial_id, config=None, *, finished=False):
        self.trial_id = trial_id
        self._finished = finished
        self.config = config if config is not None else {}

    def is_finished(self):
        return self._finished

    def __repr__(self):
        return f"DummyTrial('{self.trial_id}')"


class DummyState:
    def __init__(self, last_score):
        self.last_score = last_score

    def __repr__(self):
        return f"DummyState(last_score={self.last_score})"


class PBTQuantileNaNTest(unittest.TestCase):
    def test_nan_last_score_in_quantiles(self):
        # Create three trials: one with nan, two with valid scores
        # Patch _trial_state with dummy states

        # To test PB2 and PopulationBasedTraining in ray need https://github.com/ray-project/ray/pull/57160
        for scheduler_class in [ReTuneScheduler, TopPBTTrialScheduler]:
            t1 = DummyTrial("t1", config=MagicMock())
            t2 = DummyTrial("t2", config=MagicMock())
            t3 = DummyTrial("t3", config=MagicMock())
            max_states: dict[Any, Any] = {
                t1: DummyState(last_score=20.0),  # best
                t2: DummyState(last_score=float("nan")),
                t3: DummyState(last_score=10.0),  # worst
            }
            min_states: dict[Any, Any] = {
                t3: DummyState(last_score=10.0),  # best
                t2: DummyState(last_score=float("nan")),
                t1: DummyState(last_score=20.0),  # worst
            }
            with self.subTest(scheduler_class=scheduler_class.__name__):
                hp_kwargs: dict[str, Any] = {
                    "hyperparam_mutations" if scheduler_class is not PB2 else "hyperparam_bounds": {"lr": [1e-4, 1e-3]}
                }
                # test max mode
                max_scheduler = scheduler_class(
                    metric="reward",
                    mode="max",
                    quantile_fraction=0.51 if scheduler_class is ReTuneScheduler else 0.5,
                    **hp_kwargs,
                )
                max_scheduler._trial_state = max_states
                for t, state in max_states.items():
                    max_scheduler._save_trial_state(
                        state, 100, {"reward": state.last_score, "time_total_s": 1, "training_iteration": 1}, t
                    )

                # Should not raise, but nan disrupts sorting
                max_bottom, max_top = max_scheduler._quantiles()
                max_other_trials = [t for t in max_scheduler._trial_state if t not in max_bottom + max_top]
                max_ordered_results = [
                    max_scheduler._trial_state[t].last_score for t in [*max_bottom, *max_other_trials, *max_top]
                ]

                self.assertIn(t1, max_top)
                self.assertIn(t2, max_other_trials)
                self.assertIn(t3, max_bottom)
                self.assertEqual(max_ordered_results[-1], 20)

                # Test min mode
                min_scheduler = scheduler_class(
                    metric="reward",
                    mode="min",
                    quantile_fraction=0.51 if scheduler_class is ReTuneScheduler else 0.5,
                    **hp_kwargs,
                )
                min_scheduler._trial_state = min_states
                for t, state in min_states.items():
                    min_scheduler._save_trial_state(
                        state, 100, {"reward": state.last_score, "time_total_s": 1, "training_iteration": 1}, t
                    )
                min_bottom, min_top = min_scheduler._quantiles()
                min_other_trials = [t for t in min_scheduler._trial_state if t not in min_bottom + min_top]
                min_ordered_results = [
                    min_scheduler._trial_state[t].last_score for t in [*min_bottom, *min_other_trials, *min_top]
                ]

                self.assertIn(t1, min_bottom)
                self.assertIn(t2, min_other_trials)
                self.assertIn(t3, min_top)
                self.assertEqual(abs(min_ordered_results[-1]), 10)


class MockPBTTrialState:
    def __init__(
        self,
        last_score,
        last_checkpoint=None,
        last_perturbation_time=0,
        last_train_time=900,
        last_result=None,
        last_training_iteration=1,
        current_env_steps=900,
        last_update_timestamp=None,
    ):
        self.last_score = last_score
        self.last_checkpoint = last_checkpoint
        self.last_perturbation_time = last_perturbation_time
        self.last_train_time = last_train_time
        self.last_result = last_result if last_result is not None else {"reward": last_score, TRAINING_ITERATION: 1}
        self.last_training_iteration = last_training_iteration
        self.current_env_steps = current_env_steps
        self.last_update_timestamp = last_update_timestamp


class TestTopTrialSchedulerSlowTrials(DisableLoggers, TestHelpers):
    """Tests for handling slow and bad performing trials in synchronous PBT mode."""

    def setUp(self):
        """Set up test fixtures for slow trial tests."""
        super().setUp()
        self.scheduler = TopPBTTrialScheduler(
            metric="reward",
            mode="max",
            perturbation_interval=1000,
            burn_in_period=100,
            hyperparam_mutations={
                "lr": {"grid_search": [0.001, 0.01, 0.1]},
            },
            quantile_fraction=0.25,
            synch=True,  # Enable synchronous mode for slow trial handling
        )

        # Create mock TuneController
        self.mock_controller = MagicMock(spec=TuneController)
        self.mock_controller.get_trials.return_value = []
        self.mock_controller.get_live_trials.return_value = []
        self.mock_controller._queued_trial_decisions = {}

        # Create 20 trials with varying performance, need 20 for 5% hurdle
        self.trials = []
        current_time = time.time()
        for i in range(20):
            trial = MagicMock(spec=Trial)
            trial.trial_id = f"trial_{i}"
            trial.is_finished.return_value = False
            trial.status = Trial.RUNNING
            trial.config = {"lr": 0.001}

            state = MockPBTTrialState(
                last_score=50 + i * 5,
                last_checkpoint=None,
                last_perturbation_time=0,
                last_train_time=900,
                last_result={"reward": 50 + i * 5, TRAINING_ITERATION: 1},
                last_training_iteration=1,
                current_env_steps=900,
                last_update_timestamp=current_time,
            )

            self.trials.append(trial)
            trial.run_metadata = MagicMock()
            trial.run_metadata.checkpoint_manager.checkpoint_config.num_to_keep = 4
            trial.experiment_tag = f"testing_{i}"
            trial.local_experiment_path = "./outputs/experiments/TESTING"
            self.scheduler.on_trial_add(self.mock_controller, trial)
            self.scheduler._trial_state[trial] = cast("_PBTTrialState2", state)

        self.scheduler._next_perturbation_sync = 1000

    def test_slow_trial_not_in_early_perturbation_window(self):
        """Test that trials outside the early perturbation window (66%-95%) are not paused."""
        # Trial at 50% of interval - too early
        trial = self.trials[0]
        state = self.scheduler._trial_state[trial]
        state.last_train_time = 500
        state.last_perturbation_time = 0

        result = {
            "reward": 50,
            self.scheduler._time_attr: 500,
            TRAINING_ITERATION: 1,
        }

        self.mock_controller.get_live_trials.return_value = self.trials

        decision = self.scheduler.on_trial_result(self.mock_controller, trial, result)

        # Should continue, not pause
        self.assertEqual(decision, self.scheduler.CONTINUE)

    def test_slow_trial_too_late_for_early_perturbation(self):
        """Test that trials past 95% of interval are not paused early."""
        # Trial at 96% of interval - too late
        trial = self.trials[0]
        state = self.scheduler._trial_state[trial]
        state.last_train_time = 960
        state.last_perturbation_time = 0

        result = {
            "reward": 50,
            self.scheduler._time_attr: 960,
            TRAINING_ITERATION: 1,
        }

        self.mock_controller.get_live_trials.return_value = self.trials

        decision = self.scheduler.on_trial_result(self.mock_controller, trial, result)

        # Should continue, not pause
        self.assertEqual(decision, self.scheduler.CONTINUE)

    def test_slow_trial_within_time_threshold_not_paused(self):
        """Test that slow trials within 5 min of last update are not paused."""
        trial = self.trials[0]
        state = self.scheduler._trial_state[trial]
        state.last_train_time = 700
        state.last_perturbation_time = 0
        # Set recent update timestamp (within 5 min)
        current_time = time.time()
        state.last_update_timestamp = current_time - 100

        # Make all other trials have recent updates too
        for t in self.trials:
            self.scheduler._trial_state[t].last_update_timestamp = current_time - 50

        result = {
            "reward": 50,
            self.scheduler._time_attr: 700,
            TRAINING_ITERATION: 1,
        }

        self.mock_controller.get_live_trials.return_value = self.trials

        decision = self.scheduler.on_trial_result(self.mock_controller, trial, result)

        # Should continue because time threshold not met
        self.assertEqual(decision, self.scheduler.CONTINUE)

    def test_slow_trial_not_in_bottom_5_percent_continues(self):
        """Test that trials not in the slowest 5% continue normally."""
        # Set up: 8 trials finished, 2 still active (not in bottom 5% of 10)
        for i, trial in enumerate(self.trials[:-2]):
            self.scheduler._trial_state[trial].last_train_time = 1000

        slow_trial = self.trials[-2]
        state = self.scheduler._trial_state[slow_trial]
        state.last_train_time = 700
        state.last_perturbation_time = 0
        current_time = time.time()
        state.last_update_timestamp = current_time - 400  # Old enough

        result = {
            "reward": 50,
            self.scheduler._time_attr: 700,
            TRAINING_ITERATION: 1,
        }

        self.mock_controller.get_live_trials.return_value = self.trials

        decision = self.scheduler.on_trial_result(self.mock_controller, slow_trial, result)

        # Should continue because 2/10 = 20% > 5%
        self.assertEqual(decision, self.scheduler.CONTINUE)

    def test_slow_trial_good_performance_continues(self):
        """Test that slow trials with good performance (not in bottom 33%) continue."""
        # Set up: all other trials finished
        current_time = time.time()
        for trial in self.trials[1:]:
            self.scheduler._trial_state[trial].last_train_time = 1000
            self.scheduler._trial_state[trial].last_update_timestamp = current_time - 1000

        slow_trial = self.trials[0]
        state = self.scheduler._trial_state[slow_trial]
        state.last_train_time = 700
        state.last_perturbation_time = 0
        state.last_score = 80  # Good score, in top 33%
        state.last_update_timestamp = current_time - 400

        # Set up lowest_scores so 80 is not in bottom 33%
        for i, t in enumerate(self.trials):
            self.scheduler._trial_state[t].last_score = 50 + i * 5

        result = {
            "reward": 80,
            self.scheduler._time_attr: 700,
            TRAINING_ITERATION: 1,
        }

        self.mock_controller.get_live_trials.return_value = self.trials

        decision = self.scheduler.on_trial_result(self.mock_controller, slow_trial, result)

        # Should continue because performance is good
        self.assertEqual(decision, self.scheduler.CONTINUE)

    def test_slow_bad_trial_is_last_triggers_perturbation(self):
        """Test that the last slow, bad trial triggers early perturbation."""
        # All trials finished except the slow, bad one
        current_time = time.time()
        for trial in self.trials[1:]:
            self.scheduler._trial_state[trial].last_train_time = 1000
            # Set timestamps more than 300 seconds ago
            self.scheduler._trial_state[trial].last_update_timestamp = current_time - 1000

        slow_trial = self.trials[0]
        state = self.scheduler._trial_state[slow_trial]
        state.last_train_time = 700
        state.last_perturbation_time = 0
        state.last_score = 52  # Bad score, in bottom 33%
        # Slow trial timestamp should be more than 300 seconds behind the max
        state.last_update_timestamp = current_time - 400

        result = {
            "reward": 52,
            self.scheduler._time_attr: 700,
            TRAINING_ITERATION: 1,
            "time_total_s": 700,  # Add required field for parent class
        }

        self.mock_controller.get_live_trials.return_value = self.trials
        self.mock_controller.get_trials.return_value = self.trials
        # Mock parent class to return CONTINUE so our early perturbation logic runs
        with (
            patch.object(PopulationBasedTraining, "on_trial_result", return_value=TopPBTTrialScheduler.CONTINUE),
            patch.object(self.scheduler, "_perturbation_sync_mode") as mock_perturb,
        ):
            _decision = self.scheduler.on_trial_result(self.mock_controller, slow_trial, result)

            # Should trigger perturbation
            mock_perturb.assert_called_once()
            # Verify perturbation was called with correct time
            call_args = mock_perturb.call_args[0]
            self.assertEqual(call_args[0], self.mock_controller)
            self.assertEqual(call_args[1], 1000)  # max of all last_train_times

    def test_slow_bad_trial_not_last_is_paused(self):
        """Test that slow, bad trials that are not the last one are paused."""
        # 2 slow trials, rest finished
        current_time = time.time()
        for trial in self.trials:
            self.scheduler._trial_state[trial].last_train_time = 1000
            self.scheduler._trial_state[trial].last_update_timestamp = current_time - 1000

        slow_trial_1 = self.trials[0]
        slow_trial_2 = self.trials[1]

        # Slow trials have not reported yet
        for slow_trial in [slow_trial_1, slow_trial_2]:
            state = self.scheduler._trial_state[slow_trial]
            state.last_train_time = 500
            state.last_perturbation_time = 0
            state.last_score = 52
            state.last_update_timestamp = current_time - 1400

        result = {
            "reward": 52,
            self.scheduler._time_attr: 700,
            TRAINING_ITERATION: 1,
        }

        self.mock_controller.get_live_trials.return_value = self.trials
        self.mock_controller.get_trials.return_value = self.trials

        with patch.object(self.scheduler, "_perturbation_sync_mode") as mock_perturb:
            decision = self.scheduler.on_trial_result(self.mock_controller, slow_trial_1, result)

            # Should pause and not trigger perturbation (still waiting for slow_trial_2)
            self.assertEqual(decision, self.scheduler.PAUSE)
            mock_perturb.assert_not_called()
            self.scheduler._trial_state[slow_trial_1].last_perturbation_time = 1000
            slow_trial_1.status = Trial.PAUSED if decision in (self.scheduler.NOOP, self.scheduler.PAUSE) else decision
            self.assertEqual(slow_trial_1.status, Trial.PAUSED)
        # As we pause trial1 we update the max last_update_timestamp, trial2 will therefore continue
        # With slow trial 2 we get perturbation
        with patch.object(self.scheduler, "_perturbation_sync_mode") as mock_perturb:
            decision = self.scheduler.on_trial_result(self.mock_controller, slow_trial_2, result)

            self.assertEqual(decision, self.scheduler.PAUSE)
            mock_perturb.assert_called()

    def test_slow_trial_already_paused_returns_noop(self):
        """Test that already paused slow trials return NOOP instead of PAUSE."""
        # All trials finished except the slow one
        current_time = time.time()
        for trial in self.trials[1:]:
            self.scheduler._trial_state[trial].last_train_time = 1000
            self.scheduler._trial_state[trial].last_update_timestamp = current_time - 1000

        slow_trial = self.trials[0]
        slow_trial.status = Trial.PAUSED
        state = self.scheduler._trial_state[slow_trial]
        state.last_train_time = 700
        state.last_perturbation_time = 0
        state.last_score = 52
        state.last_update_timestamp = current_time - 400

        result = {
            "reward": 52,
            self.scheduler._time_attr: 700,
            TRAINING_ITERATION: 1,
        }

        self.mock_controller.get_live_trials.return_value = self.trials
        self.mock_controller.get_trials.return_value = self.trials

        # Multiple slow trials, so won't trigger perturbation
        self.scheduler._trial_state[self.trials[1]].last_train_time = 700
        self.scheduler._trial_state[self.trials[1]].last_update_timestamp = current_time - 400

        decision = self.scheduler.on_trial_result(self.mock_controller, slow_trial, result)

        # Should return NOOP for already paused trial
        self.assertEqual(decision, self.scheduler.NOOP)

    def test_slow_trial_state_saved_correctly(self):
        """Test that slow trial state is saved at perturbation time."""
        # All trials finished except the slow one
        current_time = time.time()
        for trial in self.trials[1:]:
            self.scheduler._trial_state[trial].last_train_time = 1000
            self.scheduler._trial_state[trial].last_update_timestamp = current_time - 1000

        slow_trial = self.trials[0]
        state = self.scheduler._trial_state[slow_trial]
        state.last_train_time = 700
        state.last_perturbation_time = 0
        state.last_score = 52
        state.last_update_timestamp = current_time - 1400

        result = {
            "reward": 52,
            self.scheduler._time_attr: 700,
            TRAINING_ITERATION: 5,
            "current_step": 12345,
        }

        self.mock_controller.get_live_trials.return_value = self.trials
        self.mock_controller.get_trials.return_value = self.trials
        slow_trial.run_metadata = MagicMock()

        with patch.object(self.scheduler, "_perturbation_sync_mode"):
            self.scheduler.on_trial_result(self.mock_controller, slow_trial, result)

            # Verify state was saved with correct values
            self.assertEqual(state.last_training_iteration, 5)
            self.assertEqual(state.current_env_steps, 12345)
            self.assertIsNotNone(state.last_update_timestamp)

    def test_slow_trial_non_synch_mode_no_early_perturbation(self):
        """Test that non-synchronous mode doesn't trigger early perturbation."""
        # Create scheduler in non-synch mode
        scheduler = TopPBTTrialScheduler(
            metric="reward",
            mode="max",
            perturbation_interval=1000,
            hyperparam_mutations={"lr": {"grid_search": [0.001, 0.01]}},
            synch=False,  # Non-synchronous mode
        )

        trial = self.trials[0]
        state = cast("_PBTTrialState2", MagicMock())
        state.last_train_time = 700
        state.last_perturbation_time = 0
        state.last_score = 52
        current_time = time.time()
        state.last_update_timestamp = current_time - 400

        scheduler._trial_state[trial] = state
        scheduler._next_perturbation_sync = 1000

        result = {
            "reward": 52,
            scheduler._time_attr: 700,
            TRAINING_ITERATION: 1,
        }

        with patch.object(scheduler, "_perturbation_sync_mode") as mock_perturb:
            # Mock the parent class to return CONTINUE
            # with patch.object(PopulationBasedTraining, "on_trial_result", return_value=scheduler.CONTINUE):
            decision = scheduler.on_trial_result(self.mock_controller, trial, result)

            # Should continue normally, no early perturbation in non-synch mode
            self.assertEqual(decision, scheduler.CONTINUE)
            mock_perturb.assert_not_called()

    def test_slow_trial_before_burn_in_continues(self):
        """Test that slow trials before burn-in period are not paused."""
        trial = self.trials[0]
        state = self.scheduler._trial_state[trial]
        state.last_train_time = 50
        state.last_perturbation_time = 0

        result = {
            "reward": 50,
            self.scheduler._time_attr: 50,  # Before burn_in_period of 100
            TRAINING_ITERATION: 1,
        }

        self.mock_controller.get_live_trials.return_value = self.trials

        decision = self.scheduler.on_trial_result(self.mock_controller, trial, result)

        # Should continue because still in burn-in period
        self.assertEqual(decision, self.scheduler.CONTINUE)


if __name__ == "__main__":
    unittest.main()
