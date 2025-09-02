from __future__ import annotations

import logging
import os
import pickle
import random
import tempfile
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

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
from ray.tune.experiment import Trial
from ray.tune.result import SHOULD_CHECKPOINT, TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import CombinedStopper
from ray.tune.stopper.maximum_iteration import MaximumIterationStopper
from ray.tune.utils.mock_trainable import MOCK_TRAINABLE_NAME, register_mock_trainable  # noqa: PLC0415

from ray_utilities.callbacks.algorithm import exact_sampling_callback
from ray_utilities.callbacks.tuner.metric_checkpointer import StepCheckpointer  # pyright: ignore[reportDeprecated]
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.constants import (
    EVAL_METRIC_RETURN_MEAN,
    NUM_ENV_STEPS_PASSED_TO_LEARNER,
    NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
)
from ray_utilities.misc import raise_tune_errors
from ray_utilities.runfiles import run_tune
from ray_utilities.setup import optuna_setup
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.optuna_setup import OptunaSearchWithPruner
from ray_utilities.testing_utils import (
    ENV_RUNNER_CASES,
    Cases,
    DisableLoggers,
    InitRay,
    SetupWithCheck,
    TestHelpers,
    TrainableWithChecks,
    _MockTrial,
    _MockTrialRunner,
    format_result_errors,
    iter_cases,
    mock_result,
    patch_args,
)
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.helpers import make_divisible
from ray_utilities.tune.scheduler.re_tune_scheduler import ReTuneScheduler

if TYPE_CHECKING:
    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData
    from ray_utilities.typing.metrics import LogMetricsDict
    from ray_utilities.typing.trainable_return import TrainableReturnData

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
MINIBATCH_SIZE = 32


@pytest.mark.tuner
class TestTuner(InitRay, TestHelpers, DisableLoggers, num_cpus=4):
    def test_optuna_search_added(self):
        with patch_args("--optimize_config", "--num_samples", "1"):
            optuna_setup = AlgorithmSetup()
            self.assertTrue(optuna_setup.args.optimize_config)
            tuner = optuna_setup.create_tuner()
            assert tuner._local_tuner and tuner._local_tuner._tune_config
            self.assertIsInstance(tuner._local_tuner._tune_config.search_alg, OptunaSearch)
            # verify metrics key
            assert tuner._local_tuner._tune_config.search_alg
            self.assertEqual(tuner._local_tuner._tune_config.search_alg.metric, EVAL_METRIC_RETURN_MEAN)
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

            self.check_stopper_added(tuner, MaximumIterationStopper, check=check_stopper)

    def test_run_tune_function(self):
        batch_size = make_divisible(BATCH_SIZE, DefaultArgumentParser.num_envs_per_env_runner)
        with patch_args("--num_samples", "3", "--num_jobs", "3", "--batch_size", batch_size, "--iterations", "3"):
            with AlgorithmSetup(init_trainable=False) as setup:
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
        ):  # fmt: skip
            # FIXME: new default steps does not align with other tests!
            setup = AlgorithmSetup()
            # Workaround for NOT working StepCheckpointer as it works on a copy of the result dict.
            # AlgoStepCheckpointer is not added, hardcoded checkpointing!
            # self.is_algorithm_callback_added(
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

        checkpoint_now_implemented = None

        class CheckpointCallback(Callback):
            num_checkpoints = 0

            def on_trial_result(self, iteration, trials, trial: Trial, result, **info):
                nonlocal checkpoint_now_implemented
                # Checkpoint every two iterations
                if result[TRAINING_ITERATION] % 2 == 0:
                    self.num_checkpoints += 1
                    try:
                        trial.checkpoint_now()
                    except AttributeError:
                        checkpoint_now_implemented = False
                    else:
                        checkpoint_now_implemented = True

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
            # assert checkpoint_now_implemented  # custom patch
            if checkpoint_now_implemented:
                assert trial.storage
                assert sum(item.startswith("checkpoint_") for item in os.listdir(trial.storage.trial_fs_path)) == 3


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
        ):  # fmt: skip
            setup = AlgorithmSetup()
        tuner = setup.create_tuner()
        tuner._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(  # pyright: ignore[reportOptionalMemberAccess]
            checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
            checkpoint_score_order="max",
            checkpoint_frequency=2,  # Save every two iterations
            # NOTE: num_keep does not appear to work here
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
        ):  # fmt: skip

            class CheckpointSetup(AlgorithmSetup):
                def _create_trainable(self):
                    class DefaultTrainableWithCheckpoint(DefaultTrainable):
                        def step(self):
                            result = super().step()
                            result[SHOULD_CHECKPOINT] = True
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
                with patch_args(
                    "--num_samples", "1",
                    "--num_jobs", "1",
                    "--batch_size", batch_size,  # overwrite
                    "--use_exact_total_steps",  # do not adjust total_steps
                    "--minibatch_size", MINIBATCH_SIZE,  # keep
                    "--iterations", "1",  # overwrite
                ):  # fmt: skip
                    with AlgorithmSetup() as setup1:
                        setup1.config.env_runners(num_env_runners=num_env_runners)
                tuner1 = setup1.create_tuner()
                tuner1._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(  # pyright: ignore[reportOptionalMemberAccess]
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
                self.compare_configs(
                    trainable2_local.algorithm_config.to_dict(),
                    setup2.config.to_dict(),
                )
                self.assertEqual(trainable2_local.algorithm_config.train_batch_size_per_learner, batch_size * 2)
                trainable2_local.stop()

                tuner2 = setup2.create_tuner()
                tuner2._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(  # pyright: ignore[reportOptionalMemberAccess]
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
                with patch_args(
                    "--num_samples", "1",
                    "--num_jobs", 1,
                    "--batch_size", batch_size,  # overwrite
                    "--minibatch_size", MINIBATCH_SIZE,  # keep
                    "--iterations", "1",  # overwrite
                ):  # fmt: skip
                    with AlgorithmSetup() as setup1:
                        setup1.config.env_runners(num_env_runners=num_env_runners)
                tuner1 = setup1.create_tuner()
                tuner1._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(  # pyright: ignore[reportOptionalMemberAccess]
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
                    "--tune", "batch_size", "rollout_size",
                    "--iterations", "3",
                ):  # fmt: skip
                    Setup.batch_size_sample_space = {"grid_search": SAMPLE_SPACE}
                    Setup.rollout_size_sample_space = {"grid_search": SAMPLE_SPACE}
                    with Setup() as setup2:
                        setup2.config.env_runners(num_env_runners=num_env_runners)
                assert setup2.config.num_envs_per_env_runner is not None
                self.assertTrue(all(s % setup2.config.num_envs_per_env_runner == 0 for s in SAMPLE_SPACE))
                tuner2 = setup2.create_tuner()
                tuner2._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(  # pyright: ignore[reportOptionalMemberAccess]
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
                rollout_sizes = set()

                checkpoints = {}
                for result in results2:
                    assert result.config
                    batch_sizes.add(result.config["train_batch_size_per_learner"])
                    rollout_sizes.add(result.config["rollout_size"])
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
        with patch_args("--optimize_config", "--num_samples", "1"):
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
    def test_pruning(self):
        """
        Test might fail due to bad luck, low numbers first then high.

        Note:
            Remember Optuna might not prune the first 10 trials.
            Reduce num_jobs or adjust seed and test again.
        """
        MAX_STEP = 15
        with patch_args("--optimize_config", "--num_samples", "15", "--num_jobs", "4", "--seed", "42"):

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
                        "module": "OptunaTest",
                        "env": "CartPole-v1",
                    }

                def _create_trainable(self):  # type: ignore[override]
                    return trainable

            with RandomParamsSetup() as setup:
                setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=64)
                setup.config.evaluation(evaluation_interval=1)

            with self.assertLogs(optuna_setup._logger, level="INFO") as log:
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
            # NOTE: This can be OK even if runs fail!


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
            trial = _MockTrial(i, trial_config, self.storage)
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

    def testPermutationContinuation(self):  # noqa: N802
        # taken from ray's testing suite
        # self.enable_loggers()

        os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "500"  # timeout for save checkpoint

        class MockTrainable(tune.Trainable):
            def setup(self, config):
                self.iter = 0
                self.a = config["a"]
                self.b = config["b"]
                self.c = config["c"]

            def step(self):
                self.iter += 1
                return {"mean_accuracy": (self.a - self.iter) * self.b, "a": self.a, "b": self.b, "c": self.c}

            def save_checkpoint(self, checkpoint_dir):
                # breakpoint()
                # remote_breakpoint()
                checkpoint_path = os.path.join(checkpoint_dir, "model.mock")
                with open(checkpoint_path, "wb") as fp:
                    pickle.dump((self.a, self.b, self.iter), fp)

            def load_checkpoint(self, checkpoint: str):  # pyright: ignore[reportIncompatibleMethodOverride]
                # NOTE: loading a checkpoint changes training_iteration when synch=False
                # breakpoint()
                # remote_breakpoint()
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
        trial2 = results.trials[2]
        assert trial2.checkpoint
        with open(os.path.join(trial2.checkpoint.path, "model.mock"), "rb") as fp:
            trial2_ckpt = pickle.load(fp)
        for trial in results.trials[1:]:
            self.assertDictEqual(trial.config, trial2.config)
        for i, trial in enumerate(results.trials):
            if i != 3:
                self.assertEqual(trial.metric_analysis["a"]["min"], param_a._params[i])
            else:
                self.assertEqual(trial.metric_analysis["a"]["max"], param_a._params[i])
            self.assertEqual(trial.metric_analysis["a"]["last"], 30)
            self.assertEqual(trial.metric_analysis["b"]["last"], 1.1)
            assert trial.checkpoint
            with open(os.path.join(trial.checkpoint.path, "model.mock"), "rb") as fp:
                self.assertEqual(trial2_ckpt, pickle.load(fp))
            if i != 2:
                with open(os.path.join(trial.checkpoint.path[:-1] + "0", "model.mock"), "rb") as fp:
                    self.assertNotEqual(trial2_ckpt, pickle.load(fp))
