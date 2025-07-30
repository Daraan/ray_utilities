import logging
import os
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from unittest import skip

from ray import tune
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.tune.result import SHOULD_CHECKPOINT, TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import CombinedStopper
from ray.tune.stopper.maximum_iteration import MaximumIterationStopper

from ray_utilities.callbacks.algorithm import exact_sampling_callback
from ray_utilities.constants import (
    EVAL_METRIC_RETURN_MEAN,
    NUM_ENV_STEPS_PASSED_TO_LEARNER,
    NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
)
from ray_utilities.runfiles import run_tune
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.optuna_setup import OptunaSearchWithPruner
from ray_utilities.testing_utils import (
    DisableGUIBreakpoints,
    DisableLoggers,
    InitRay,
    SetupDefaults,
    TestHelpers,
    format_result_errors,
    patch_args,
)
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.functional import training_step
from ray_utilities.typing.trainable_return import TrainableReturnData

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
MINIBATCH_SIZE = 32

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig  # noqa: F401


class TestTuner(InitRay, TestHelpers, DisableLoggers):
    def test_tuner_setup(self):
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

    def test_run_tune_function(self):
        with patch_args("--num_samples", "3", "--num_jobs", "2", "--batch_size", BATCH_SIZE, "--iterations", "3"):
            with AlgorithmSetup(init_trainable=False) as setup:
                setup.config.training(num_epochs=2, minibatch_size=BATCH_SIZE)
                setup.config.evaluation(evaluation_interval=1)  # else eval metric not in dict
            results = run_tune(setup)
            assert not isinstance(results, dict)
            # NOTE: This can be OK even if runs fail!
            for result in results:
                assert result.metrics
                self.assertEqual(result.metrics["current_step"], 3 * BATCH_SIZE)
                self.assertEqual(result.metrics[TRAINING_ITERATION], 3)
            self.assertEqual(results.num_errors, 0, "Encountered errors: " + format_result_errors(results.errors))


class TestTunerCheckpointing(InitRay, TestHelpers, DisableLoggers):
    def test_checkpoint_auto(self):
        # self.enable_loggers()
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size", BATCH_SIZE,
            "--minibatch_size", MINIBATCH_SIZE,
            "--iterations", "4",
        ):  # fmt: off
            setup = AlgorithmSetup()
        tuner = setup.create_tuner()
        tuner._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(  # pyright: ignore[reportOptionalMemberAccess]
            checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
            checkpoint_score_order="max",
            checkpoint_frequency=2,  # Save every two iterations
            # NOTE: num_keep does not appear to work here
        )
        results = tuner.fit()
        self.assertEqual(results.num_errors, 0, "Encountered errors: " + format_result_errors(results.errors))  # pyright: ignore[reportAttributeAccessIssue,reportOptionalSubscript]
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
        ):  # fmt: off

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

    @skip("This test is not implemented yet")
    def test_checkpoint_standard(self): ...

    @skip("This test is not implemented yet")
    def test_checkpoint_and_load(self): ...


class TestReTuning(
    InitRay,
    TestHelpers,
    DisableLoggers,
):
    def test_retune_with_different_config(self):
        self.enable_loggers()
        NUM_ITERS_2 = 3
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size", BATCH_SIZE,  # overwrite
            "--minibatch_size", MINIBATCH_SIZE, # keep
            "--iterations", "1",  # overwrite
        ):  # fmt: off
            with AlgorithmSetup() as setup1:
                setup1.config.env_runners(num_env_runners=0)
        tuner1 = setup1.create_tuner()
        tuner1._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(  # pyright: ignore[reportOptionalMemberAccess]
            checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
            checkpoint_score_order="max",
            checkpoint_frequency=1,
        )
        results1 = tuner1.fit()
        self.assertEqual(results1.num_errors, 0, "Encountered errors: " + format_result_errors((results1.errors)))  # pyright: ignore[reportAttributeAccessIssue,reportOptionalSubscript]
        # Check metrics:
        result1 = results1[0]
        assert result1.metrics
        self.assertEqual(result1.metrics[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME], BATCH_SIZE)  # pyright: ignore[reportOptionalSubscript]
        self.assertEqual(result1.metrics[TRAINING_ITERATION], 1)  # pyright: ignore[reportOptionalSubscript]
        checkpoint_dir, checkpoints = self.get_checkpoint_dirs(results1[0])
        self.assertEqual(
            len(checkpoints),
            1,
            f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir)} in {checkpoint_dir}",
        )
        self.assertTrue(os.path.exists(checkpoints[0]), "Checkpoint file does not exist: " + checkpoints[0])

        class TrainableWithChecks(DefaultTrainable[Any, "AlgorithmConfig", Any]):
            def setup(self, config, *, algorithm_overrides=None):
                super().setup(config, algorithm_overrides=algorithm_overrides)
                assert self._iteration == 1, "Trainable should be setup with iteration 1"

            def step(self):
                assert self.algorithm_config.train_batch_size_per_learner == BATCH_SIZE * 2, (
                    f"Batch size should be 2x the original batch size, not {self.algorithm_config.train_batch_size_per_learner}"
                )
                # print("starting debugpy")
                # import debugpy
                # debugpy.listen(5678)
                # debugpy.wait_for_client()
                # breakpoint()
                result, metrics, rewards = training_step(
                    self.algorithm,
                    reward_updaters=self._reward_updaters,
                    discrete_eval=self.discrete_eval,
                    disable_report=True,
                    log_stats=self.log_stats,
                )
                assert result["training_iteration"] >= 2, (
                    "Should start with at least 1 iteration. Should now be at least 2"
                )
                expected = 2 * BATCH_SIZE
                assert (value := result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED]) == expected, (
                    f"Expected {expected} env steps sampled, got {value}"
                )
                assert (value := result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER]) == expected, (
                    f"Expected {expected} env steps passed to learner, got {value}"
                )
                metrics["_checking_class_"] = True  # pyright: ignore[reportGeneralTypeIssues]

                return metrics

        class Setup(AlgorithmSetup):
            def _create_trainable(self):
                return TrainableWithChecks.define(self)

        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size",
            BATCH_SIZE * 2,
            "--minibatch_size",
            MINIBATCH_SIZE,
            "--total_steps",
            BATCH_SIZE * 2 * NUM_ITERS_2 + BATCH_SIZE,  # 1 + NUM_ITERS_2 iterations
            "--from_checkpoint", checkpoints[0],
            "--log_stats", "most",
        ):  # fmt: off
            with Setup() as setup2, Setup() as setup2b:  # second setup to make sure no side-effects are tested
                setup2.config.env_runners(num_env_runners=0)
                setup2b.config.env_runners(num_env_runners=0)
            self.assertEqual(setup2.args.total_steps, BATCH_SIZE * 2 * NUM_ITERS_2 + BATCH_SIZE)
            # Auto iteration will be 4; but only 3 new should be done.
            self.assertEqual(setup2.args.train_batch_size_per_learner, BATCH_SIZE * 2)
        Trainable2 = setup2b.create_trainable()
        if TYPE_CHECKING:
            Trainable2 = setup2b.trainable_class
        trainable2_local = Trainable2(setup2b.param_space)
        if trainable2_local.algorithm_config.callbacks_on_sample_end and isinstance(
            trainable2_local.algorithm_config.callbacks_on_sample_end, Iterable
        ):
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
        self.assertEqual(trainable2_local.algorithm_config.train_batch_size_per_learner, BATCH_SIZE * 2)

        tuner2 = setup2.create_tuner()
        tuner2._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(  # pyright: ignore[reportOptionalMemberAccess]
            checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
            checkpoint_score_order="max",
            checkpoint_frequency=1,
        )
        assert tuner2._local_tuner
        # assert that the stopper stops not too early, e.g. because parser.args.iterations was not updated.
        stoppers = tuner2._local_tuner.get_run_config().stop
        if isinstance(stoppers, (dict, Mapping)):
            self.assertEqual(stoppers.get("training_iteration"), NUM_ITERS_2 + 1)
        elif stoppers is None:
            pass
        else:
            if not isinstance(stoppers, list):
                if isinstance(stoppers, Iterable):
                    stoppers = list(stoppers)
                else:
                    stoppers = [stoppers]
            while stoppers:
                stopper = stoppers.pop()
                if isinstance(stopper, MaximumIterationStopper):
                    self.assertEqual(stopper._max_iter, NUM_ITERS_2 + 1)
                    break
                if isinstance(stopper, CombinedStopper):
                    for s in stopper._stoppers:
                        if isinstance(s, MaximumIterationStopper):
                            self.assertEqual(s._max_iter, NUM_ITERS_2 + 1)
                            break
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
        self.assertEqual(result2.metrics["current_step"], BATCH_SIZE * 2 * NUM_ITERS_2 + BATCH_SIZE)
        self.assertEqual(result2.metrics[TRAINING_ITERATION], NUM_ITERS_2 + 1)
        self.assertEqual(result2.metrics["iterations_since_restore"], NUM_ITERS_2)
        self.assertEqual(
            result2.metrics[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME],
            BATCH_SIZE * 2 * NUM_ITERS_2 + BATCH_SIZE,
        )

        # Change batch size change:
        self.assertEqual(
            result2.metrics[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME],
            BATCH_SIZE + (BATCH_SIZE * 2) * NUM_ITERS_2,
        )
        checkpoint_dir2, checkpoints2 = self.get_checkpoint_dirs(results2[0])
        self.assertEqual(
            len(checkpoints2),
            NUM_ITERS_2,  # 2 checkpoints as in total 3 steps; or does it save ?
            f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir2)} in {checkpoint_dir2}",
        )
        self.assertTrue(os.path.exists(checkpoints2[0]), "Checkpoint file does not exist: " + checkpoints2[0])


class TestOptunaTuner(SetupDefaults):
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

    def test_pruning(self):
        """
        Test might fail due to bad luck, low numbers first then high.

        Note:
            Remember Optuna might not prune the first 10 trials.
            Reduce num_jobs or adjust seed and test again.
        """
        with patch_args("--optimize_config", "--num_samples", "20", "--num_jobs", "4", "--seed", "42"):

            def trainable(params: dict[str, Any]) -> TrainableReturnData:
                logger.info("Running trainable with value: %s", params["fake_result"])
                for i in range(20):
                    tune.report(
                        {
                            "current_step": i,
                            EVALUATION_RESULTS: {ENV_RUNNER_RESULTS: {EPISODE_RETURN_MEAN: params["fake_result"]}},
                        }
                    )
                return {
                    "done": True,
                    "current_step": 20,
                    EVALUATION_RESULTS: {ENV_RUNNER_RESULTS: {EPISODE_RETURN_MEAN: params["fake_result"]}},
                }

            class RandomParamsSetup(AlgorithmSetup):
                PROJECT = "OptunaTest"

                def create_param_space(self):
                    return {
                        "fake_result": tune.grid_search([*[1] * 4, *[2] * 4, *[3] * 4, *[4] * 4, *[5] * 4]),
                        "module": "OptunaTest",
                        "env": "CartPole-v1",
                    }

                def _create_trainable(self):  # type: ignore[override]
                    return trainable

            setup = RandomParamsSetup()
            setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=64)
            setup.config.evaluation(evaluation_interval=1)
            from ray_utilities.setup import optuna_setup

            with self.assertLogs(optuna_setup._logger, level="INFO") as log:
                _results = run_tune(setup)

            self.assertTrue(
                any("Optuna pruning trial" in out for out in log.output),
                "Logger did not report a pruned trial, searching for 'Optuna pruning trial' in:\n"
                + "\n".join(log.output),
            )
            self.assertGreaterEqual(len([result for result in _results if result.metrics["current_step"] < 20]), 3)  # pyright: ignore[reportOptionalSubscript,reportAttributeAccessIssue]
            # NOTE: This can be OK even if runs fail!
