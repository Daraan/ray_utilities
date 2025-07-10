from ray import tune
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
)
from ray.tune.search.optuna import OptunaSearch

from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN
from ray_utilities.runfiles import run_tune
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.optuna_setup import OptunaSearchWithPruner
from ray_utilities.testing_utils import InitRay, SetupDefaults, patch_args
from ray_utilities.typing.trainable_return import TrainableReturnData

logger = __import__("logging").getLogger(__name__)


class TestTuner(InitRay, SetupDefaults):
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

    def test_run_tune_with_tuner(self):
        with patch_args("--optimize_config", "--num_samples", "5", "--num_jobs", "2"):
            optuna_setup = AlgorithmSetup()
            optuna_setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=64)
            optuna_setup.config.evaluation(evaluation_interval=1)  # else eval metric not in dict
            _results = run_tune(optuna_setup)
            # NOTE: This can be OK even if runs fail!


class TestOptunaTuner(SetupDefaults):
    def test_optuna_tuner_setup(self):
        with patch_args("--optimize_config", "--num_samples", "1"):
            optuna_setup = AlgorithmSetup()
            self.assertTrue(optuna_setup.args.optimize_config)
            tuner = optuna_setup.create_tuner()
            assert tuner._local_tuner and tuner._local_tuner._tune_config
            self.assertIsInstance(tuner._local_tuner._tune_config.search_alg, OptunaSearch)
            self.assertIsInstance(tuner._local_tuner.get_run_config().stop, OptunaSearchWithPruner)
            # stopper and search are the same instance
            self.assertIs(tuner._local_tuner.get_run_config().stop, tuner._local_tuner._tune_config.search_alg)
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

            def trainable(params) -> TrainableReturnData:
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

                def create_trainable(self):
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
