from ray.tune.search.optuna import OptunaSearch

from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.runfiles import run_tune

from .utils import SetupDefaults, patch_args


class TestTuner(SetupDefaults):
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
            results = run_tune(optuna_setup)
            # NOTE: This can be OK even if runs fail!
