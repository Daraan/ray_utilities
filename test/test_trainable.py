from __future__ import annotations

import io
import pickle
import tempfile
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

from ray.experimental import tqdm_ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import COMPONENT_ENV_RUNNER
from ray.tune.result import TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.search.sample import Categorical, Domain
from ray.tune.utils import validate_save_restore

from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.setup.algorithm_setup import PPOSetup
from ray_utilities.testing_utils import (
    DisableGUIBreakpoints,
    DisableLoggers,
    InitRay,
    TestCases,
    TestHelpers,
    iter_cases,
    patch_args,
)
from ray_utilities.training.default_class import DefaultTrainable

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co


class TestTraining(InitRay, TestHelpers, DisableLoggers, DisableGUIBreakpoints):
    @patch_args()
    def test_trainable_function(self):
        # with self.subTest("No parameters"):
        #    _result = trainable({})
        with self.subTest("With parameters"):
            setup = PPOSetup(init_param_space=True)
            setup.config.evaluation(evaluation_interval=1)
            setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=32)
            trainable = setup.create_trainable()
            params = setup.sample_params()
            _result = trainable(params)

    @patch_args()
    def test_trainable_class_and_overrides(self):
        # Kind of like setUp for the other tests but with default args
        TrainableClass = DefaultTrainable.define(PPOSetup.typed())
        trainable = TrainableClass(
            overwrite_algorithm=AlgorithmConfig.overrides(
                num_epochs=2, minibatch_size=32, train_batch_size_per_learner=64
            )
        )
        self.assertAlmostEqual(trainable.algorithm_config.minibatch_size, 32)
        self.assertAlmostEqual(trainable.algorithm_config.train_batch_size_per_learner, 64)
        self.assertEqual(trainable.algorithm_config.num_epochs, 2)
        _result1 = trainable.step()
        trainable.cleanup()


class TestTrainableClass(InitRay, TestHelpers, DisableLoggers, DisableGUIBreakpoints):
    def setUp(self):
        super().setUp()

    @patch_args("--iterations", "5", "--total_steps", "320", "--batch_size", "64", "--comment", "running tests")
    def get_trainable(self, num_env_runners: int = 0):
        # NOTE: In this test attributes are shared BY identity, this is just a weak test.
        self.TrainableClass: type[DefaultTrainable[DefaultArgumentParser, PPOConfig, PPO]] = DefaultTrainable.define(
            PPOSetup.typed()
        )
        # this initializes the algorithm; overwrite batch_size of 64 again.
        trainable = self.TrainableClass(
            overwrite_algorithm=AlgorithmConfig.overrides(
                num_env_runners=num_env_runners, num_epochs=2, minibatch_size=32, train_batch_size_per_learner=32
            )
        )
        self.assertEqual(trainable.algorithm_config.num_env_runners, num_env_runners)
        self.assertEqual(trainable.algorithm_config.minibatch_size, 32)
        self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 32)
        self.assertEqual(trainable.algorithm_config.num_epochs, 2)
        self.assertEqual(trainable._setup.args.iterations, 5)
        self.assertEqual(trainable._setup.args.total_steps, 320)
        self.assertEqual(trainable._setup.args.train_batch_size_per_learner, 64)  # not overwritten

        result1 = trainable.step()
        return trainable, result1

    @TestCases([0, 1, 2])
    def test_save_checkpoint(self, cases):
        # NOTE: In this test attributes are shared BY identity, this is just a weak test.
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with tempfile.TemporaryDirectory() as tmpdir:
                # NOTE This loads some parts by identity!
                saved_ckpt = trainable.save_checkpoint(tmpdir)
                saved_ckpt = deepcopy(saved_ckpt)  # make sure we do not modify the original
                # pickle and load
                if False:
                    # fails because of:  AttributeError: Can't pickle local object 'mix_learners.<locals>.MixedLearner'
                    # Serialize saved_ckpt to a BytesIO object
                    buf = io.BytesIO()
                    pickle.dump(saved_ckpt, buf)
                    buf.seek(0)
                    # Deserialize from BytesIO
                    saved_ckpt = pickle.load(buf)
                with patch_args():  # make sure that args do not influence the restore
                    trainable2 = self.TrainableClass()
                    trainable2.load_checkpoint(saved_ckpt)
            self.compare_trainables(trainable, trainable2)

    @TestCases([1, 2, 3])
    def test_test_cases(self, cases):
        tested = []
        for i, r in enumerate(iter_cases(cases), start=1):
            self.assertEqual(r, i)
            tested.append(r)
        self.assertEqual(tested, [1, 2, 3])

    @TestCases([0, 1, 2])
    def test_save_restore(self, cases):
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with self.subTest(num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir:
                    training_result = trainable.save(tmpdir)  # calls save_checkpoint
                    print(f"Saved training result: {training_result}")
                    with patch_args():  # make sure that args do not influence the restore
                        trainable2 = self.TrainableClass()
                        trainable2.restore(deepcopy(training_result))
                        self.compare_trainables(trainable, trainable2)

    @TestCases([0, 1, 2])
    def test_get_set_state(self, cases):
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            state = trainable.get_state()
            # TODO: add no warning test
            self.assertIn(COMPONENT_ENV_RUNNER, state.get("algorithm", {}))
            trainable2 = self.TrainableClass()
            trainable2.set_state(deepcopy(state))
            # class is missing in config dict
            self.compare_trainables(trainable, trainable2)

    @TestCases([0, 1, 2])
    def test_safe_to_path(self, cases):
        """Test that the trainable can be saved to a path and restored."""
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with tempfile.TemporaryDirectory() as tmpdir:
                trainable.save_to_path(tmpdir)
                import os

                print(os.listdir(tmpdir))
                print(os.listdir(tmpdir + "/algorithm"))
                with patch_args():
                    trainable2 = self.TrainableClass()
                    trainable2.restore_from_path(tmpdir)
            self.compare_trainables(trainable, trainable2)

    def compare_trainables(
        self,
        trainable: DefaultTrainable["DefaultArgumentParser", "ConfigType_co", AlgorithmType_co],
        trainable2: DefaultTrainable["DefaultArgumentParser", "ConfigType_co", AlgorithmType_co],
    ) -> None:
        """Test functions for trainables obtained in different ways"""
        self.maxDiff = 60_000
        with self.subTest("Step 1: Compare trainables"):
            if hasattr(trainable, "_args") or hasattr(trainable2, "_args"):
                self.assertDictEqual(trainable2._args, trainable._args)  # type: ignore[attr-defined]
            self.assertEqual(trainable.algorithm_config.minibatch_size, 32)
            self.assertEqual(trainable2.algorithm_config.minibatch_size, trainable.algorithm_config.minibatch_size)
            self.assertEqual(trainable2._iteration, trainable._iteration)

            # get_state stores "class" : type(self) of the config, this allows from_state to work correctly
            # original trainable does not have that key
            config_dict1 = trainable.algorithm_config.to_dict()
            config_dict1.pop("class", None)
            config_dict2 = trainable2.algorithm_config.to_dict()
            config_dict2.pop("class", None)
            self.assertDictEqual(config_dict2, config_dict1)
            setup_data1 = trainable._setup.save()  # does not compare setup itself
            setup_data2 = trainable2._setup.save()
            # check all keys
            self.assertEqual(setup_data1.keys(), setup_data2.keys())
            keys = set(setup_data1.keys())
            keys.remove("__init_config__")
            self.assertDictEqual(vars(setup_data1["args"]), vars(setup_data2["args"]))  # SimpleNamespace
            keys.remove("args")
            self.assertIs(setup_data1["setup_class"], setup_data2["setup_class"])
            keys.remove("setup_class")
            assert setup_data1["config"] and setup_data2["config"]
            setup1_config = setup_data1["config"].to_dict()
            setup2_config = setup_data2["config"].to_dict()
            # remove class
            setup1_config.pop("class", None)
            setup2_config.pop("class", None)
            # Is set to False for one config, OldAPI value
            setup1_config.pop("simple_optimizer", None)
            setup2_config.pop("simple_optimizer", None)
            self.assertDictEqual(setup1_config, setup2_config)  # ConfigType
            keys.remove("config")
            param_space1 = setup_data1["param_space"]
            param_space2 = setup_data2["param_space"]
            keys.remove("param_space")
            self.assertEqual(len(keys), 0, f"Unchecked keys: {keys}")  # checked all params
            self.assertCountEqual(param_space1, param_space2)
            self.assertDictEqual(param_space1["cli_args"], param_space2["cli_args"])
            for key in param_space1.keys() | param_space2.keys():
                value1 = param_space1[key]
                value2 = param_space2[key]
                if isinstance(value1, Domain) or isinstance(value2, Domain):
                    # Domain is not hashable, so we cannot compare them directly
                    self.assertIs(type(value1), type(value2))
                    if isinstance(value1, Categorical):
                        assert isinstance(value2, Categorical)
                        self.assertListEqual(value1.categories, value2.categories)
                    else:
                        # This will likely fail, need to compare attributes
                        self.assertEqual(value1, value2, f"Domain {key} differs: {value1} != {value2}")
                else:
                    self.assertEqual(value1, value2, f"Parameter {key} differs: {value1} != {value2}")

            # Compare attrs
            self.assertIsNot(trainable2._reward_updaters, trainable._reward_updaters)
            for key in trainable2._reward_updaters.keys() | trainable._reward_updaters.keys():
                updater1 = trainable._reward_updaters[key]
                updater2 = trainable2._reward_updaters[key]
                self.assertIsNot(updater1, updater2)
                assert isinstance(updater1, partial) and isinstance(updater2, partial)
                self.assertDictEqual(updater1.keywords, updater2.keywords)
                self.assertIsNot(updater1.keywords["reward_array"], updater2.keywords["reward_array"])

            self.assertIsNot(trainable2._pbar, trainable._pbar)
            self.assertIs(type(trainable2._pbar), type(trainable._pbar))
            if isinstance(trainable2._pbar, tqdm_ray.tqdm):
                pbar1_state = trainable2._pbar._get_state()
                pbar2_state = trainable._pbar._get_state()  # type: ignore
                pbar1_state = {k: v for k, v in pbar1_state.items() if k not in ("desc", "uuid")}
                pbar2_state = {k: v for k, v in pbar2_state.items() if k not in ("desc", "uuid")}
                self.assertEqual(pbar1_state, pbar2_state)

            # Step 2
            result2 = trainable.step()
            result2_restored = trainable2.step()
            self.assertEqual(result2[TRAINING_ITERATION], result2_restored[TRAINING_ITERATION])
            self.assertEqual(result2[TRAINING_ITERATION], 2)

        # Compare env_runners
        with self.subTest("Step 2 Compare env_runner configs"):
            if trainable.algorithm.env_runner or trainable2.algorithm.env_runner:
                assert trainable.algorithm.env_runner and trainable2.algorithm.env_runner
                self.compare_env_runner_configs(trainable.algorithm, trainable2.algorithm)

    def test_validate_save_restore(self):
        """Basically test if TRAINING_ITERATION is set correctly."""
        # ray.init(include_dashboard=False, ignore_reinit_error=True)

        with patch_args("--iterations", "5", "--total_steps", "320", "--batch_size", "64"):
            # Need to fix argv for remote
            PPOTrainable = DefaultTrainable.define(PPOSetup.typed(), fix_argv=True)
            trainable = PPOTrainable()
            self.assertEqual(trainable._setup.args.iterations, 5)
            self.assertEqual(trainable._setup.args.total_steps, 320)
            validate_save_restore(PPOTrainable)
        # ray.shutdown()
