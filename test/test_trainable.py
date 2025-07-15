from __future__ import annotations

import io
import pickle
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest import mock, skip

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import COMPONENT_ENV_RUNNER
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
    from ray_utilities.typing.trainable_return import TrainableReturnData
    from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig

    from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co


class TestTraining(InitRay, TestHelpers, DisableLoggers, DisableGUIBreakpoints):
    @patch_args()
    def test_trainable_simple(self):
        # with self.subTest("No parameters"):
        #    _result = trainable({})
        def _create_trainable(self: PPOSetup):
            def trainable(params) -> TrainableReturnData:  # noqa: ARG001
                # This is a placeholder for the actual implementation of the trainable.
                # It should return a dictionary with training data.
                return self.config.build().train()  # type: ignore

            return trainable

        with mock.patch.object(PPOSetup, "_create_trainable", _create_trainable):
            with self.subTest("With parameters"):
                setup = PPOSetup(init_param_space=True, init_trainable=False)
                setup.config.evaluation(evaluation_interval=1)
                setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=32)
                trainable = setup.create_trainable()
                self.assertNotIsInstance(trainable, DefaultTrainable)
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


class TestClassCheckpointing(InitRay, TestHelpers, DisableLoggers, DisableGUIBreakpoints):
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

    @skip("TODO implement")
    def check_dynamic_settings_on_reload(self):
        # check _get_global_step on reload
        ...
