from __future__ import annotations

import io
import os
import pickle
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest import mock, skip

from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import COMPONENT_ENV_RUNNER
from ray.tune.utils import validate_save_restore
from typing_extensions import Final

from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN
from ray_utilities.setup.algorithm_setup import AlgorithmSetup, PPOSetup
from ray_utilities.testing_utils import (
    DisableGUIBreakpoints,
    DisableLoggers,
    InitRay,
    Cases,
    TestHelpers,
    iter_cases,
    patch_args,
)
from ray_utilities.training.default_class import DefaultTrainable

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig

    from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co
    from ray_utilities.typing.trainable_return import TrainableReturnData

ENV_RUNNER_TESTS = [0]


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

    def test_subclass_check(self):
        TrainableClass = DefaultTrainable.define(PPOSetup.typed())
        TrainableClass2 = DefaultTrainable.define(PPOSetup.typed())
        self.assertTrue(issubclass(TrainableClass, TrainableClass2))
        self.assertTrue(issubclass(TrainableClass2, TrainableClass2))
        self.assertTrue(issubclass(TrainableClass, DefaultTrainable))
        self.assertTrue(issubclass(TrainableClass2, DefaultTrainable))
        self.assertFalse(issubclass(DefaultTrainable, TrainableClass))
        self.assertFalse(issubclass(DefaultTrainable, TrainableClass2))


OVERRIDE_KEYS: Final[set[str]] = {"num_env_runners", "num_epochs", "minibatch_size", "train_batch_size_per_learner"}


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
        # This does not modify the state["setup"]["config"]
        overrides = AlgorithmConfig.overrides(
            num_env_runners=num_env_runners, num_epochs=2, minibatch_size=32, train_batch_size_per_learner=32
        )
        trainable = self.TrainableClass(overwrite_algorithm=overrides)
        self.assertEqual(trainable._overwrite_algorithm, overrides)
        self.assertEqual(overrides.keys(), OVERRIDE_KEYS)
        self.assertEqual(trainable.algorithm_config.num_env_runners, num_env_runners)
        self.assertEqual(trainable.algorithm_config.minibatch_size, 32)
        self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 32)
        self.assertEqual(trainable.algorithm_config.num_epochs, 2)
        self.assertEqual(trainable._setup.args.iterations, 5)
        self.assertEqual(trainable._setup.args.total_steps, 320)
        self.assertEqual(trainable._setup.args.train_batch_size_per_learner, 64)  # not overwritten

        result1 = trainable.step()
        return trainable, result1

    @Cases(ENV_RUNNER_TESTS)
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
            self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)

    @Cases(ENV_RUNNER_TESTS)
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
                        self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)

    @Cases(ENV_RUNNER_TESTS)
    def test_get_set_state(self, cases):
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            state = trainable.get_state()
            # TODO: add no warning test
            self.assertIn(COMPONENT_ENV_RUNNER, state.get("algorithm", {}))

            trainable2 = self.TrainableClass()
            trainable2.set_state(deepcopy(state))
            # class is missing in config dict
            self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)

    @Cases(ENV_RUNNER_TESTS)
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
            self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)

    def test_validate_save_restore(self):
        """Basically test if TRAINING_ITERATION is set correctly."""
        # ray.init(include_dashboard=False, ignore_reinit_error=True)

        with patch_args("--iterations", "5", "--total_steps", "320", "--batch_size", "64", "--minibatch_size", "32"):
            # Need to fix argv for remote
            PPOTrainable = DefaultTrainable.define(PPOSetup.typed(), fix_argv=True)
            trainable = PPOTrainable()
            self.assertEqual(trainable._setup.args.iterations, 5)
            self.assertEqual(trainable._setup.args.total_steps, 320)
            validate_save_restore(PPOTrainable)
        # ray.shutdown()

    @Cases(ENV_RUNNER_TESTS)
    def test_with_tuner(self, cases):
        # self.enable_loggers()
        with patch_args(
            "--num_samples", "1", "--num_jobs", "1", "--batch_size", "32", "--minibatch_size", "16", "--iterations", "3"
        ):
            for num_env_runners in iter_cases(cases):
                with self.subTest(num_env_runners=num_env_runners):
                    setup = AlgorithmSetup(init_trainable=False)
                    setup.config.env_runners(num_env_runners=num_env_runners)
                    setup.config.training(minibatch_size=32)  # insert some noise
                    setup.create_trainable()
                    tuner = setup.create_tuner()
                    assert tuner._local_tuner
                    tuner._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(
                        checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                        checkpoint_score_order="max",
                        checkpoint_frequency=1,  # Save every iteration
                        # NOTE: num_keep does not appear to work here
                    )
                    result = tuner.fit()
                    self.assertEquals(result.num_errors, 0, "Encountered errors: " + str(result.errors))  # pyright: ignore[reportAttributeAccessIssue,reportOptionalSubscript]
                    checkpoint_dir, checkpoints = self.get_checkpoint_dirs(result[0])
                    self.assertEqual(
                        len(checkpoints),
                        3,
                        f"Checkpoints were not created. Found: {os.listdir(checkpoint_dir)} in {checkpoint_dir}",
                    )
                    # sort to get checkpoints in order 000, 001, 002
                    for step, checkpoint in enumerate(sorted(checkpoints), 1):
                        self.assertTrue(os.path.exists(checkpoint))

                        trainable_from_path = DefaultTrainable.define(setup)()
                        trainable_from_path.restore_from_path(checkpoint)
                        trainable_from_ckpt: DefaultTrainable = DefaultTrainable.define(setup).from_checkpoint(
                            checkpoint
                        )  # pyright: ignore[reportAssignmentType]
                        # restore is bad if algorithm_checkpoint_dir is a temp dir
                        self.assertEqual(trainable_from_ckpt.algorithm.iteration, step)
                        self.assertEqual(trainable_from_path.algorithm.iteration, step)
                        self.compare_trainables(
                            trainable_from_path,
                            trainable_from_ckpt,
                            "compare from_path with from_checkpoint",
                            iteration_after_step=step + 1,
                            step=step,
                        )
                        trainable_restore = DefaultTrainable.define(setup)()
                        trainable_restore.restore(checkpoint)
                        self.assertEqual(trainable_restore.algorithm.iteration, step)
                        trainable_from_path.restore_from_path(checkpoint)  # load a second time to test as well
                        self.compare_trainables(
                            trainable_restore,
                            trainable_from_path,
                            "compare trainable_restore with from_path x2",
                            iteration_after_step=step + 1,
                            step=step,
                        )

    @skip("TODO implement")
    def check_dynamic_settings_on_reload(self):
        # check _get_global_step on reload
        ...
