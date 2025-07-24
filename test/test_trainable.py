from __future__ import annotations

import os
import pickle
import sys
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest import mock, skip

from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import COMPONENT_ENV_RUNNER
from ray.rllib.utils.metrics import EVALUATION_RESULTS
from ray.tune.utils import validate_save_restore
from ray.util.multiprocessing import Pool

from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN
from ray_utilities.setup.algorithm_setup import AlgorithmSetup, PPOSetup
from ray_utilities.testing_utils import (
    Cases,
    DisableGUIBreakpoints,
    DisableLoggers,
    InitRay,
    TestHelpers,
    format_result_errors,
    iter_cases,
    patch_args,
)
from ray_utilities.training.default_class import DefaultTrainable
from test._mp_trainable import remote_process

if TYPE_CHECKING:
    from ray_utilities.typing.trainable_return import TrainableReturnData

if "--fast" in sys.argv:
    ENV_RUNNER_TESTS = [0]
else:
    ENV_RUNNER_TESTS = [0, 1, 2]


class TestTrainable(TestHelpers, DisableLoggers, DisableGUIBreakpoints):
    def test_1_subclass_check(self):
        """This test should run first as it has side-effects concerning ABCMeta"""
        TrainableClass = DefaultTrainable.define(PPOSetup.typed())
        TrainableClass2 = DefaultTrainable.define(PPOSetup.typed())
        self.assertTrue(issubclass(TrainableClass, TrainableClass2))
        self.assertTrue(issubclass(TrainableClass2, TrainableClass2))
        self.assertTrue(issubclass(TrainableClass, DefaultTrainable))
        self.assertTrue(issubclass(TrainableClass2, DefaultTrainable))

        self.assertFalse(issubclass(DefaultTrainable, TrainableClass))
        self.assertFalse(issubclass(DefaultTrainable, TrainableClass2))

    @patch_args()
    def test_trainable_simple(self):
        def _create_trainable(self: PPOSetup):
            global a_trainable_function  # noqa: PLW0603

            def a_trainable_function(params) -> TrainableReturnData:  # noqa: ARG001
                # This is a placeholder for the actual implementation of the trainable.
                # It should return a dictionary with training data.
                return self.config.build().train()  # type: ignore

            return a_trainable_function

        with mock.patch.object(PPOSetup, "_create_trainable", _create_trainable):
            with self.subTest("With parameters"):
                setup = PPOSetup(init_param_space=True, init_trainable=False)
                setup.config.evaluation(evaluation_interval=1)
                setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=32)
                trainable = setup.create_trainable()
                self.assertIs(trainable, a_trainable_function)
                print("Invalid check")
                self.assertNotIsInstance(trainable, DefaultTrainable)

                # train 1 step
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

    @patch_args()
    def test_train(self):
        self.TrainableClass = DefaultTrainable.define(PPOSetup.typed())
        trainable = self.TrainableClass(overwrite_algorithm=AlgorithmConfig.overrides(evaluation_interval=1))
        result = trainable.train()
        self.assertIn(EVALUATION_RESULTS, result)
        self.assertGreater(len(result[EVALUATION_RESULTS]), 0)

    def test_overrides_after_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_size1 = 40
            with patch_args(
                "--total_steps", "80", "--batch_size", batch_size1, "--minibatch_size", "20", "--comment", "A"
            ):
                trainable = AlgorithmSetup().trainable_class()
                self.assertEqual(trainable._total_steps["total_steps"], 80)
                self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 40)
                self.assertEqual(trainable.algorithm_config.minibatch_size, 20)
                self.assertEqual(trainable._setup.args.comment, "A")
                for i in range(1, 3):
                    result = trainable.step()
                    self.assertEqual(result["training_iteration"], i)
                    self.assertEqual(result["current_step"], batch_size1 * i)
                trainable.save(tmpdir)
                trainable.stop()
            with patch_args(
                "--total_steps", 80 + 120,
                "--batch_size", "60",
                "--comment", "B",
                "--from_checkpoint", tmpdir,
            ):  # fmt: off
                trainable2 = AlgorithmSetup().trainable_class()
                # left unchanged
                self.assertEqual(trainable2.algorithm_config.minibatch_size, 20)
                # Should change
                self.assertEqual(trainable2.algorithm_config.train_batch_size_per_learner, 60)
                self.assertEqual(trainable2._total_steps["total_steps"], 80 + 120)
                self.assertEqual(trainable2._setup.args.comment, "B")


class TestClassCheckpointing(InitRay, TestHelpers, DisableLoggers, DisableGUIBreakpoints):
    def setUp(self):
        super().setUp()

    @Cases(ENV_RUNNER_TESTS)
    def test_save_checkpoint(self, cases):
        # NOTE: In this test attributes are shared BY identity, this is just a weak test.
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with tempfile.TemporaryDirectory() as tmpdir:
                # NOTE This loads some parts by identity!
                saved_ckpt = trainable.save_checkpoint(tmpdir)
                saved_ckpt = deepcopy(saved_ckpt)  # assure to not compare by identity
                with patch_args():  # make sure that args do not influence the restore
                    trainable2 = self.TrainableClass()
                    trainable2.load_checkpoint(saved_ckpt)
            self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)
            trainable2.cleanup()

    @Cases(ENV_RUNNER_TESTS)
    def test_save_restore_dict(self, cases):
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with self.subTest(num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir:
                    training_result = trainable.save(tmpdir)  # calls save_checkpoint
                    with patch_args():  # make sure that args do not influence the restore
                        trainable2 = self.TrainableClass()
                        with self.subTest("Restore trainable from dict"):
                            self.assertIsInstance(training_result, dict)
                            trainable2.restore(deepcopy(training_result))  # calls load_checkpoint
                            self.compare_trainables(trainable, trainable2, "from dict", num_env_runners=num_env_runners)
                            trainable2.stop()

    @Cases(ENV_RUNNER_TESTS)
    def test_save_restore_path(self, cases):
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with self.subTest(num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainable.save(tmpdir)  # calls save_checkpoint
                    with patch_args():  # make sure that args do not influence the restore
                        trainable3 = self.TrainableClass()
                        self.assertIsInstance(tmpdir, str)
                        trainable3.restore(tmpdir)  # calls load_checkpoint
                        self.compare_trainables(trainable, trainable3, "from path", num_env_runners=num_env_runners)
                        trainable3.stop()

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
            # trainable.cleanup()
            trainable2.cleanup()

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
            trainable.cleanup()
        # ray.shutdown()

    @Cases([0])
    def test_interface_interchangeability(self, cases):
        """Test if methods can be used interchangeably."""
        for num_env_runners in iter_cases(cases):
            # create two trainables as compare_trainables takes a step
            trainable_a, _ = self.get_trainable(num_env_runners=num_env_runners)
            trainable_b, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_checkpoint -> restore_from_path"):
                with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
                    trainable_a.save_checkpoint(tmpdir1)
                    trainable_b.save_checkpoint(tmpdir2)
                    with patch_args():
                        trainable2 = self.TrainableClass()
                        trainable2.restore_from_path(tmpdir1)
                        trainable2_b: DefaultTrainable = self.TrainableClass.from_checkpoint(tmpdir2)  # pyright: ignore[reportAssignmentType]
                self.compare_trainables(trainable_a, trainable2, num_env_runners=num_env_runners)
                self.compare_trainables(trainable_b, trainable2_b, num_env_runners=num_env_runners)
            del trainable2
            del trainable2_b
            del trainable_a
            del trainable_b
            with self.subTest("save_to_path -> load_checkpoint"):
                trainable_c, _ = self.get_trainable(num_env_runners=num_env_runners)
                trainable_d, _ = self.get_trainable(num_env_runners=num_env_runners)
                with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
                    trainable_c.save_to_path(tmpdir1)
                    trainable_d.save_to_path(tmpdir2)
                    with patch_args():
                        trainable3 = self.TrainableClass()
                        trainable3.load_checkpoint(tmpdir1)
                        trainable3_b: DefaultTrainable = self.TrainableClass.from_checkpoint(tmpdir2)  # pyright: ignore[reportAssignmentType]
                self.compare_trainables(trainable_c, trainable3, num_env_runners=num_env_runners)
                self.compare_trainables(trainable_d, trainable3_b, num_env_runners=num_env_runners)

    def test_restore_multiprocessing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # pool works
            pool = Pool(1)
            pickled_trainable = pool.map(remote_process, [(tmpdir, None)])[0]
            # pickled_trainable_2 = parent_conn.recv()
            pool.close()
            print("Restoring trainable from saved data")
            trainable_restored = DefaultTrainable.define(PPOSetup.typed()).from_checkpoint(tmpdir)
            # Compare with default trainable:
            print("Create new default trainable")
            self._disable_tune_loggers.start()
            trainable, _ = self.get_trainable()
            self.compare_trainables(
                trainable,
                trainable_restored,
            )
            print("Compareing restored trainable with pickled trainable")
            if pickled_trainable is not None:
                trainable_restored2 = pickle.loads(pickled_trainable)
                self.compare_trainables(
                    trainable_restored,  # <-- need new trainable here
                    trainable_restored2,
                )

    @Cases(ENV_RUNNER_TESTS)
    def test_tuner_checkpointing(self, cases):
        # self.enable_loggers()
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--batch_size", "32",
            "--minibatch_size", "16",
            "--iterations", "3",
        ):  # fmt: off
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
                    self.assertEqual(result.num_errors, 0, format_result_errors(result.errors))  # pyright: ignore[reportAttributeAccessIssue,reportOptionalSubscript]
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
                        self.assertIsInstance(checkpoint, str)
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
