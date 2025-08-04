from __future__ import annotations

import os
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

try:
    from ray.train._internal.session import _TrainingResult
except ImportError:
    _TrainingResult = None

if TYPE_CHECKING:
    from ray_utilities.typing.trainable_return import TrainableReturnData

if "--fast" in sys.argv:
    ENV_RUNNER_TESTS = [0]
elif "--mp-only" in sys.argv:
    ENV_RUNNER_TESTS = [1, 2]
else:
    ENV_RUNNER_TESTS = [0, 1]


class TestTrainable(InitRay, TestHelpers, DisableLoggers, DisableGUIBreakpoints):
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

    def test_get_trainable_util(self):
        trainable1, _ = self.get_trainable(num_env_runners=0, env_seed=5)
        trainable2, _ = self.get_trainable(num_env_runners=0, env_seed=5)
        self.compare_trainables(trainable1, trainable2, num_env_runners=0, ignore_timers=True)

        trainable1_1, _ = self.get_trainable(num_env_runners=1, env_seed=5)
        trainable2_1, _ = self.get_trainable(num_env_runners=1, env_seed=5)
        self.compare_trainables(trainable1_1, trainable2_1, num_env_runners=1, ignore_timers=True)

    @patch_args()
    def test_trainable_class_and_overrides(self):
        # Kind of like setUp for the other tests but with default args
        TrainableClass = DefaultTrainable.define(PPOSetup.typed())
        trainable = TrainableClass(
            algorithm_overrides=AlgorithmConfig.overrides(
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
        trainable = self.TrainableClass(algorithm_overrides=AlgorithmConfig.overrides(evaluation_interval=1))
        result = trainable.train()
        self.assertIn(EVALUATION_RESULTS, result)
        self.assertGreater(len(result[EVALUATION_RESULTS]), 0)

    def test_overrides_after_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_size1 = 40
            with patch_args(
                "--total_steps", "80",
                "--batch_size", batch_size1,
                "--minibatch_size", "20",
                "--comment", "A",
                "--tags", "test",
            ):  # fmt: off
                with AlgorithmSetup() as setup:
                    setup.config.evaluation(evaluation_interval=1)
                    setup.config.training(
                        num_epochs=2,
                        minibatch_size=10,  # overwrite CLI
                    )
                trainable = setup.trainable_class(algorithm_overrides=AlgorithmConfig.overrides(gamma=0.11, lr=2.0))
                self.assertEqual(trainable._total_steps["total_steps"], 80)
                self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 40)
                self.assertEqual(trainable.algorithm_config.minibatch_size, 10)
                self.assertEqual(trainable.algorithm_config.num_epochs, 2)
                self.assertEqual(trainable.algorithm_config.gamma, 0.11)
                self.assertEqual(trainable.algorithm_config.lr, 2.0)
                self.assertEqual(trainable._setup.args.comment, "A")
                self.assertEqual(trainable._setup.args.tags, ["test"])
                for i in range(1, 3):
                    result = trainable.train()
                    self.assertEqual(result["training_iteration"], i)
                    self.assertEqual(result["current_step"], batch_size1 * i)
                    self.assertEqual(trainable._iteration, i)
                    self.assertEqual(trainable.get_state()["trainable"]["iteration"], i)
                trainable.save(tmpdir)
                trainable.stop()
            del trainable
            del setup
            with patch_args(
                "--total_steps", 80 + 120,  # Should be divisible by new batch_size
                "--batch_size", "60",
                "--comment", "B",
                "--from_checkpoint", tmpdir,
            ):  # fmt: off
                with AlgorithmSetup(init_trainable=False) as setup2:
                    setup2.config.training(
                        num_epochs=5,
                    )
                    setup2.config_overrides(num_epochs=5)
                self.assertDictEqual(
                    setup2.config_overrides(),
                    {
                        "num_epochs": 5,
                    },
                )
                # self.assertEqual(setup2.args.total_steps % setup2.args.train_batch_size_per_learner, 0)
                trainable2 = setup2.trainable_class(
                    setup2.param_space, algorithm_overrides=AlgorithmConfig.overrides(gamma=0.22, grad_clip=4.321)
                )
                assert trainable2._algorithm_overrides is not None
                self.assertDictEqual(
                    trainable2._algorithm_overrides,
                    {
                        "gamma": 0.22,
                        "grad_clip": 4.321,
                    },
                )
                assert trainable2._algorithm_overrides
                self.assertEqual(setup2.args.comment, "B")
                self.assertEqual(trainable2._iteration, 2)
                # Do not trust restored setup; still original args
                self.assertEqual(trainable2._setup.args.comment, "A")
                self.assertEqual(trainable2.algorithm_config.lr, 0.001)  # default value
                # left unchanged
                # From manual adjustment
                self.assertEqual(trainable2.algorithm_config.num_epochs, 5)
                # Should change
                # from override
                self.assertEqual(trainable2.algorithm_config.gamma, 0.22)
                self.assertEqual(trainable2.algorithm_config.grad_clip, 4.321)
                # From CLI
                self.assertEqual(trainable2.algorithm_config.train_batch_size_per_learner, 60)
                self.assertEqual(trainable2._total_steps["total_steps"], 80 + 120)
                # NOT restored as set by config_from_args
                self.assertEqual(trainable2.algorithm_config.minibatch_size, 20)


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
            trainable.stop()
            trainable2.stop()

    @Cases(ENV_RUNNER_TESTS)
    def test_save_restore_dict(self, cases):
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with self.subTest(num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir:
                    training_result: _TrainingResult = trainable.save(tmpdir)  # pyright: ignore[reportInvalidTypeForm] # calls save_checkpoint
                    with patch_args():  # make sure that args do not influence the restore
                        trainable2 = self.TrainableClass()
                        with self.subTest("Restore trainable from dict"):
                            if _TrainingResult is not None:
                                self.assertIsInstance(training_result, _TrainingResult)
                            trainable2.restore(deepcopy(training_result))  # calls load_checkpoint
                            self.compare_trainables(trainable, trainable2, "from dict", num_env_runners=num_env_runners)
                            trainable2.stop()
            trainable.stop()

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
            trainable.stop()

    @Cases(ENV_RUNNER_TESTS)
    def test_1_get_set_state(self, cases):
        # If this test fails all others will most likely fail too, run it first.
        self.maxDiff = None
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            state = trainable.get_state()
            # TODO: add no warning test
            self.assertIn(COMPONENT_ENV_RUNNER, state.get("algorithm", {}))

            trainable2 = self.TrainableClass()
            trainable2.set_state(deepcopy(state))
            # class is missing in config dict
            self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)
            trainable.stop()
            trainable2.stop()

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
            trainable.stop()
            trainable2.stop()

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
            trainable.stop()

    @Cases(ENV_RUNNER_TESTS)
    def test_interchange_save_checkpoint_restore_from_path(self, cases):
        """Test if methods can be used interchangeably."""
        # NOTE: restore_from_path currently does not set (local) env_runner state when num_env_runners > 0
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_checkpoint -> restore_from_path", num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir1:
                    trainable.save_checkpoint(tmpdir1)
                    with patch_args():
                        trainable_from_path = self.TrainableClass()
                        trainable_from_path.restore_from_path(tmpdir1)
                self.compare_trainables(
                    trainable,
                    trainable_from_path,
                    "save_checkpoint -> restore_from_path",
                    num_env_runners=num_env_runners,
                )
            trainable.stop()
            trainable_from_path.stop()

    @Cases(ENV_RUNNER_TESTS)
    def test_interchange_save_checkpoint_from_checkpoint(self, cases):
        """Test if methods can be used interchangeably."""
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_checkpoint -> from_checkpoint", num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir1:
                    trainable.save_checkpoint(tmpdir1)
                    with patch_args():
                        trainable_from_checkpoint: DefaultTrainable = self.TrainableClass.from_checkpoint(tmpdir1)
                self.compare_trainables(
                    trainable,
                    trainable_from_checkpoint,
                    "save_checkpoint -> from_checkpoint",
                    num_env_runners=num_env_runners,
                    # NOTE: from_checkpoint loads the local env_runner state correctly, but it reflects remote
                    # the restore_from_path variant does not do that
                    ignore_env_runner_state=num_env_runners > 0,
                )
            trainable.stop()
            trainable_from_checkpoint.stop()

    @Cases(ENV_RUNNER_TESTS)
    def test_interchange_save_to_path_restore_from_path(self, cases):
        """Test if methods can be used interchangeably."""
        # NOTE: restore_from_path currently does not set (local) env_runner state when num_env_runners > 0
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_to_path -> restore_from_path", num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir1:
                    trainable.save_to_path(tmpdir1)
                    with patch_args():
                        trainable_from_path = self.TrainableClass()
                        trainable_from_path.restore_from_path(tmpdir1)
                self.compare_trainables(
                    trainable,
                    trainable_from_path,
                    "save_to_path -> restore_from_path",
                    num_env_runners=num_env_runners,
                    # NOTE: from_checkpoint loads the local env_runner state correctly, but it reflects remote
                    ignore_env_runner_state=num_env_runners > 0,
                )
            trainable.stop()
            trainable_from_path.stop()

    @Cases(ENV_RUNNER_TESTS)
    def test_interchange_save_to_path_from_checkpoint(self, cases):
        """Test if methods can be used interchangeably."""
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_to_path -> from_checkpoint", num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir1:
                    trainable.save_to_path(tmpdir1)
                    with patch_args():
                        trainable_from_checkpoint: DefaultTrainable = self.TrainableClass.from_checkpoint(tmpdir1)
                if num_env_runners > 0:
                    assert trainable_from_checkpoint.algorithm.env_runner_group
                    remote_configs = trainable_from_checkpoint.algorithm.env_runner_group.foreach_env_runner(
                        lambda r: r.config
                    )
                self.compare_trainables(
                    trainable,
                    trainable_from_checkpoint,
                    "save_to_path -> from_checkpoint",
                    num_env_runners=num_env_runners,
                    ignore_env_runner_state=num_env_runners > 0,
                )
            trainable.stop()
            trainable_from_checkpoint.stop()

    @Cases(ENV_RUNNER_TESTS)
    def test_restore_multiprocessing(self, cases):
        for num_env_runners in iter_cases(cases):
            with tempfile.TemporaryDirectory() as tmpdir:
                # pool works
                pool = Pool(1)
                data = {
                    "dir": tmpdir,
                    "connection": None,
                    "num_env_runners": num_env_runners,
                    "env_seed": 5,
                }
                pickled_trainable = pool.map(remote_process, [data])[0]
                # pickled_trainable_2 = parent_conn.recv()
                pool.close()
                print("Restoring trainable from saved data")
                trainable_restored = DefaultTrainable.define(PPOSetup.typed()).from_checkpoint(tmpdir)
                # Compare with default trainable:
                print(f"Create new default trainable num_env_runners={num_env_runners}")
                self._disable_tune_loggers.start()
                trainable, _ = self.get_trainable(num_env_runners=num_env_runners, env_seed=data["env_seed"])
                self.compare_trainables(
                    trainable, trainable_restored, num_env_runners=num_env_runners, ignore_timers=True
                )
                trainable.stop()
                if pickled_trainable is not None:  # TODO: Use cloudpickle
                    print("Comparing restored trainable with pickled trainable")
                    import cloudpickle

                    trainable_restored2 = cloudpickle.loads(pickled_trainable)
                    self.compare_trainables(
                        trainable_restored,  # <-- need new trainable here
                        trainable_restored2,
                        num_env_runners=num_env_runners,
                    )
                    trainable_restored2.stop()
                trainable_restored.stop()

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
                        trainable_from_path.stop()
                        trainable_restore = DefaultTrainable.define(setup)()
                        # Problem restore uses load_checkpoint, which passes a dict to load_checkpoint
                        # however the checkpoint dir is unknown inside the loaded dict
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
                        trainable_from_path.stop()
                        trainable_restore.stop()

    @skip("TODO implement")
    def check_dynamic_settings_on_reload(self):
        # check _get_global_step on reload
        ...
