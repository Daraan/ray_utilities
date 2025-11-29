from __future__ import annotations

import os
import shutil
import sys
import tempfile
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest import mock, skip

import cloudpickle
import pytest
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import COMPONENT_ENV_RUNNER
from ray.rllib.utils.metrics import EVALUATION_RESULTS
from ray.tune.utils import validate_save_restore
from ray.util.multiprocessing import Pool

from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import DynamicEvalInterval
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN, FORK_FROM, PERTURBED_HPARAMS
from ray_utilities.dynamic_config.dynamic_buffer_update import split_timestep_budget
from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import PPOTorchLearnerWithGradientAccumulation
from ray_utilities.setup.algorithm_setup import AlgorithmSetup, PPOSetup
from ray_utilities.setup.ppo_mlp_setup import MLPSetup
from ray_utilities.testing_utils import (
    ENV_RUNNER_CASES,
    Cases,
    DisableGUIBreakpoints,
    DisableLoggers,
    InitRay,
    TestHelpers,
    iter_cases,
    mock_trainable_algorithm,
    no_parallel_envs,
    patch_args,
)
from ray_utilities.training.default_class import DefaultTrainable
from ray_utilities.training.helpers import make_divisible

try:
    from ray.train._internal.session import _TrainingResult
except ImportError:
    _TrainingResult = None

if TYPE_CHECKING:
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    from ray_utilities.typing.trainable_return import TrainableReturnData

try:
    sys.argv.remove("test/test_trainable.py")
except ValueError:
    pass


class TestTrainable(InitRay, TestHelpers, DisableLoggers, DisableGUIBreakpoints, num_cpus=4):
    def test_1_subclass_check(self):
        """This test should run first as it has side-effects concerning ABCMeta."""
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
            global _a_trainable_function  # noqa: PLW0603

            def _a_trainable_function(params) -> TrainableReturnData:  # noqa: ARG001
                # This is a placeholder for the actual implementation of the trainable.
                # It should return a dictionary with training data.
                return self.config.build().train()  # type: ignore

            return _a_trainable_function

        with mock.patch.object(PPOSetup, "_create_trainable", _create_trainable):
            with self.subTest("With parameters"):  # noqa: SIM117
                setup = PPOSetup(init_param_space=True, init_trainable=False)
                setup.config.evaluation(evaluation_interval=1)
                setup.config.training(num_epochs=2, train_batch_size_per_learner=64, minibatch_size=32)
                trainable = setup.create_trainable()
                self.assertIs(trainable, _a_trainable_function)
                print("Invalid check")
                self.assertNotIsInstance(trainable, DefaultTrainable)

                # train 1 step
                params = setup.sample_params()
                _result = trainable(params)

    @no_parallel_envs
    def test_get_trainable_util(self):
        trainable1, _ = self.get_trainable(num_env_runners=0, env_seed=5)
        trainable2, _ = self.get_trainable(num_env_runners=0, env_seed=5)
        self.compare_trainables(
            trainable1, trainable2, num_env_runners=0, ignore_timers=True, ignore_env_runner_state=False
        )

        trainable1_1, _ = self.get_trainable(num_env_runners=1, env_seed=5)
        trainable2_1, _ = self.get_trainable(num_env_runners=1, env_seed=5)
        self.compare_trainables(
            trainable1_1, trainable2_1, num_env_runners=1, ignore_timers=True, ignore_env_runner_state=False
        )

    @pytest.mark.basic
    @no_parallel_envs
    def test_get_trainable_fast_model(self):
        # Test model sizes:
        DefaultArgumentParser.num_envs_per_env_runner = 1
        trainable1, _ = self.get_trainable(num_env_runners=0, env_seed=5, train=False, fast_model=True)
        assert self._model_config is not None
        new_trainable = self.TrainableClass()  # with model_config in hparams
        for trainable in (trainable1, new_trainable):
            with self.subTest("Model config check", model_config_in_hparams=trainable is new_trainable):
                runner_module = cast("SingleAgentEnvRunner", trainable.algorithm.env_runner).module
                assert (
                    runner_module
                    and trainable.algorithm.learner_group is not None
                    and trainable.algorithm.learner_group._learner
                )
                learner_module = trainable.algorithm.learner_group._learner.module["default_policy"]
                for ctx, model_config_dict in [
                    ("runner", runner_module.model_config),
                    ("algorithm_config", trainable.algorithm_config.model_config),
                    ("learner_module", learner_module.model_config),
                ]:
                    if trainable is new_trainable:
                        pass
                    self.assertEqual(model_config_dict["fcnet_hiddens"], [self._fast_model_fcnet_hiddens], ctx)
                    self.assertEqual(model_config_dict["head_fcnet_hiddens"], [], ctx)
                    self.assertDictContainsSubset(self._model_config, model_config_dict, ctx)

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
        trainable.stop()

    @patch_args("--batch_size", "64", "--minibatch_size", "32", "--num_envs_per_env_runner", "4")
    def test_train(self):
        self.TrainableClass = DefaultTrainable.define(PPOSetup.typed())
        trainable = self.TrainableClass(algorithm_overrides=AlgorithmConfig.overrides(evaluation_interval=1))
        result = trainable.train()
        self.assertIn(EVALUATION_RESULTS, result)
        self.assertGreater(len(result[EVALUATION_RESULTS]), 0)
        self.assertEqual(result["current_step"], 64)

    @patch_args("--minibatch_scale", "1.0", "--batch_size", "256", "--minibatch_size", "64")
    def test_minibatch_scale_trainable(self):
        """Test that minibatch_scale is applied at trainable level"""
        with PPOSetup() as setup:
            # check if override works
            setup.config.train_batch_size_per_learner = 512
        trainable = setup.trainable_class()

        # Verify that minibatch_scale=1.0 sets minibatch_size equal to train_batch_size_per_learner
        self.assertEqual(trainable.algorithm_config.minibatch_size, 512)
        self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 512)
        trainable.stop()

    def test_minibatch_scale_with_checkpoint_restore(self):
        """Test that minibatch_scale is preserved through checkpoint save/restore with different batch sizes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First setup with batch_size=256, no scaling
            with patch_args("--batch_size", "256", "--minibatch_size", "64"):
                setup1 = PPOSetup()
            TrainableClass1 = setup1.trainable_class
            trainable1 = TrainableClass1()

            # Verify initial state
            self.assertEqual(trainable1.algorithm_config.minibatch_size, 64)
            self.assertEqual(trainable1.algorithm_config.train_batch_size_per_learner, 256)

            # Save checkpoint
            checkpoint_path = trainable1.save_to_path(tmpdir)
            trainable1.stop()

            # Second setup with larger batch_size=512 and minibatch_scale=0.5
            with patch_args("--minibatch_scale", "0.5", "--batch_size", "512", "--minibatch_size", "64"):
                setup2 = PPOSetup()
            TrainableClass2 = setup2.trainable_class
            trainable2 = TrainableClass2()

            # Verify trainable2 has scaled minibatch_size (512 * 0.5 = 256)
            self.assertEqual(trainable2.algorithm_config.minibatch_size, 256)
            self.assertEqual(trainable2.algorithm_config.train_batch_size_per_learner, 512)

            # Restore from checkpoint created with first setup
            trainable2.load_checkpoint(checkpoint_path)

            # After restore, should maintain minibatch_scale behavior with setup2's larger batch size
            self.assertEqual(trainable2.algorithm_config.train_batch_size_per_learner, 512)
            self.assertEqual(trainable2.algorithm_config.minibatch_size, 256)  # 512 * 0.5 = 256
            trainable2.stop()

    def test_overrides_after_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_size1 = 40
            # Ensure divisibility by num_envs_per_env_runner without reducing the value
            batch_size1 = make_divisible(batch_size1, DefaultArgumentParser.num_envs_per_env_runner)
            batch_size2 = make_divisible(80, DefaultArgumentParser.num_envs_per_env_runner)
            mini_batch_size1 = make_divisible(20, DefaultArgumentParser.num_envs_per_env_runner)
            mini_batch_size2 = make_divisible(40, DefaultArgumentParser.num_envs_per_env_runner)
            override_mini_batch_size = make_divisible(10, DefaultArgumentParser.num_envs_per_env_runner)
            # Meta checks, sizes should differ for effective tests
            self.assertNotEqual(batch_size1, batch_size2)
            self.assertNotEqual(mini_batch_size1, mini_batch_size2)
            self.assertNotEqual(override_mini_batch_size, mini_batch_size2)
            self.assertNotEqual(batch_size1, mini_batch_size1)
            self.assertNotEqual(batch_size2, mini_batch_size2)
            with patch_args(
                "--total_steps", batch_size1 * 2,
                "--use_exact_total_steps",  # Do not adjust total_steps
                "--batch_size", batch_size1,
                "--minibatch_size", mini_batch_size1,
                "--comment", "A",
                "--tags", "test",
            ):  # fmt: skip
                with AlgorithmSetup() as setup:
                    setup.config.evaluation(evaluation_interval=1)
                    setup.config.training(
                        num_epochs=2,
                        minibatch_size=override_mini_batch_size,  # overwrite CLI
                    )
                trainable = setup.trainable_class(algorithm_overrides=AlgorithmConfig.overrides(gamma=0.11, lr=2.0))
                self.assertEqual(trainable._total_steps["total_steps"], batch_size1 * 2)
                self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, batch_size1)
                self.assertEqual(trainable.algorithm_config.minibatch_size, override_mini_batch_size)
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
                "--total_steps", 2 * batch_size1 + 2 * batch_size2,  # Should be divisible by new batch_size
                "--use_exact_total_steps",  # Do not adjust total_steps
                "--batch_size", batch_size2,
                "--comment", "B",
                "--from_checkpoint", tmpdir,
                "--num_envs_per_env_runner", 1,
            ):  # fmt: skip
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
                    setup2.sample_params(), algorithm_overrides=AlgorithmConfig.overrides(gamma=0.22, grad_clip=4.321)
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
                self.assertEqual(trainable2.algorithm_config.lr, DefaultArgumentParser.lr)  # default value
                # left unchanged
                # From manual adjustment
                self.assertEqual(trainable2.algorithm_config.num_epochs, 5)
                # Should change
                # from override
                self.assertEqual(trainable2.algorithm_config.gamma, 0.22)
                self.assertEqual(trainable2.algorithm_config.grad_clip, 4.321)
                # From CLI
                self.assertEqual(trainable2.algorithm_config.train_batch_size_per_learner, batch_size2)
                self.assertEqual(trainable2._total_steps["total_steps"], 2 * batch_size1 + 2 * batch_size2)
                # NOT restored as set by config_from_args
                self.assertEqual(trainable2.algorithm_config.minibatch_size, mini_batch_size1)

    @no_parallel_envs
    def test_perturbed_keys(self):
        with (
            patch_args("--batch_size", 512, "--fcnet_hiddens", "[1]"),
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory() as tmpdir2,
            tempfile.TemporaryDirectory() as tmpdir3,
        ):
            setup = MLPSetup()
            trainable = setup.trainable_class()
            self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 512)
            trainable.stop()
            trainable_perturbed = setup.trainable_class(
                {"train_batch_size_per_learner": 222, PERTURBED_HPARAMS: {"train_batch_size_per_learner": 222}}
            )
            self.assertEqual(trainable_perturbed.algorithm_config.train_batch_size_per_learner, 222)
            ckpt = trainable.save_checkpoint(tmpdir)
            ckpt_perturbed = trainable_perturbed.save_checkpoint(tmpdir2)
            trainable_perturbed.stop()

            class PerturbedInt(int):
                """Class to check if the final chosen value is the int from the perturbed subdict."""

            trainable2 = setup.trainable_class(
                # NOTE: Normally should be the same but check that perturbed is used after load_checkpoint
                {
                    "train_batch_size_per_learner": 123,
                    PERTURBED_HPARAMS: {"train_batch_size_per_learner": PerturbedInt(123)},
                }
            )
            self.assertEqual(trainable2.algorithm_config.train_batch_size_per_learner, 123)
            # trainable2.config["train_batch_size_per_learner"] = 444
            # Usage of algorithm_state will log an error
            trainable2.load_checkpoint(ckpt)
            # perturbation of trainable2 is respected
            trainable2.stop()
            self.assertEqual(trainable2.algorithm_config.train_batch_size_per_learner, 123)
            self.assertIsInstance(trainable2.algorithm_config.train_batch_size_per_learner, PerturbedInt)
            trainable2b = setup.trainable_class(
                # NOTE: Normally should be the same but check that perturbed is used after load_checkpoint
                {"train_batch_size_per_learner": 123}
            )
            trainable2b.load_checkpoint(ckpt_perturbed)
            trainable2b.stop()
            self.assertEqual(trainable2b.algorithm_config.train_batch_size_per_learner, 222)
            trainable2c = setup.trainable_class(
                # NOTE: Normally should be the same but check that perturbed is used after load_checkpoint
                {
                    "train_batch_size_per_learner": 333,  # <-- this could be different but we assert it later
                    PERTURBED_HPARAMS: {"train_batch_size_per_learner": PerturbedInt(333)},
                }
            )
            trainable2c.load_checkpoint(ckpt_perturbed)
            self.assertEqual(trainable2c.algorithm_config.train_batch_size_per_learner, 333)

            # check that a perturbed checkpoint is restored:
            checkpoint2 = trainable2c.save_checkpoint(tmpdir3)
            trainable2c.stop()
            trainable3 = setup.trainable_class()
            self.assertEqual(trainable3.algorithm_config.train_batch_size_per_learner, 512)  # not yet loaded
            trainable3.load_checkpoint(checkpoint2)
            # Check that perturbed value was loaded
            self.assertEqual(trainable3.algorithm_config.train_batch_size_per_learner, 333)
            self.assertIsInstance(trainable3.algorithm_config.train_batch_size_per_learner, PerturbedInt)

    def test_further_subclassing(self):
        # for example by resource placement request
        setup = AlgorithmSetup()
        setup.trainable_class.use_pbar = not setup.trainable_class.use_pbar  # change from default value
        setup.trainable_class.discrete_eval = not setup.trainable_class.discrete_eval

        class SubclassedTrainable(setup.trainable_class):
            pass

        self.assertIs(SubclassedTrainable.setup_class, setup.trainable_class.setup_class)  # pyright: ignore[reportGeneralTypeIssues]
        self.assertIs(SubclassedTrainable.setup_class, setup)
        self.assertEqual(SubclassedTrainable._git_repo_sha, setup.trainable_class._git_repo_sha)
        if "GITHUB_REF" not in os.environ:
            self.assertNotEqual(SubclassedTrainable._git_repo_sha, "unknown")
        self.assertEqual(SubclassedTrainable.use_pbar, setup.trainable_class.use_pbar)
        self.assertEqual(SubclassedTrainable.discrete_eval, setup.trainable_class.discrete_eval)

    @mock_trainable_algorithm
    def test_total_steps_and_iterations(self):
        with patch_args("--batch_size", "2048", "--iterations", 100, "--fcnet_hiddens", "[1]"), MLPSetup() as setup:
            self.assertEqual(setup.args.iterations, 100)
            # do not check args.total_steps
        trainable = setup.trainable_class()
        self.assertEqual(trainable._total_steps["total_steps"], 2048 * 100)
        self.assertEqual(trainable._setup.args.iterations, 100)
        if "cli_args" in trainable.config:
            self.assertEqual(trainable.config["cli_args"]["iterations"], 100)
        trainable.stop()
        del setup

        with patch_args("--batch_size", "2048", "--dynamic_buffer", "--total_steps", "1_000_000"):
            setup2 = MLPSetup()

        trainable = setup2.trainable_class()
        budget = split_timestep_budget(
            total_steps=1_000_000,
            min_size=setup2.args.min_step_size,
            max_size=setup2.args.max_step_size,
            assure_even=True,
        )
        self.assertEqual(trainable._total_steps["total_steps"], budget["total_steps"])  # evened default value
        self.assertEqual(trainable._setup.args.iterations, budget["total_iterations"])
        trainable.stop()

    @mock_trainable_algorithm(mock_learner=False)
    def test_learner_class_changed(self):
        setup = PPOSetup()
        trainable = setup.trainable_class({"accumulate_gradients_every": 2})
        self.assertTrue(issubclass(trainable.algorithm_config.learner_class, PPOTorchLearnerWithGradientAccumulation))
        trainable = setup.trainable_class({"accumulate_gradients_every": 1})
        self.assertFalse(issubclass(trainable.algorithm_config.learner_class, PPOTorchLearnerWithGradientAccumulation))

    def test_evaluation_interval_consistency(self):
        """
        Verify that evaluation intervals are consistent across different step sizes.
        The requirement is that no matter the train_batch_size_per_learner (step size),
        the evaluation should happen at the same current_step interval.

        This means:
            step_size * evaluation_interval_iterations = constant_evaluation_step_interval
        However to evaluate smaller step sizes more frequently it is ok that the result is a
            divider of the constant interval, i.e.,
            small_step_size * evaluation_interval_iterations * n =  constant_evaluation_step_interval
        """
        # Setup Mock Algorithm & Config
        algo = mock.MagicMock()
        # Base evaluation interval (Power of 2 for clean testing)
        base_eval_interval = 16
        algo.config.evaluation_interval = base_eval_interval

        # Base batch size (e.g. 2048 - must be power of 2)
        base_batch_size = 2048
        algo.config.train_batch_size_per_learner = base_batch_size

        # Config required for budget calculation in DynamicEvalInterval
        algo.config.learner_config_dict = {
            "total_steps": 131072 * 10,  # Power of 2
            "min_dynamic_buffer_size": 128,
            "max_dynamic_buffer_size": 8192 * 2,
        }

        # Instantiate Callback
        callback = DynamicEvalInterval()

        # Initialize (triggers interval calculation)
        callback.on_algorithm_init(algorithm=algo, metrics_logger=mock.MagicMock())

        # Get the calculated intervals
        # _evaluation_intervals is a dict mapping step_size -> iterations_between_evaluations
        intervals = callback._evaluation_intervals
        print("Dynamic evaluation intervals are: %s", intervals)

        # Calculate the target evaluation interval in steps based on the base configuration
        target_step_interval = base_eval_interval * base_batch_size

        # We expect: iterations = nearest_power_of_2(sqrt(raw_iterations))
        # raw_iterations = target_step_interval / step_size

        # manual not automatic!
        # FIXME: 4096 hardcoded to 2 atm
        expected = {128: 16, 256: 16, 512: 8, 1024: 4, 2048: 4, 4096: 2, 8192: 2, 16384: 1}
        if expected:  # hardcoded expectation
            for step_size, iterations in intervals.items():
                if step_size in expected:
                    self.assertEqual(
                        iterations,
                        expected[step_size],
                        msg=f"Iterations for step size {step_size} should be {expected[step_size]}, "
                        f"but got {iterations}. All intervals: {intervals}",
                    )
                else:
                    self.fail(f"Step size {step_size} not found in expected results.")
        else:  # dynamic calc but without manual tweaks
            import numpy as np

            for step_size, iterations in intervals.items():
                if iterations == 0:
                    self.fail(f"Step size {step_size} resulted in 0 evaluation iterations, which is invalid.")

                # Problem? Dynamic is calculated based on batch_size 2048 as base
                raw_iterations = target_step_interval / step_size
                sqrt_iterations = np.sqrt(raw_iterations)

                if sqrt_iterations < 1:
                    expected_iterations = 1
                else:
                    expected_iterations = int(2 ** round(np.log2(sqrt_iterations)))
                    expected_iterations = max(1, expected_iterations)
                # Furthermore there is this custom rule - when there are not many iteration in the budget
                # for the step size we evaluate every step, e.g. 8192 every time and not every 2.
                try:
                    total_iters = callback._budget["iterations_per_step_size"][
                        callback._budget["step_sizes"].index(step_size)
                    ]
                    if total_iters <= 2 and expected_iterations > 1:
                        expected_iterations = 1
                except ValueError:
                    # wired batch size not in list
                    pass

                # if iterations == 1 and expected_iterations == 2:
                #    # this is okay
                #    continue

                self.assertEqual(
                    iterations,
                    expected_iterations,
                    msg=f"Iterations for step size {step_size} should be {expected_iterations} (raw {raw_iterations}, sqrt {sqrt_iterations:.2f}), but got {iterations}.",
                )


class TestClassCheckpointing(InitRay, TestHelpers, DisableLoggers, num_cpus=4):
    def setUp(self):
        super().setUp()

    @pytest.mark.env_runner_cases
    @Cases(ENV_RUNNER_CASES)
    def test_save_checkpoint(self, cases):
        # NOTE: In this test attributes are shared BY identity, this is just a weak test.
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners, fast_model=True)
            with tempfile.TemporaryDirectory() as tmpdir:
                # NOTE This loads some parts by identity!
                saved_ckpt = trainable.save_checkpoint(checkpoint_dir=tmpdir)
                saved_ckpt = deepcopy(saved_ckpt)  # assure to not compare by identity
                with patch_args(
                    "--num_env_runners", num_env_runners,
                ):  # fmt: skip # make sure that args do not influence the restore
                    trainable2 = self.TrainableClass()
                    trainable2.load_checkpoint(saved_ckpt)
            self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)
            trainable.stop()
            trainable2.stop()

    @pytest.mark.env_runner_cases
    @Cases(ENV_RUNNER_CASES)
    def test_save_restore_dict(self, cases):
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with self.subTest(num_env_runners=num_env_runners), tempfile.TemporaryDirectory() as tmpdir:
                training_result: _TrainingResult = trainable.save(tmpdir)  # pyright: ignore[reportInvalidTypeForm] # calls save_checkpoint
                with patch_args("--num_env_runners", num_env_runners):
                    trainable2 = self.TrainableClass()
                    with self.subTest("Restore trainable from dict"):
                        if _TrainingResult is not None:
                            self.assertIsInstance(training_result, _TrainingResult)
                        trainable2.restore(deepcopy(training_result))  # calls load_checkpoint
                        self.compare_trainables(trainable, trainable2, "from dict", num_env_runners=num_env_runners)
                        trainable2.stop()
            trainable.stop()

    @pytest.mark.env_runner_cases
    @Cases(ENV_RUNNER_CASES)
    def test_save_restore_path(self, cases):
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            with self.subTest(num_env_runners=num_env_runners), tempfile.TemporaryDirectory() as tmpdir:
                trainable.save(tmpdir)  # calls save_checkpoint
                with patch_args(
                    "--num_env_runners", num_env_runners
                ):  # make sure that args do not influence the restore
                    trainable3 = self.TrainableClass()
                    self.assertIsInstance(tmpdir, str)
                    trainable3.restore(tmpdir)  # calls load_checkpoint
                    self.compare_trainables(trainable, trainable3, "from path", num_env_runners=num_env_runners)
                    trainable3.stop()
            trainable.stop()

    def test_fork_from_restore(self):
        num_env_runners = 0
        trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
        with self.subTest(num_env_runners=num_env_runners):
            with tempfile.TemporaryDirectory() as tmpdir:
                trainable.save(tmpdir)  # calls save_checkpoint
                # make sure that args do not influence the restore
                with patch_args("--num_env_runners", num_env_runners):
                    trainable3 = self.TrainableClass({FORK_FROM: "something"})
                    self.assertIsInstance(tmpdir, str)
                    trainable3.restore(tmpdir)  # calls load_checkpoint
                    self.compare_trainables(trainable, trainable3, "from path", num_env_runners=num_env_runners)
                    trainable3.stop()
            trainable.stop()

    @pytest.mark.env_runner_cases
    @pytest.mark.basic
    @Cases(ENV_RUNNER_CASES)
    def test_1_get_set_state(self, cases):
        # If this test fails all others will most likely fail too, run it first.
        self.maxDiff = None
        for num_env_runners in iter_cases(cases):
            try:
                trainable, _ = self.get_trainable(num_env_runners=num_env_runners, fast_model=True)
                state = trainable.get_state()
                # For this test ignore the evaluation_interval set by DynamicEvalInterval callback
                # TODO: add no warning test
                self.assertIn(COMPONENT_ENV_RUNNER, state.get("algorithm", {}))

                # NOTE: If too many env_runners are created args.parallel is likely set to true,
                # due to parsing of test args.
                trainable2 = self.TrainableClass({"num_env_runners": num_env_runners})
                trainable2.set_state(deepcopy(state))
                self.on_checkpoint_loaded_callbacks(trainable2)

                # class is missing in config dict
                self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)
            finally:
                try:
                    trainable.stop()  # pyright: ignore[reportPossiblyUnboundVariable]
                    trainable2.stop()  # pyright: ignore[reportPossiblyUnboundVariable]
                except UnboundLocalError:
                    pass

    @Cases(ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases  # might deadlock with num_env_runners >= 2
    def test_safe_to_path(self, cases):
        """Test that the trainable can be saved to a path and restored."""
        for num_env_runners in iter_cases(cases):
            try:
                trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainable.save_to_path(tmpdir)
                    trainable2 = self.TrainableClass()
                    trainable2.restore_from_path(tmpdir)
                    # does not trigger on_checkpoint_load
                self.on_checkpoint_loaded_callbacks(trainable2)
                self.compare_trainables(trainable, trainable2, num_env_runners=num_env_runners)
            finally:
                trainable.stop()  # pyright: ignore[reportPossiblyUnboundVariable]
                trainable2.stop()  # pyright: ignore[reportPossiblyUnboundVariable]

    def test_validate_save_restore(self):
        """Basically test if TRAINING_ITERATION is set correctly."""
        # ray.init(include_dashboard=False, ignore_reinit_error=True)

        with patch_args(
            "--iterations", "5",
            "--total_steps", "320",
            "--use_exact_total_steps",
            "--batch_size", "64",
            "--minibatch_size", "32",
        ):  # fmt: skip
            # Need to fix argv for remote
            PPOTrainable = DefaultTrainable.define(PPOSetup.typed(), fix_argv=True, use_pbar=False)
            trainable = PPOTrainable()
            self.assertEqual(trainable._setup.args.iterations, 5)
            self.assertEqual(trainable._setup.args.total_steps, 320)
            validate_save_restore(PPOTrainable)
            trainable.stop()

    @Cases(ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases
    def test_interchange_save_checkpoint_restore_from_path(self, cases):
        """Test if methods can be used interchangeably."""
        # NOTE: restore_from_path currently does not set (local) env_runner state when num_env_runners > 0
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_checkpoint -> restore_from_path", num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir1:
                    trainable.save_checkpoint(tmpdir1)
                    trainable_from_path = self.TrainableClass()
                    trainable_from_path.restore_from_path(tmpdir1)
                    self.on_checkpoint_loaded_callbacks(trainable_from_path)
                self.compare_trainables(
                    trainable,
                    trainable_from_path,
                    "save_checkpoint -> restore_from_path",
                    num_env_runners=num_env_runners,
                )
            trainable.stop()
            trainable_from_path.stop()

    @Cases(ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases
    def test_interchange_save_checkpoint_from_checkpoint(self, cases):
        """Test if methods can be used interchangeably."""
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_checkpoint -> from_checkpoint", num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir1:
                    trainable.save_checkpoint(tmpdir1)
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
                trainable_from_checkpoint.stop()
            trainable.stop()

    @Cases(ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases
    def test_interchange_save_to_path_restore_from_path(self, cases):
        """Test if methods can be used interchangeably."""
        # NOTE: restore_from_path currently does not set (local) env_runner state when num_env_runners > 0
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_to_path -> restore_from_path", num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir1:
                    trainable.save_to_path(tmpdir1)
                    trainable_from_path = self.TrainableClass()
                    module = trainable_from_path.algorithm.get_module()
                    assert module
                    # FIXME: model_config is partially based on cls_model_config!
                    self.assertEqual(
                        module.model_config["fcnet_hiddens"],  # pyright: ignore
                        [self._model_config["fcnet_hiddens"][0]]  # pyright: ignore
                        if isinstance(self._model_config["fcnet_hiddens"], int)  # pyright: ignore
                        else self._model_config["fcnet_hiddens"],  # pyright: ignore
                    )
                    trainable_from_path.restore_from_path(tmpdir1)
                self.on_checkpoint_loaded_callbacks(trainable_from_path)
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

    @Cases(ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases
    def test_interchange_save_to_path_from_checkpoint(self, cases):
        """Test if methods can be used interchangeably."""
        for num_env_runners in iter_cases(cases):
            trainable, _ = self.get_trainable(num_env_runners=num_env_runners)
            # Save to save_checkpoint saves less data; i.e. no class and kwargs
            with self.subTest("save_to_path -> from_checkpoint", num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir1:
                    trainable.save_to_path(tmpdir1)
                    trainable_from_checkpoint: DefaultTrainable = self.TrainableClass.from_checkpoint(tmpdir1)
                self.compare_trainables(
                    trainable,
                    trainable_from_checkpoint,
                    "save_to_path -> from_checkpoint",
                    num_env_runners=num_env_runners,
                    ignore_env_runner_state=num_env_runners > 0,
                )
            trainable.stop()
            trainable_from_checkpoint.stop()

    @Cases(ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases
    def test_restore_multiprocessing(self, cases):
        from test._mp_trainable import remote_process  # noqa: PLC0415

        self._disable_save_model_architecture_callback_added.stop()  # remote is not mocked
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
                self.on_checkpoint_loaded_callbacks(trainable_restored)
                # Compare with default trainable:
                print(f"Create new default trainable num_env_runners={num_env_runners}")
                self._disable_tune_loggers.start()
                trainable, _ = self.get_trainable(
                    num_env_runners=num_env_runners, env_seed=data["env_seed"], eval_interval=None
                )
                self.compare_trainables(
                    trainable, trainable_restored, num_env_runners=num_env_runners, ignore_timers=True
                )
                trainable.stop()
                # Pickling with pickle and cloudpickle does not work here
                if pickled_trainable is not None:
                    print("Comparing restored trainable with pickled trainable")

                    trainable_restored2: DefaultTrainable = cloudpickle.loads(pickled_trainable)
                    self.on_checkpoint_loaded_callbacks(trainable_restored2)
                    self.compare_trainables(
                        trainable_restored,  # <-- need new trainable here
                        trainable_restored2,
                        num_env_runners=num_env_runners,
                    )
                    trainable_restored2.stop()
                trainable_restored.stop()

    @Cases(ENV_RUNNER_CASES)
    @pytest.mark.env_runner_cases
    @pytest.mark.tuner
    @pytest.mark.length(speed="medium")  # 2-3 min
    @pytest.mark.timeout(440)
    def test_tuner_checkpointing(self, cases):
        # self.enable_loggers()
        # self.no_pbar_updates()
        with patch_args(
            "--num_samples", "1",
            "--num_jobs", "1",
            "--iterations", "3",
            "--use_exact_total_steps",  # Do not adjust total_steps
            "--batch_size", make_divisible(32, DefaultArgumentParser.num_envs_per_env_runner),
            "--minibatch_size", make_divisible(16, DefaultArgumentParser.num_envs_per_env_runner),
            "--num_envs_per_env_runner", 1,  # Stuck when using more
            "--fcnet_hiddens", "[8]",
        ):  # fmt: skip
            for num_env_runners in iter_cases(cases):
                with self.subTest(num_env_runners=num_env_runners):
                    setup = MLPSetup(init_trainable=False)
                    setup.config.env_runners(num_env_runners=num_env_runners)
                    setup.config.training(
                        minibatch_size=make_divisible(32, DefaultArgumentParser.num_envs_per_env_runner)
                    )  # insert some noise
                    setup.create_trainable()
                    setup.trainable_class.use_pbar = False
                    tuner = setup.create_tuner()
                    assert tuner._local_tuner
                    tuner._local_tuner.get_run_config().checkpoint_config = tune.CheckpointConfig(
                        checkpoint_score_attribute=EVAL_METRIC_RETURN_MEAN,
                        checkpoint_score_order="max",
                        checkpoint_frequency=1,  # Save every iteration
                        # NOTE: num_keep does not appear to work here
                    )
                    result = tuner.fit()
                    self.check_tune_result(result)
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
                        # Account for dynamic eval changes
                        trainable_from_path._call_on_setup_callbacks()
                        self.sync_eval_interval_to_env_runners(trainable_from_path)
                        # doesn't call on_checkpoint_load but loads config correctly
                        trainable_from_ckpt: DefaultTrainable = DefaultTrainable.define(setup).from_checkpoint(
                            checkpoint
                        )
                        self.sync_eval_interval_to_env_runners(trainable_from_ckpt)
                        # restore is bad if algorithm_checkpoint_dir is a temp dir
                        self.assertEqual(trainable_from_ckpt.algorithm.iteration, step)
                        self.assertEqual(trainable_from_path.algorithm.iteration, step)
                        self.compare_trainables(
                            trainable_from_path,
                            trainable_from_ckpt,
                            "compare from_path with from_checkpoint",
                            iteration_after_step=step + 1,
                            step=step,
                            minibatch_size=make_divisible(32, DefaultArgumentParser.num_envs_per_env_runner),
                        )
                        trainable_from_ckpt.stop()
                        trainable_restore = DefaultTrainable.define(setup)()
                        # Problem restore uses load_checkpoint, which passes a dict to load_checkpoint
                        # however the checkpoint dir is unknown inside the loaded dict
                        trainable_restore.restore(checkpoint)
                        # With DynamicEvalInterval the configs are out of sync:
                        self.sync_eval_interval_to_env_runners(trainable_restore)
                        self.assertEqual(trainable_restore.algorithm.iteration, step)
                        self.assertIsInstance(checkpoint, str)
                        # load a second time to test as well
                        # NOTE: reuse: Currently this does not set some states correctly on the metrics_logger
                        # https://github.com/ray-project/ray/issues/55248, larger batch_size should fix it
                        # HACK: Only one might contain mean/max/min stats for env_runners--(module/agent)episode_return
                        # Should be fixed in 2.50
                        trainable_from_path.algorithm.metrics.reset()  # pyright: ignore[reportOptionalMemberAccess]
                        assert trainable_from_path.algorithm.learner_group is not None
                        assert trainable_from_path.algorithm.learner_group._learner is not None
                        trainable_from_path.algorithm.learner_group._learner.config._is_frozen = (
                            False  # HACK; why does this error appear now?
                        )
                        trainable_from_path.restore_from_path(checkpoint)
                        self.on_checkpoint_loaded_callbacks(trainable_from_path)
                        self.compare_trainables(
                            trainable_restore,
                            trainable_from_path,
                            "compare trainable_restore with from_path x2",
                            iteration_after_step=step + 1,
                            step=step,
                            minibatch_size=make_divisible(32, DefaultArgumentParser.num_envs_per_env_runner),
                        )
                        trainable_from_path.stop()
                        trainable_restore.stop()

    @pytest.mark.xfail(reason="compare_trainables not applicable")
    def test_reset_config(self):
        """Test reset_config method with two different configurations."""
        # Create two different setups with different arg combinations
        with patch_args(
            "--total_steps", "200",
            "--train_batch_size_per_learner", "32",
            "--num_env_runners", "0",
            "--minibatch_size", "16",
            "--use_exact_total_steps",
            "--num_envs_per_env_runner", "1",
        ):  # fmt: skip
            setup1 = PPOSetup(init_param_space=True, init_trainable=False)
            config1 = setup1.sample_params()

            # Verify patched args were applied on setup1
            self.assertEqual(setup1.args.total_steps, 200)
            self.assertEqual(setup1.args.train_batch_size_per_learner, 32)
            self.assertEqual(setup1.args.num_env_runners, 0)
            self.assertEqual(config1["cli_args"]["train_batch_size_per_learner"], 32)

        with patch_args(
            "--total_steps", "400",
            "--train_batch_size_per_learner", "64",
            "--num_env_runners", "1",
            "--minibatch_size", "32",
            "--use_exact_total_steps",
            "--num_envs_per_env_runner", "1",
        ):  # fmt: skip
            setup2 = PPOSetup(init_param_space=True, init_trainable=False)
            config2 = setup2.sample_params()

            # Verify patched args were applied on setup2
            self.assertEqual(setup2.args.total_steps, 400)
            self.assertEqual(setup2.args.train_batch_size_per_learner, 64)
            self.assertEqual(setup2.args.num_env_runners, 1)
            self.assertEqual(config2["cli_args"]["train_batch_size_per_learner"], 64)

        # Create trainables with the sampled configs
        # FIXME?: Not using define here ignores the cli_args in the config!
        TrainableClass1 = DefaultTrainable.define(
            setup1, model_config={"fcnet_hiddens": [self._fast_model_fcnet_hiddens], "head_fcnet_hiddens": []}
        )
        TrainableClass2 = DefaultTrainable.define(
            setup2, model_config={"fcnet_hiddens": [self._fast_model_fcnet_hiddens], "head_fcnet_hiddens": []}
        )
        trainable1 = TrainableClass1(config1)
        trainable2 = TrainableClass2(config2)
        self._created_trainables.extend([trainable1, trainable2])

        # Verify configs were applied correctly on trainables
        self.assertEqual(trainable1.algorithm_config.train_batch_size_per_learner, 32)
        self.assertEqual(trainable1.algorithm_config.num_env_runners, 0)
        self.assertEqual(trainable1.algorithm_config.minibatch_size, 16)

        self.assertEqual(trainable2.algorithm_config.train_batch_size_per_learner, 64)
        self.assertEqual(trainable2.algorithm_config.num_env_runners, 1)
        self.assertEqual(trainable2.algorithm_config.minibatch_size, 32)

        # Train both trainables once
        result1 = trainable1.train()
        result2 = trainable2.train()

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)

        # Store original config of trainable1 for comparison
        original_config1 = trainable1.algorithm_config.to_dict()

        # Call reset_config on trainable1 with config2
        checkpoint = trainable2.save()
        # Perturb the config
        config_restore = deepcopy(config2)
        config_restore["cli_args"]["total_steps"] = 8000
        config_restore["cli_args"]["log_stats"] = "all"
        config_restore["__extra_test_key"] = True
        config_restore["lr"] = 0.9997
        config_restore["env_seed"] = 98765
        reset_success = trainable1.reset(config_restore)
        trainable1.restore(checkpoint)
        if checkpoint.checkpoint:
            shutil.rmtree(checkpoint.checkpoint.path)
        self.assertEqual(trainable1.config["cli_args"]["total_steps"], 8000)
        self.assertEqual(trainable1.config["cli_args"]["log_stats"], "all")
        self.assertTrue(trainable1.config.get("__extra_test_key", False))
        self.assertEqual(trainable1.algorithm_config.lr, 0.9997)
        self.assertEqual(trainable1._setup.config.minibatch_size, 32)
        self.assertEqual(trainable1._setup.args.minibatch_size, 32)

        # Verify reset was successful
        self.assertTrue(reset_success)

        # Verify trainable1 now has trainable2's configuration
        self.assertEqual(trainable1.algorithm_config.train_batch_size_per_learner, 64)
        self.assertEqual(trainable1.algorithm_config.num_env_runners, 1)
        self.assertEqual(trainable1.algorithm_config.minibatch_size, 32)

        assert trainable1.algorithm.learner_group is not None
        lr_data = trainable1.algorithm.learner_group.foreach_learner(
            lambda lrn, x=None: (lrn.config.lr, lrn.config.learner_config_dict)
        ).result_or_errors
        for data in lr_data:
            lr, learner_config = data.get()  # pyright: ignore[reportGeneralTypeIssues]
            self.assertEqual(lr, 0.9997)
            self.assertEqual(learner_config["total_steps"], 8000)
        assert trainable1.algorithm.env_runner_group is not None
        env_runner_data = trainable1.algorithm.env_runner_group.foreach_env_runner(lambda r: r.config.to_dict())
        self.assertEqual(trainable1.algorithm.env_runner_group.num_remote_env_runners(), 1)
        for runner_config in env_runner_data:
            # self.assertEqual(runner_config["env_seed"], 98765)  # default env_seed is 42
            self.assertEqual(runner_config["num_env_runners"], 1)
            self.assertEqual(
                runner_config["_train_batch_size_per_learner"], config2["cli_args"]["train_batch_size_per_learner"]
            )

        # Verify the config actually changed
        new_config1 = trainable1.algorithm_config.to_dict()
        # When using to_dict train_batch_size_per_learner will be a private _underscore key
        self.assertNotEqual(
            original_config1["_train_batch_size_per_learner"],
            new_config1["_train_batch_size_per_learner"],
        )
        self.assertNotEqual(original_config1["num_env_runners"], new_config1["num_env_runners"])

        # Cannot compare trainables as currently different model states, need load checkpoint first.
        for conf in (trainable2.algorithm.learner_group._learner.config, trainable2.algorithm_config):  # pyright: ignore[reportOptionalMemberAccess]
            with type(setup2).open_config(conf):  # pyright: ignore[reportArgumentType]
                conf.learner_config_dict["total_steps"] = 8000  # Not updated on remotes
                conf.lr = 0.9997  # Not updated on remotes

        def set_lr(r):
            object.__setattr__(r.config, "lr", 0.9997)
            r.config.learner_config_dict["total_steps"] = 8000
            r.config.callbacks_on_environment_created.env_seed = 98765

        trainable2.algorithm.env_runner_group.foreach_env_runner(set_lr)  # pyright: ignore[reportOptionalMemberAccess]

        trainable2._setup.args = SimpleNamespace(**setup2.clean_args_to_hparams(trainable2._setup.args))  # pyright: ignore[reportAttributeAccessIssue]
        trainable2._setup.args.total_steps = 8000
        trainable2._setup.args.log_stats = "all"
        assert TrainableClass2.cls_model_config
        # trainable2._setup.config.model_config.update(TrainableClass2.cls_model_config) # pyright: ignore[reportCallIssue]
        if "--udiscovery" not in sys.argv:
            self.compare_trainables(trainable1, trainable2, "after reset_config", minibatch_size=32)
        # Train again to ensure trainable1 works with new config
        # result1_after_reset = trainable1.train()
        # self.assertIsNotNone(result1_after_reset)
        # Cleanup
        trainable1.stop()
        trainable2.stop()

    @skip("TODO implement")
    def check_dynamic_settings_on_reload(self):
        # check _get_global_step on reload
        ...


if __name__ == "__main__":
    import unittest

    unittest.main()
