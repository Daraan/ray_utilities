from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from test._mp_trainable import remote_process
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

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN, FORK_FROM, PERTURBED_HPARAMS
from ray_utilities.dynamic_config.dynamic_buffer_update import split_timestep_budget
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
            with self.subTest("With parameters"):
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
                    "train_batch_size_per_learner": 333,
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
            min_size=32,
            max_size=4096 * 2,
            assure_even=True,
        )
        self.assertEqual(trainable._total_steps["total_steps"], budget["total_steps"])  # evened default value
        self.assertEqual(trainable._setup.args.iterations, budget["total_iterations"])
        trainable.stop()


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
                saved_ckpt = trainable.save_checkpoint(tmpdir)
                saved_ckpt = deepcopy(saved_ckpt)  # assure to not compare by identity
                with patch_args(
                    "--num_env_runners", num_env_runners,
                    "--no_dynamic_eval_interval",
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
            with self.subTest(num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir:
                    training_result: _TrainingResult = trainable.save(tmpdir)  # pyright: ignore[reportInvalidTypeForm] # calls save_checkpoint
                    with patch_args(
                        "--num_env_runners",
                        num_env_runners,
                        "--no_dynamic_eval_interval",
                    ):  # make sure that args do not influence the restore
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
            with self.subTest(num_env_runners=num_env_runners):
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainable.save(tmpdir)  # calls save_checkpoint
                    with patch_args(
                        "--num_env_runners",
                        num_env_runners,
                        "--no_dynamic_eval_interval",
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
                with patch_args(
                    "--num_env_runners", num_env_runners,
                    "--no_dynamic_eval_interval",
                ):  # fmt: skip
                    trainable3 = self.TrainableClass({FORK_FROM: "something"})
                    self.assertIsInstance(tmpdir, str)
                    trainable3.restore(tmpdir)  # calls load_checkpoint
                    self.compare_trainables(trainable, trainable3, "from path", num_env_runners=num_env_runners)
                    trainable3.stop()
            trainable.stop()

    @pytest.mark.env_runner_cases
    @pytest.mark.basic
    @Cases([0])
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
                with patch_args(
                    "--no_dynamic_eval_interval",
                ):
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
                    with patch_args("--no_dynamic_eval_interval"):
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
            PPOTrainable = DefaultTrainable.define(PPOSetup.typed(), fix_argv=True)
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
                    with patch_args("--no_dynamic_eval_interval"):
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
                    with patch_args("--no_dynamic_eval_interval"):
                        trainable_from_path = self.TrainableClass()
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
                    with patch_args():
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
                # Compare with default trainable:
                print(f"Create new default trainable num_env_runners={num_env_runners}")
                self._disable_tune_loggers.start()
                trainable, _ = self.get_trainable(num_env_runners=num_env_runners, env_seed=data["env_seed"])
                self.compare_trainables(
                    trainable, trainable_restored, num_env_runners=num_env_runners, ignore_timers=True
                )
                trainable.stop()
                # Pickling with pickle and cloudpickle does not work here
                if pickled_trainable is not None:
                    print("Comparing restored trainable with pickled trainable")

                    trainable_restored2 = cloudpickle.loads(pickled_trainable)
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
        ):  # fmt: skip
            for num_env_runners in iter_cases(cases):
                with self.subTest(num_env_runners=num_env_runners):
                    setup = AlgorithmSetup(init_trainable=False)
                    setup.config.env_runners(num_env_runners=num_env_runners)
                    setup.config.training(
                        minibatch_size=make_divisible(32, DefaultArgumentParser.num_envs_per_env_runner)
                    )  # insert some noise
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
                        # doesn't call on_checkpoint_load but loads config correctly # TODO store original eval_interval
                        trainable_from_ckpt: DefaultTrainable = DefaultTrainable.define(setup).from_checkpoint(
                            checkpoint
                        )
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
                        self.assertEqual(trainable_restore.algorithm.iteration, step)
                        self.assertIsInstance(checkpoint, str)
                        # load a second time to test as well
                        # NOTE: reuse: Currently this does not set some states correctly on the metrics_logger
                        # https://github.com/ray-project/ray/issues/55248, larger batch_size should fix it
                        # HACK: Only one might contain mean/max/min stats for env_runners--(module/agent)episode_return
                        # Should be fixed in 2.50
                        trainable_from_path.algorithm.metrics.reset()  # pyright: ignore[reportOptionalMemberAccess]
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

    @skip("TODO implement")
    def check_dynamic_settings_on_reload(self):
        # check _get_global_step on reload
        ...
