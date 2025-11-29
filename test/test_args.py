# ruff: noqa: FBT003  # positional bool

from __future__ import annotations

import argparse
import io
import sys
import unittest
import warnings
from contextlib import redirect_stderr
from inspect import isclass

import pytest
from typing_extensions import TYPE_CHECKING, get_args

from experiments.create_tune_parameters import default_distributions, write_distributions_to_json
from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import DynamicEvalInterval
from ray_utilities.callbacks.algorithm.exact_sampling_callback import exact_sampling_callback
from ray_utilities.callbacks.algorithm.reset_episode_metrics import reset_episode_metrics_each_iteration
from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser, LogStatsChoices
from ray_utilities.connectors.remove_masked_samples_connector import RemoveMaskedSamplesConnector
from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations, split_timestep_budget
from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import PPOTorchLearnerWithGradientAccumulation
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.ppo_mlp_setup import MLPSetup
from ray_utilities.testing_utils import DisableLoggers, SetupLowRes, SetupWithEnv, mock_trainable_algorithm, patch_args
from ray_utilities.training.helpers import is_algorithm_callback_added, make_divisible

if TYPE_CHECKING:
    from ray_utilities.config.parser.pbt_scheduler_parser import PopulationBasedTrainingParser


pytestmark = pytest.mark.basic


class TestExtensionsAdded(SetupWithEnv, SetupLowRes, DisableLoggers):
    @patch_args()
    def test_patch_args(self):
        with patch_args("--no_exact_sampling"):
            self.assertIn(
                "--no_exact_sampling",
                sys.argv,
                "Expected --no_exact_sampling to be in sys.argv when patch_args is used.",
            )

    @patch_args()
    @mock_trainable_algorithm
    def test_exact_sampling_callback_added(self):
        setup = self._DEFAULT_SETUP_LOW_RES
        self.assertFalse(setup.args.no_exact_sampling)
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                assert config.callbacks_on_sample_end is not None
                self.assertTrue(
                    exact_sampling_callback is config.callbacks_on_sample_end
                    or exact_sampling_callback in config.callbacks_on_sample_end,  # pyright: ignore
                    "Expected exact_sampling_callback to be in callbacks_on_sample_end "
                    "when --no_exact_sampling is not set.",
                )

        setup = self._create_low_res_setup("--no_exact_sampling", init_trainable=False)
        self.assertTrue(
            setup.args.no_exact_sampling,
            "Expected no_exact_sampling to be False when --no_exact_sampling is not set.",
        )

    @unittest.skip("Decide on default value on argument")
    # @patch_args()
    # @mock_trainable_algorithm
    def test_episode_metrics_removed_callback_added(self):
        setup = self._DEFAULT_SETUP_LOW_RES
        # self.assertFalse(setup.args.limit_episode_metrics_to_iteration)
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                assert config.callbacks_on_sample_end is not None
                self.assertTrue(
                    reset_episode_metrics_each_iteration is config.callbacks_on_sample_end
                    or reset_episode_metrics_each_iteration in config.callbacks_on_sample_end,  # pyright: ignore
                    "Expected reset_episode_metrics_each_iteration to be in callbacks_on_sample_end "
                    "when --limit_episode_metrics_to_iteration is not set.",
                )
        return
        setup = self._create_low_res_setup("--limit_episode_metrics_to_iteration", init_trainable=False)
        self.assertTrue(
            setup.args.limit_episode_metrics_to_iteration,
            "Expected limit_episode_metrics_to_iteration to be True when --limit_episode_metrics_to_iteration is set.",
        )

    @patch_args()
    @mock_trainable_algorithm(mock_learner=False)
    def test_remove_masked_samples_added(self):
        setup = AlgorithmSetup(init_trainable=False)
        setup.config.environment(observation_space=self._OBSERVATION_SPACE, action_space=self._ACTION_SPACE)
        self.assertFalse(setup.args.keep_masked_samples)
        setup.create_trainable()
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertTrue(
                    issubclass(config.learner_class, RemoveMaskedSamplesLearner),
                    "Expected learner_class to be a subclass of RemoveMaskedSamplesLearner "
                    "when --keep_masked_samples is not set.",
                )
                learner = config.learner_class(config=config)
                learner.build()
                # Check that there is a RemoveMaskedSamplesConnector in the learner's connector pipeline
                self.assertTrue(
                    learner._learner_connector
                    and any(
                        isinstance(connector, RemoveMaskedSamplesConnector) for connector in learner._learner_connector
                    )
                )

        setup = self._create_low_res_setup("--keep_masked_samples", init_trainable=False)
        setup.config.environment(observation_space=self._OBSERVATION_SPACE, action_space=self._ACTION_SPACE)
        self.assertTrue(
            setup.args.keep_masked_samples,
            "Expected keep_masked_samples to be True when --keep_masked_samples is set.",
        )
        setup.create_trainable()
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertFalse(
                    issubclass(config.learner_class, RemoveMaskedSamplesLearner),
                    "Expected learner_class to not be a subclass of RemoveMaskedSamplesLearner "
                    "when --keep_masked_samples is set.",
                )
                learner = config.learner_class(config=config)
                learner.build()
                # Check that there is no RemoveMaskedSamplesConnector in the learner's connector pipeline
                if learner._learner_connector is not None:
                    self.assertFalse(
                        any(
                            isinstance(connector, RemoveMaskedSamplesConnector)
                            for connector in learner._learner_connector
                        )
                    )

    @patch_args()
    @mock_trainable_algorithm
    def test_dynamic_buffer_added(self):
        setup = self._DEFAULT_SETUP_LOW_RES
        self.assertFalse(setup.args.dynamic_buffer)
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertFalse(is_algorithm_callback_added(config, DynamicBufferUpdate))

        setup = self._create_low_res_setup("--dynamic_buffer")
        self.assertTrue(
            setup.args.dynamic_buffer,
            "Expected dynamic_buffer to be True when --dynamic_buffer is set.",
        )
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertTrue(
                    is_algorithm_callback_added(config, DynamicBufferUpdate),
                    "Expected DynamicBufferUpdate to be in callbacks_class when --dynamic_buffer is set.",
                )

    @patch_args()
    @mock_trainable_algorithm(mock_learner=False)
    def test_dynamic_batch_added(self):
        setup = self._DEFAULT_SETUP_LOW_RES
        self.assertFalse(setup.args.dynamic_buffer)
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertFalse(is_algorithm_callback_added(config, DynamicGradientAccumulation))
                # Test that also the Supporting learner is added
                self.assertFalse(issubclass(config.learner_class, PPOTorchLearnerWithGradientAccumulation))

        setup = self._create_low_res_setup("--dynamic_batch")
        self.assertTrue(
            setup.args.dynamic_batch,
            "Expected dynamic_batch to be True when --dynamic_batch is set.",
        )
        trainable = setup.trainable_class()
        for config in (setup.config, trainable.algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertTrue(is_algorithm_callback_added(config, DynamicGradientAccumulation))
                self.assertTrue(
                    issubclass(config.learner_class, PPOTorchLearnerWithGradientAccumulation),
                    "Expected learner_class to be a subclass of PPOTorchLearnerWithGradientAccumulation "
                    "when --dynamic_batch is set.",
                )
        trainable.stop()

    @patch_args()
    @mock_trainable_algorithm()
    def test_dynamic_eval_interval_added(self):
        # Check DynamicEvalInterval is added as a default as well
        setup = self._DEFAULT_SETUP_LOW_RES
        trainable = setup.trainable_class()
        for config in (setup.config, trainable.algorithm_config):
            with self.subTest(
                "Test 1", config="setup.config" if config is setup.config else "trainable.algorithm_config"
            ):
                self.assertTrue(is_algorithm_callback_added(config, DynamicEvalInterval))
        trainable.stop()
        # Check that the DynamicEvalInterval is especially added when using dynamic methods
        for args in ("--dynamic_buffer", "--dynamic_batch"):
            with self.subTest(args=args):
                setup = self._create_low_res_setup(args)
                trainable = setup.trainable_class()
                for config in (setup.config, trainable.algorithm_config):
                    with self.subTest(
                        "Test 2", config="setup.config" if config is setup.config else "trainable.algorithm_config"
                    ):
                        self.assertTrue(
                            is_algorithm_callback_added(config, DynamicEvalInterval),
                            msg=f"Expected DynamicEvalInterval to be in callbacks_class when {args} is set.",
                        )
                trainable.stop()
        # Check that only one is added
        setup = self._create_low_res_setup("--dynamic_batch", "--dynamic_buffer")
        trainable = setup.trainable_class()
        for config in (setup.config, trainable.algorithm_config):
            with self.subTest(
                "Test 3", config="setup.config" if config is setup.config else "trainable.algorithm_config"
            ):
                self.assertTrue(
                    (
                        isinstance(config.callbacks_class, type)
                        and issubclass(config.callbacks_class, DynamicEvalInterval)
                    )
                    or (
                        isinstance(config.callbacks_class, (list, tuple))
                        and DynamicEvalInterval in config.callbacks_class
                        and config.callbacks_class.count(DynamicEvalInterval) == 1
                    )
                )
        trainable.stop()

    @patch_args()
    @mock_trainable_algorithm
    def test_eval_ema_metric_callback_added(self):
        """
        Test that EvalEMAMetricCallback is added to the algorithm's callbacks.
        """
        from ray_utilities.callbacks.algorithm.eval_ema_metric_callback import (
            EvalEMAMetricCallback,
            _PartialEvalEMAMetaCallback,
        )

        setup = self._DEFAULT_SETUP_LOW_RES
        trainable = setup.trainable_class()
        for config in (setup.config, trainable.algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                # Check if EvalEMAMetricCallback is present in callbacks_class
                if isinstance(config.callbacks_class, type):
                    self.assertTrue(
                        issubclass(config.callbacks_class, EvalEMAMetricCallback)
                        or isinstance(config.callbacks_class, _PartialEvalEMAMetaCallback),
                        "Expected EvalEMAMetricCallback to be in callbacks_class.",
                    )
                elif isinstance(config.callbacks_class, (list, tuple)):
                    self.assertTrue(
                        any(
                            cb is EvalEMAMetricCallback
                            or (
                                (isinstance(cb, type) and issubclass(cb, EvalEMAMetricCallback))
                                or isinstance(cb, _PartialEvalEMAMetaCallback)
                            )
                            for cb in config.callbacks_class
                        ),
                        "Expected EvalEMAMetricCallback to be in callbacks_class list.",
                    )
        self.assertTrue(any(isinstance(cb, EvalEMAMetricCallback) for cb in trainable.algorithm.callbacks))
        trainable.stop()


class TestProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        write_distributions_to_json(default_distributions, AlgorithmSetup.TUNE_PARAMETER_FILE)

    @patch_args("--batch_size", "64", "--minibatch_size", "128")
    def test_to_large_minibatch_size(self):
        """minibatch_size cannot be larger than train_batch_size_per_learner"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            setup = AlgorithmSetup(init_trainable=False)
        self.assertEqual(setup.args.minibatch_size, 64)
        self.assertEqual(setup.args.train_batch_size_per_learner, 64)
        self.assertEqual(setup.config.minibatch_size, 64)
        self.assertEqual(setup.config.train_batch_size_per_learner, 64)
        self.assertTrue(
            any(
                "minibatch_size 128 is greater than train_batch_size_per_learner 64" in str(warning.message)
                for warning in w
            ),
            "Expected a warning when minibatch_size is greater than train_batch_size_per_learner",
        )

    @patch_args("--minibatch_scale", "1.0", "--batch_size", "256", "--minibatch_size", "64")
    def test_minibatch_scale_full(self):
        """Test that minibatch_scale=1.0 sets minibatch_size equal to train_batch_size_per_learner"""
        setup = AlgorithmSetup(init_trainable=False)
        # The scale should override the explicit minibatch_size argument
        self.assertEqual(setup.args.minibatch_size, 256)
        self.assertEqual(setup.args.train_batch_size_per_learner, 256)
        self.assertEqual(setup.config.minibatch_size, 256)
        self.assertEqual(setup.config.train_batch_size_per_learner, 256)
        self.assertEqual(setup.args.minibatch_scale, 1.0)

    @patch_args("--minibatch_scale", "0.5", "--batch_size", "512", "--minibatch_size", "128")
    def test_minibatch_scale_half(self):
        """Test that minibatch_scale=0.5 scales minibatch_size to half of train_batch_size_per_learner"""
        setup = AlgorithmSetup(init_trainable=False)
        # The scale should override the explicit minibatch_size argument
        self.assertEqual(setup.args.minibatch_size, 256)  # 512 * 0.5 = 256
        self.assertEqual(setup.args.train_batch_size_per_learner, 512)
        self.assertEqual(setup.config.minibatch_size, 256)
        self.assertEqual(setup.config.train_batch_size_per_learner, 512)
        self.assertEqual(setup.args.minibatch_scale, 0.5)

    @patch_args("--batch_size", "512", "--minibatch_size", "128")
    def test_minibatch_scale_disabled(self):
        """Test that without minibatch_scale, minibatch_size stays independent"""
        setup = AlgorithmSetup(init_trainable=False)
        # Without the scale, minibatch_size should remain as set
        self.assertEqual(setup.args.minibatch_size, 128)
        self.assertEqual(setup.args.train_batch_size_per_learner, 512)
        self.assertEqual(setup.config.minibatch_size, 128)
        self.assertEqual(setup.config.train_batch_size_per_learner, 512)
        self.assertIsNone(setup.args.minibatch_scale)

    def test_log_stats(self):
        for choice in get_args(LogStatsChoices):
            with self.subTest(f"Testing log_stats with choice: {choice}"):
                with patch_args(
                    "--log_stats", choice,
                    "--minibatch_size", "8",
                    "--batch_size", "8",
                    "--fcnet_hiddens", "[1]",
                    "--num_envs_per_env_runner", "1",
                ):  # fmt: skip
                    with MLPSetup(init_trainable=False) as setup:
                        setup.config.num_epochs = 1
                    self.assertEqual(setup.args.log_stats, choice)
                    if isclass(setup.trainable):
                        _result = setup.trainable_class(setup.sample_params()).train()
                    else:  # Otherwise call trainable function
                        _result = setup.trainable(setup.sample_params())
            with (
                patch_args("--log_stats", "invalid_choice"),
                self.assertRaises(SystemExit) as context,
                redirect_stderr(io.StringIO()),
            ):
                AlgorithmSetup()
            self.assertIsInstance(context.exception.__context__, argparse.ArgumentError)

    def test_not_a_model_parameter_clean(self):
        with patch_args("--not_parallel", "--optimize_config", "--tune", "batch_size", "--num_jobs", 3):
            setup = AlgorithmSetup()
            removable_params = setup.parser.get_non_cli_args()  # pyright: ignore[reportAttributeAccessIssue]
            self.assertGreater(len(removable_params), 0)
            for param in removable_params:
                self.assertNotIn(param, setup.param_space, msg=f"Expected {param} to not be in param_space")

    def test_iterations_and_total_steps(self):
        with patch_args("--total_steps", "2000", "--batch_size", 1999, "--use_exact_total_steps"):
            args = DefaultArgumentParser().parse_args()
            self.assertEqual(args.total_steps, 2000)
            self.assertEqual(args.iterations, 2)
        min_step_size = 1024
        max_step_size = 2048
        budget = split_timestep_budget(
            total_steps=2048,
            min_size=min_step_size,
            max_size=max_step_size,
            assure_even=True,
        )
        self.assertEqual(budget["total_iterations"], 3)
        self.assertEqual(
            calculate_iterations(
                dynamic_buffer=False,
                batch_size=2048,  # not needed when dynamic
                total_steps=budget["total_steps"],  # 4096
                assure_even=True,
                min_size=min_step_size,
                max_size=max_step_size,
            ),
            2,
        )
        self.assertEqual(
            calculate_iterations(
                dynamic_buffer=True,
                batch_size=2048,  # not needed when dynamic
                total_steps=budget["total_steps"],  # 4096
                assure_even=True,
                min_size=min_step_size,
                max_size=max_step_size,
            ),
            3,
        )
        # because total steps 2048< 4096 (even auto value), args.iterations is 2 (2* 1024) and not 3 (2048 + 1024 * 2)
        with patch_args(
            "--total_steps", "2048",
            "--batch_size", 2048,
            "--min_step_size", min_step_size,
            "--max_step_size", max_step_size,
        ):  # fmt: skip
            args = DefaultArgumentParser().parse_args()
            self.assertEqual(args.total_steps, 4096)  # 4096
            self.assertEqual(args.iterations, 2)  # One step of 2048
        with patch_args(
            "--total_steps", "2048",
            "--batch_size", 2048,
            "--min_step_size", min_step_size,
            "--max_step_size", max_step_size,
            "--dynamic_buffer",
        ):  # fmt: skip
            args = DefaultArgumentParser().parse_args()
            self.assertEqual(args.total_steps, 4096)  # 4096
            self.assertEqual(args.iterations, 3)  # One step of 2048, two steps of 1024

    def test_class_patch_args(self):
        with patch_args():  # Highest priority
            # Default values
            self.assertListEqual(sys.argv[1:], ["-a", "no_actor_by_patch", "--log_level", "DEBUG"])
            with DefaultArgumentParser.patch_args():
                # The order of arguments might be changed by patch_args
                self.assertTrue(
                    sys.argv[1:] == ["-a", "no_actor_by_patch", "--log_level", "DEBUG"]
                    or sys.argv[1:] == ["--log_level", "DEBUG", "-a", "no_actor_by_patch"],
                    f"Not patched correctly: {sys.argv[1:]}",
                )
                args = DefaultArgumentParser().parse_args()
                self.assertEqual(args.comet, False)
                self.assertEqual(args.agent_type, "no_actor_by_patch")
                self.assertEqual(args.log_level, "DEBUG")

        with patch_args(log_level=None):  # Highest priority
            # Default values
            self.assertListEqual(sys.argv[1:], ["-a", "no_actor_by_patch"])
            with DefaultArgumentParser.patch_args():
                self.assertListEqual(sys.argv[1:], ["-a", "no_actor_by_patch"])
                args = DefaultArgumentParser().parse_args()
                self.assertEqual(args.comet, False)
                self.assertEqual(args.agent_type, "no_actor_by_patch")
                self.assertEqual(args.log_level, "INFO")  # Default value

        with patch_args("--comet"):  # Highest priority
            with DefaultArgumentParser.patch_args():
                args = DefaultArgumentParser().parse_args()
                self.assertEqual(args.comet, "online")
            args = DefaultArgumentParser().parse_args()
            self.assertEqual(args.comet, "online")

        with patch_args():  # Highest priority
            with DefaultArgumentParser.patch_args("--comet"):
                args = DefaultArgumentParser().parse_args()
                self.assertEqual(args.comet, "online")
            args = DefaultArgumentParser().parse_args()
            self.assertEqual(args.comet, False)

        with patch_args("--comet", "offline"):  # Highest priority
            with DefaultArgumentParser.patch_args():
                args = DefaultArgumentParser().parse_args()
                self.assertEqual(args.comet, "offline")
            args = DefaultArgumentParser().parse_args()
            self.assertEqual(args.comet, "offline")

        with patch_args():  # Highest priority
            with DefaultArgumentParser.patch_args("--comet", "offline"):
                args = DefaultArgumentParser().parse_args()
                self.assertEqual(args.comet, "offline")
            args = DefaultArgumentParser().parse_args()
            self.assertEqual(args.comet, False)
        with patch_args("--comet", "online"):  # Highest priority
            with DefaultArgumentParser.patch_args("--comet", "offline"):
                args = DefaultArgumentParser().parse_args()
                self.assertEqual(args.comet, "online")
            args = DefaultArgumentParser().parse_args()
            self.assertEqual(args.comet, "online")

        with patch_args("--no_exact_sampling"):  # Highest priority
            with DefaultArgumentParser.patch_args("--dynamic_buffer"):
                args = DefaultArgumentParser().parse_args()
                self.assertTrue(args.no_exact_sampling)
                self.assertTrue(args.dynamic_buffer)
            args = DefaultArgumentParser().parse_args()
            self.assertTrue(args.no_exact_sampling)
            self.assertFalse(args.dynamic_buffer)

        with patch_args("--total_steps", "100"):  # Highest priority
            with DefaultArgumentParser.patch_args("--iterations", "10"):
                args = DefaultArgumentParser().parse_args()
                self.assertTrue(args.total_steps, 100)
                self.assertTrue(args.iterations, 10)
            args = DefaultArgumentParser().parse_args()
            self.assertTrue(args.total_steps, 100)

    def test_patch_args_with_subparser_commands(self):
        with patch_args("--comet", "--no_exact_sampling", "-n", 4, "--log_level", "DEBUG"):
            with DefaultArgumentParser.patch_args(
                # main args for this experiment
                "--tune", "batch_size",
                # Meta / less influential arguments for the experiment.
                "--num_samples", 16,
                "--min_step_size", 64,
                "--max_step_size", 8192 * 2,
                "--tags", "tune-batch_size", "mlp",
                "--comment", "Default training run. Tune batch size",
                "--env_seeding_strategy", "same",
                # constant
                "-a", DefaultArgumentParser.agent_type,
                "--seed", "42",
                "--wandb", "offline+upload",
                "--comet", "offline+upload",
                "--log_level", "INFO",
                "--use_exact_total_steps",
                "pbt",
                "--perturbation_interval", "0.5",
            ):  # fmt: skip
                args = AlgorithmSetup().args
                self.assertListEqual(args.tune, ["batch_size"], f"is {args.tune}")  # pyright: ignore[reportArgumentType]
                self.assertEqual(args.max_step_size, 8192 * 2)
                self.assertEqual(args.tags, ["tune-batch_size", "mlp"])
                self.assertEqual(args.comment, "Default training run. Tune batch size")
                self.assertEqual(args.env_seeding_strategy, "same")
                self.assertEqual(args.seed, 42)
                self.assertEqual(args.wandb, "offline+upload")
                self.assertEqual(args.log_level, "DEBUG")
                self.assertIs(args.use_exact_total_steps, True)
                # superseeded by sys.argv
                self.assertEqual(args.num_samples, 4)
                self.assertEqual(args.comet, "online")
                self.assertIs(args.no_exact_sampling, True)
                expected_perturbation_interval = int(args.total_steps * 0.5)
                if expected_perturbation_interval % (8192 * 2) != 0:
                    # make divisible (round down)
                    expected_perturbation_interval = make_divisible(expected_perturbation_interval, 8192 * 2) - 8192 * 2
                self.assertEqual(args.command.perturbation_interval, expected_perturbation_interval)  # pyright: ignore[reportOptionalMemberAccess]
            args = AlgorithmSetup().args
            self.assertIs(args.use_exact_total_steps, False)
            self.assertIs(args.wandb, False)
            self.assertIs(args.tune, False)

    @mock_trainable_algorithm
    def test_ppo_args(self):
        params = "clip_param", "vf_clip_param", "entropy_coeff", "vf_loss_coeff"
        with patch_args("--tune", *params):
            setup = AlgorithmSetup()
            self.assertListEqual(setup.args.tune, list(params))
            for param in params:
                self.assertIn(param, setup.param_space, msg=f"Expected {param} to be in param_space")
            sampled = setup.sample_params()
            for param in params:
                self.assertIn(param, sampled, msg=f"Expected {param} to be in sampled params")
        trainable = setup.trainable_class(sampled)
        for param in params:
            self.assertEqual(
                getattr(trainable.algorithm_config, param),
                sampled[param],
                msg=f"Expected {param} in trainable.algorithm_config to be equal to sampled parameter {sampled[param]}",
            )
        trainable.stop()


class TestTagArgumentProcessing(unittest.TestCase):
    @patch_args("--tag:foo")
    def test_add_extra_tag(self):
        parser = DefaultArgumentParser().parse_args(known_only=True)
        self.assertEqual(len(parser.extra_args), 0)
        self.assertIn("foo", parser.tags)
        self.assertNotIn("--tag:foo", parser.extra_args)

    @patch_args("--tag:foo", "--tag:bar", "--tags", "baz")
    def test_add_multiple_extra_tags(self):
        parser = DefaultArgumentParser().parse_args(known_only=True)
        self.assertEqual(len(parser.extra_args), 0)
        self.assertIn("foo", parser.tags)
        self.assertIn("bar", parser.tags)
        self.assertIn("baz", parser.tags)

    @patch_args("--tag:foo:1", "--tag:foo:2", "--tag:bar:3")
    def test_remove_duplicated_subtags_colon(self):
        parser = DefaultArgumentParser().parse_args(known_only=True)
        self.assertEqual(len(parser.extra_args), 0)
        self.assertIn("foo:2", parser.tags)
        self.assertIn("bar:3", parser.tags)
        self.assertSetEqual({"bar:3", "foo:2"}, set(parser.tags))

    @patch_args("--tag:foo=2", "--tag:bar=3", "--tags", "foo=1", "--tag:foo=3")
    def test_remove_duplicated_subtags_equal(self):
        parser = DefaultArgumentParser().parse_args(known_only=True)
        self.assertEqual(len(parser.extra_args), 0)
        self.assertIn("foo=1", parser.tags)
        self.assertIn("bar=3", parser.tags)
        self.assertSetEqual({"bar=3", "foo=1"}, set(parser.tags))

    @patch_args("--tag:foo:1", "--tag:foo:2", "--tag:foo", "--tag:foo=3", "--tag:foo:4")
    def test_tags_priority(self):
        parser = DefaultArgumentParser().parse_args(known_only=True)
        self.assertEqual(len(parser.extra_args), 0)
        # Only one foo: or foo= should remain
        self.assertListEqual(parser.tags, ["foo", "foo:4"])

    @patch_args("pbt", "foo:1", "--tag:foo", "--tag:foo:2")
    def test_invalid_tag_format(self):
        stderr_out = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(stderr_out):
            DefaultArgumentParser().parse_args(known_only=False)
        parser = DefaultArgumentParser().parse_args(known_only=True)
        self.assertListEqual(["foo", "foo:2"], parser.tags)
        self.assertIn("foo:1", parser.extra_args)


class TestSubparsers(unittest.TestCase):
    def test_pbt_subparser(self):
        with patch_args():
            parser = DefaultArgumentParser[None]().parse_args()
            self.assertIsNone(parser.command)

        with patch_args("pbt"):
            parser = DefaultArgumentParser["PopulationBasedTrainingParser"]().parse_args()
            self.assertIsNotNone(parser.command)

        with patch_args("pbt", "--perturbation_interval", "0.5"):
            parser = DefaultArgumentParser["PopulationBasedTrainingParser"]().parse_args()
            self.assertEqual(parser.command.perturbation_interval, parser.total_steps * 0.5)  # pyright: ignore[reportAttributeAccessIssue]

        with patch_args("pbt", "--perturbation_interval", "0.5"):
            parser = DefaultArgumentParser["PopulationBasedTrainingParser"]().parse_args()
            self.assertEqual(parser.command.perturbation_interval, parser.total_steps * 0.5)  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    unittest.main()
