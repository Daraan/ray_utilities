# ruff: noqa: FBT003  # positional bool

import argparse
import io
import logging
import sys
import unittest
from contextlib import redirect_stderr
from inspect import isclass

import pytest
from typing_extensions import get_args

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import DynamicEvalInterval
from ray_utilities.callbacks.algorithm.exact_sampling_callback import exact_sampling_callback
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser, LogStatsChoices
from ray_utilities.connectors.remove_masked_samples_connector import RemoveMaskedSamplesConnector
from ray_utilities.dynamic_config.dynamic_buffer_update import calculate_iterations, split_timestep_budget
from ray_utilities.learners.ppo_torch_learner_with_gradient_accumulation import PPOTorchLearnerWithGradientAccumulation
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import SetupDefaults, patch_args

pytestmark = pytest.mark.basic


class TestExtensionsAdded(SetupDefaults):
    @patch_args()
    def test_patch_args(self):
        with patch_args("--no_exact_sampling"):
            self.assertIn(
                "--no_exact_sampling",
                sys.argv,
                "Expected --no_exact_sampling to be in sys.argv when patch_args is used.",
            )

    @patch_args()
    def test_exact_sampling_callback_added(self):
        setup = AlgorithmSetup()
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

        with patch_args("--no_exact_sampling"):
            setup = AlgorithmSetup()
            self.assertTrue(
                setup.args.no_exact_sampling,
                "Expected no_exact_sampling to be False when --no_exact_sampling is not set.",
            )

    @patch_args()
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

        with patch_args("--keep_masked_samples"):
            setup = AlgorithmSetup(init_trainable=False)
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
    def test_dynamic_buffer_added(self):
        setup = AlgorithmSetup()
        self.assertFalse(setup.args.dynamic_buffer)
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertFalse(self.is_algorithm_callback_added(config, DynamicBufferUpdate))

        with patch_args("--dynamic_buffer"):
            setup = AlgorithmSetup()
            self.assertTrue(
                setup.args.dynamic_buffer,
                "Expected dynamic_buffer to be True when --dynamic_buffer is set.",
            )
            for config in (setup.config, setup.trainable_class().algorithm_config):
                with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                    self.assertTrue(
                        self.is_algorithm_callback_added(config, DynamicBufferUpdate),
                        "Expected DynamicBufferUpdate to be in callbacks_class when --dynamic_buffer is set.",
                    )

    @patch_args()
    def test_dynamic_batch_added(self):
        setup = AlgorithmSetup()
        self.assertFalse(setup.args.dynamic_buffer)
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertFalse(self.is_algorithm_callback_added(config, DynamicGradientAccumulation))
                # Test that also the Supporting learner is added
                self.assertFalse(issubclass(config.learner_class, PPOTorchLearnerWithGradientAccumulation))

        with patch_args("--dynamic_batch"):
            setup = AlgorithmSetup()
            self.assertTrue(
                setup.args.dynamic_batch,
                "Expected dynamic_batch to be True when --dynamic_batch is set.",
            )
            trainable = setup.trainable_class()
            for config in (setup.config, trainable.algorithm_config):
                with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                    self.assertTrue(self.is_algorithm_callback_added(config, DynamicGradientAccumulation))
                    self.assertTrue(issubclass(config.learner_class, PPOTorchLearnerWithGradientAccumulation))
            trainable.stop()

    @patch_args()
    def test_dynamic_eval_interval_added(self):
        # Check DynamicEvalInterval is not added by default
        setup = AlgorithmSetup()
        trainable = setup.trainable_class()
        for config in (setup.config, trainable.algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertFalse(self.is_algorithm_callback_added(config, DynamicEvalInterval))
        trainable.stop()
        # Check that the DynamicEvalInterval is also added
        for args in ("--dynamic_buffer", "--dynamic_batch"):
            with patch_args(args), self.subTest(args=args):
                setup = AlgorithmSetup()
                trainable = setup.trainable_class()
                for config in (setup.config, trainable.algorithm_config):
                    with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                        self.assertTrue(
                            self.is_algorithm_callback_added(config, DynamicEvalInterval),
                            msg=f"Expected DynamicEvalInterval to be in callbacks_class when {args} is set.",
                        )
                trainable.stop()
        # Check that only one is added
        with patch_args("--dynamic_batch", "--dynamic_buffer"):
            setup = AlgorithmSetup()
            trainable = setup.trainable_class()
            for config in (setup.config, trainable.algorithm_config):
                with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
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


class TestProcessing(unittest.TestCase):
    @patch_args("--batch_size", "64", "--minibatch_size", "128")
    def test_to_large_minibatch_size(self):
        """minibatch_size cannot be larger than train_batch_size_per_learner"""
        from ray_utilities.config.typed_argument_parser import logger  # noqa: PLC0415

        with self.assertLogs(
            logger,
            logging.WARNING,
        ) as ctx:
            setup = AlgorithmSetup(init_trainable=False)
        self.assertEqual(setup.args.minibatch_size, 64)
        self.assertEqual(setup.args.train_batch_size_per_learner, 64)
        self.assertEqual(setup.config.minibatch_size, 64)
        self.assertEqual(setup.config.train_batch_size_per_learner, 64)
        self.assertIn(
            "minibatch_size 128 is larger than train_batch_size_per_learner 64",
            ctx.output[0],
            "Expected an error log when minibatch_size is larger than train_batch_size_per_learner",
        )

    def test_log_stats(self):
        for choice in get_args(LogStatsChoices):
            with self.subTest(f"Testing log_stats with choice: {choice}"):
                with patch_args(
                    "--log_stats", choice, "--minibatch_size", "8", "--batch_size", "8", "--num_epochs", "1"
                ):
                    setup = AlgorithmSetup()
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
        with patch_args("--not_parallel", "--optimize_config", "--tune", "batch_size", "--num-jobs", 3):
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
            "--min_step_size",
            min_step_size,
            "--max_step_size",
            max_step_size,
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
            self.assertListEqual(sys.argv[1:], ["-a", "no_actor_provided_by_patch_args"])
            with DefaultArgumentParser.patch_args():
                self.assertListEqual(sys.argv[1:], ["-a", "no_actor_provided_by_patch_args"])
                args = DefaultArgumentParser().parse_args()
                self.assertEqual(args.comet, False)
                self.assertEqual(args.agent_type, "no_actor_provided_by_patch_args")

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

        with patch_args("--comet", "--no_exact_sampling", "-n", 4):
            with DefaultArgumentParser.patch_args(
                # main args for this experiment
                "--tune", "batch_size",
                # Meta / less influential arguments for the experiment.
                "--num_samples", 16,
                "--max_step_size", 16_000,
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
            ):  # fmt: skip
                args = AlgorithmSetup().args
                self.assertListEqual(args.tune, ["batch_size"])  # pyright: ignore[reportArgumentType]
                self.assertEqual(args.max_step_size, 16_000)
                self.assertEqual(args.tags, ["tune-batch_size", "mlp"])
                self.assertEqual(args.comment, "Default training run. Tune batch size")
                self.assertEqual(args.env_seeding_strategy, "same")
                self.assertEqual(args.seed, 42)
                self.assertEqual(args.wandb, "offline+upload")
                self.assertEqual(args.log_level, "INFO")
                self.assertIs(args.use_exact_total_steps, True)
                # superseeded by sys.argv
                self.assertEqual(args.num_samples, 4)
                self.assertEqual(args.comet, "online")
                self.assertIs(args.no_exact_sampling, True)
            args = AlgorithmSetup().args
            self.assertIs(args.use_exact_total_steps, False)
            self.assertIs(args.wandb, False)
            self.assertIs(args.tune, False)
