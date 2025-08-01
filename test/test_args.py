import argparse
import logging
import sys
import unittest
from inspect import isclass

from typing_extensions import get_args

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import DynamicEvalInterval
from ray_utilities.callbacks.algorithm.exact_sampling_callback import exact_sampling_callback
from ray_utilities.config.typed_argument_parser import LogStatsChoices
from ray_utilities.connectors.remove_masked_samples_connector import RemoveMaskedSamplesConnector
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import SetupDefaults, patch_args


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
                self.assertFalse(
                    config.callbacks_class is DynamicBufferUpdate
                    or (
                        isinstance(config.callbacks_class, type)
                        and issubclass(config.callbacks_class, DynamicBufferUpdate)
                    )
                    or (
                        isinstance(config.callbacks_class, (list, tuple))
                        and DynamicBufferUpdate in config.callbacks_class
                    )
                )

        with patch_args("--dynamic_buffer"):
            setup = AlgorithmSetup()
            self.assertTrue(
                setup.args.dynamic_buffer,
                "Expected dynamic_buffer to be True when --dynamic_buffer is set.",
            )
            for config in (setup.config, setup.trainable_class().algorithm_config):
                with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                    self.assertTrue(
                        config.callbacks_class is DynamicBufferUpdate
                        or (
                            isinstance(config.callbacks_class, type)
                            and issubclass(config.callbacks_class, DynamicBufferUpdate)
                        )
                        or (
                            isinstance(config.callbacks_class, (list, tuple))
                            and DynamicBufferUpdate in config.callbacks_class
                        ),
                        "Expected DynamicBufferUpdate to be in callbacks_class when --dynamic_buffer is set.",
                    )

    @patch_args()
    def test_dynamic_batch_added(self):
        setup = AlgorithmSetup()
        self.assertFalse(setup.args.dynamic_buffer)
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertFalse(
                    config.callbacks_class is DynamicGradientAccumulation
                    or (
                        isinstance(config.callbacks_class, type)
                        and issubclass(config.callbacks_class, DynamicGradientAccumulation)
                    )
                    or (
                        isinstance(config.callbacks_class, (list, tuple))
                        and DynamicGradientAccumulation in config.callbacks_class
                    )
                )

        with patch_args("--dynamic_batch"):
            setup = AlgorithmSetup()
            self.assertTrue(
                setup.args.dynamic_batch,
                "Expected dynamic_batch to be True when --dynamic_batch is set.",
            )
            for config in (setup.config, setup.trainable_class().algorithm_config):
                with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                    self.assertTrue(
                        config.callbacks_class is DynamicGradientAccumulation
                        or (
                            isinstance(config.callbacks_class, type)
                            and issubclass(config.callbacks_class, DynamicGradientAccumulation)
                        )
                        or (
                            isinstance(config.callbacks_class, (list, tuple))
                            and DynamicGradientAccumulation in config.callbacks_class
                        )
                    )

    @patch_args()
    def test_dynamic_eval_interval_added(self):
        # Check DynamicEvalInterval is not added by default
        setup = AlgorithmSetup()
        for config in (setup.config, setup.trainable_class().algorithm_config):
            with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                self.assertFalse(
                    (
                        isinstance(config.callbacks_class, type)
                        and issubclass(config.callbacks_class, DynamicEvalInterval)
                    )
                    or (
                        isinstance(config.callbacks_class, (list, tuple))
                        and DynamicEvalInterval in config.callbacks_class
                    ),
                )
        # Check that the DynamicEvalInterval is also added
        for args in ("--dynamic_buffer", "--dynamic_batch"):
            with patch_args(args), self.subTest(args=args):
                setup = AlgorithmSetup()
                for config in (setup.config, setup.trainable_class().algorithm_config):
                    with self.subTest("setup.config" if config is setup.config else "trainable.algorithm_config"):
                        self.assertTrue(
                            (
                                isinstance(config.callbacks_class, type)
                                and issubclass(config.callbacks_class, DynamicEvalInterval)
                            )
                            or (
                                isinstance(config.callbacks_class, (list, tuple))
                                and DynamicEvalInterval in config.callbacks_class
                            ),
                            msg=f"Expected DynamicEvalInterval to be in callbacks_class when {args} is set.",
                        )
        # Check that only one is added
        with patch_args("--dynamic_batch", "--dynamic_buffer"):
            setup = AlgorithmSetup()
            for config in (setup.config, setup.trainable_class().algorithm_config):
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


class TestProcessing(unittest.TestCase):
    @patch_args("--batch_size", "64", "--minibatch_size", "128")
    def test_to_large_minibatch_size(self):
        """minibatch_size cannot be larger than train_batch_size_per_learner"""
        from ray_utilities.config.typed_argument_parser import logger

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
                        _result = setup.trainable_class().train()
                    else:
                        _result = setup.trainable(setup.param_space)
            with patch_args("--log_stats", "invalid_choice"):
                with self.assertRaises(SystemExit) as context:
                    AlgorithmSetup()
                self.assertIsInstance(context.exception.__context__, argparse.ArgumentError)
