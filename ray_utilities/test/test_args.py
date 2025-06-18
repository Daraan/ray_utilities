import sys

from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import DynamicEvalInterval
from ray_utilities.callbacks.algorithm.exact_sampling_callback import exact_sampling_callback
from ray_utilities.connectors.remove_masked_samples_connector import RemoveMaskedSamplesConnector
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.test.utils import SetupDefaults, patch_args


@patch_args()
class TestExtensionsAdded(SetupDefaults):
    def test_patch_args(self):
        with patch_args("--no_exact_sampling"):
            self.assertIn(
                "--no_exact_sampling",
                sys.argv,
                "Expected --no_exact_sampling to be in sys.argv when patch_args is used.",
            )

    def test_exact_sampling_callback_added(self):
        setup = AlgorithmSetup()
        self.assertFalse(setup.args.no_exact_sampling)
        assert setup.config.callbacks_on_sample_end is not None
        self.assertTrue(
            exact_sampling_callback is setup.config.callbacks_on_sample_end
            or exact_sampling_callback in setup.config.callbacks_on_sample_end,  # pyright: ignore
            "Expected exact_sampling_callback to be in callbacks_on_sample_end when --no_exact_sampling is not set.",
        )

        with patch_args("--no_exact_sampling"):
            setup = AlgorithmSetup()
            self.assertTrue(
                setup.args.no_exact_sampling,
                "Expected no_exact_sampling to be False when --no_exact_sampling is not set.",
            )

    def test_remove_masked_samples_added(self):
        setup = AlgorithmSetup()
        setup.config.environment(observation_space=self._OBSERVATION_SPACE, action_space=self._ACTION_SPACE)
        self.assertFalse(setup.args.keep_masked_samples)
        self.assertTrue(
            issubclass(setup.config.learner_class, RemoveMaskedSamplesLearner),
            "Expected learner_class to be a subclass of RemoveMaskedSamplesLearner "
            "when --keep_masked_samples is not set.",
        )
        learner = setup.config.learner_class(config=setup.config)
        learner.build()
        # Check that there is a RemoveMaskedSamplesConnector in the learner's connector pipeline
        self.assertTrue(
            learner._learner_connector
            and any(isinstance(connector, RemoveMaskedSamplesConnector) for connector in learner._learner_connector)
        )

        with patch_args("--keep_masked_samples"):
            setup = AlgorithmSetup()
            setup.config.environment(observation_space=self._OBSERVATION_SPACE, action_space=self._ACTION_SPACE)
            self.assertTrue(
                setup.args.keep_masked_samples,
                "Expected keep_masked_samples to be True when --keep_masked_samples is set.",
            )
            self.assertFalse(
                issubclass(setup.config.learner_class, RemoveMaskedSamplesLearner),
                "Expected learner_class to not be a subclass of RemoveMaskedSamplesLearner "
                "when --keep_masked_samples is set.",
            )
            learner = setup.config.learner_class(config=setup.config)
            learner.build()
            # Check that there is no RemoveMaskedSamplesConnector in the learner's connector pipeline
            if learner._learner_connector is not None:
                self.assertFalse(
                    any(isinstance(connector, RemoveMaskedSamplesConnector) for connector in learner._learner_connector)
                )

    def test_dynamic_buffer_added(self):
        setup = AlgorithmSetup()
        self.assertFalse(setup.args.dynamic_buffer)
        self.assertFalse(
            setup.config.callbacks_class is DynamicBufferUpdate
            or (
                isinstance(setup.config.callbacks_class, type)
                and issubclass(setup.config.callbacks_class, DynamicBufferUpdate)
            )
            or (
                isinstance(setup.config.callbacks_class, (list, tuple))
                and DynamicBufferUpdate in setup.config.callbacks_class
            )
        )

        with patch_args("--dynamic_buffer"):
            setup = AlgorithmSetup()
            self.assertTrue(
                setup.args.dynamic_buffer,
                "Expected dynamic_buffer to be True when --dynamic_buffer is set.",
            )
            self.assertTrue(
                setup.config.callbacks_class is DynamicBufferUpdate
                or (
                    isinstance(setup.config.callbacks_class, type)
                    and issubclass(setup.config.callbacks_class, DynamicBufferUpdate)
                )
                or (
                    isinstance(setup.config.callbacks_class, (list, tuple))
                    and DynamicBufferUpdate in setup.config.callbacks_class
                )
            )

    def test_dynamic_batch(self):
        ...
        # In symbol by using gradient accumulation - here we can only modify minibatch size
        # or decouple rollout and what is passed to the learner - however same attribute.

    def test_dynamic_eval_interval_added(self):
        # Check DynamicEvalInterval is not added by default
        setup = AlgorithmSetup()
        self.assertFalse(
            (
                isinstance(setup.config.callbacks_class, type)
                and issubclass(setup.config.callbacks_class, DynamicEvalInterval)
            )
            or (
                isinstance(setup.config.callbacks_class, (list, tuple))
                and DynamicEvalInterval in setup.config.callbacks_class
            ),
        )
        # Check that the DynamicEvalInterval is also added
        for args in ("--dynamic_buffer", "--dynamic_batch"):
            with patch_args(args), self.subTest(args=args):
                setup = AlgorithmSetup()
                self.assertTrue(
                    (
                        isinstance(setup.config.callbacks_class, type)
                        and issubclass(setup.config.callbacks_class, DynamicEvalInterval)
                    )
                    or (
                        isinstance(setup.config.callbacks_class, (list, tuple))
                        and DynamicEvalInterval in setup.config.callbacks_class
                    ),
                    msg=f"Expected DynamicEvalInterval to be in callbacks_class when {args} is set.",
                )
        # Check that only one is added
        with patch_args("--dynamic_batch", "--dynamic_buffer"):
            setup = AlgorithmSetup()
            self.assertTrue(
                (
                    isinstance(setup.config.callbacks_class, type)
                    and issubclass(setup.config.callbacks_class, DynamicEvalInterval)
                )
                or (
                    isinstance(setup.config.callbacks_class, (list, tuple))
                    and DynamicEvalInterval in setup.config.callbacks_class
                    and setup.config.callbacks_class.count(DynamicEvalInterval) == 1
                )
            )
