import sys
from unittest import mock

from ray_utilities.callbacks.algorithm.exact_sampling_callback import exact_sampling_callback
from ray_utilities.connectors.remove_masked_samples_connector import RemoveMaskedSamplesConnector
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from ray_utilities.test.utils import SetupDefaults, patch_args
from ray_utilities.setup.algorithm_setup import AlgorithmSetup


@patch_args()
class TestArgs(SetupDefaults):
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
            any(isinstance(connector, RemoveMaskedSamplesConnector) for connector in learner._learner_connector)
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
            self.assertFalse(
                any(isinstance(connector, RemoveMaskedSamplesConnector) for connector in learner._learner_connector)
            )
