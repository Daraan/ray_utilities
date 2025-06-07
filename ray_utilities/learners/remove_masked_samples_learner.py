import logging

from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.learner.learner import Learner

from ray_utilities.connectors.remove_masked_samples_connector import RemoveMaskedSamplesConnector

_logger = logging.getLogger(__name__)


class RemoveMaskedSamplesLearner(Learner):
    """
    Extensions class for learners to append a RemoveMaskedSamplesConnector
    when building their learner connector pipeline.
    This reverts the effect of the `ray.rllib.connectors.learner.add_one_ts_to_episodes_and_truncate.
    AddOneTsToEpisodesAndTruncate` connector piece.

    Note:
        As a custom learner_connector argument to the AlgorithmConfig will only prepend
        connectors this class is needed to append it at the end of the pipeline.
    """

    def build_without_super(self):
        if self._learner_connector is not None:
            if self.config.learner_config_dict.get("remove_masked_samples", False):
                try:
                    self._learner_connector.insert_after(GeneralAdvantageEstimation, RemoveMaskedSamplesConnector())
                except ValueError as e:
                    _logger.warning(
                        "Failed to insert RemoveMaskedSamplesConnector, GeneralAdvantageEstimation not present %s. "
                        "Inserting at the end of the pipeline instead.",
                        e,
                    )
                    self._learner_connector.append(RemoveMaskedSamplesConnector())
            else:
                _logger.info(
                    "RemoveMaskedSamplesConnector not added to %s, "
                    "as config.learner_config_dict.remove_masked_samples is False.",
                    self.__class__.__name__,
                )
        else:
            _logger.warning(
                "Learner %s has no learner connector when tried to add %s. Ignoring.",
                self.__class__.__name__,
                RemoveMaskedSamplesLearner.__name__,
            )

    def build(self):
        super().build()
        self.build_without_super()


class RemoveMaskedSamplesPPOLearner(RemoveMaskedSamplesLearner, PPOLearner):
    """
    Subclass of rllib's PPOLearner that appends a RemoveMaskedSamplesConnector when building the learner
    connector pipeline.
    """
