from ray.rllib.core.learner import Learner

from ray_utilities.connectors.debug_connector import add_debug_connectors


class LearnerWithDebugConnectors(Learner):
    """
    Subclass of rllib's Learner that appends a DebugConnector when building the learner
    connector pipeline.

    Note:
        To assure that the DebugConnector is at the start and end of the pipeline this class
        should be the first in the inheritance chain.
        As this is often not possible use learners.mix_learners to combine the  class with other learners.
    """

    def build(self) -> None:
        super().build()
        add_debug_connectors(self)
