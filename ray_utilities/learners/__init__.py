from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from ray.rllib.core.learner import Learner


def mix_learners(learners: Sequence[type[Learner | Any]]):
    """
    Combines multiple learner classes to a single learner class inheriting from all.

    This is useful when you want to use DebugConnector to assure their correct placement:

        learner_class = mix_learners(
            LearnerWithDebugConnectors,
            RemoveMaskedSamplesLearner,
            TorchPPOLearner,
        )
    """
    assert learners, "At least one learner class must be provided."
    if len(learners) == 1:
        return learners[0]

    class MixedLearner(*learners):
        pass

    return MixedLearner
