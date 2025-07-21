from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
from abc import ABCMeta

if TYPE_CHECKING:
    from ray.rllib.core.learner import Learner


class _MixedLearnerMeta(ABCMeta):
    """Metaclass that allows to compare local MixedLearner classes"""

    def __eq__(cls, value: Any):  # noqa: PYI032
        if not hasattr(value, "__bases__"):
            return False
        return set(cls.__bases__) == set(value.__bases__)

    def __hash__(cls):  # needed for a serialization check of the learner
        # order invariant hash
        return sum(hash(base) for base in cls.__bases__) + hash(cls.__name__)

    def __repr__(cls):
        return f"MixedLearner({', '.join(base.__name__ for base in cls.__bases__)})"


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

    class MixedLearner(*learners, metaclass=_MixedLearnerMeta):
        pass

    return MixedLearner
