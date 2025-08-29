"""Advanced learner implementations and mixins for Ray RLlib.

Provides specialized learner classes and utilities for extending Ray RLlib
learning capabilities, including experimental features like gradient accumulation,
debugging connectors, and sample filtering.

Key Components:
    - :func:`mix_learners`: Combine multiple learner classes
    - Debugging and filtering learner mixins
    - Experimental gradient accumulation support

Example:
    >>> from ray_utilities.learners import mix_learners
    >>> # Combine multiple learner features
    >>> CustomLearner = mix_learners([DebugLearner, FilterLearner, BaseLearner])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
from abc import ABCMeta

if TYPE_CHECKING:
    from ray.rllib.core.learner import Learner


class _MixedLearnerMeta(ABCMeta):
    """Metaclass for mixed learner classes that enables proper comparison and hashing.

    This metaclass allows mixed learner classes to be compared based on their
    base classes rather than their identity, which is important for serialization
    and class equivalence checking in Ray distributed environment.

    The metaclass ensures that two mixed learner classes with the same base
    classes are considered equivalent regardless of creation order or location.
    """

    def __eq__(cls, value: Any):  # noqa: PYI032
        if not hasattr(value, "__bases__"):
            return False
        return set(cls.__bases__) == set(value.__bases__)

    def __hash__(cls):  # needed for a serialization check of the learner
        # order invariant hash
        return sum(hash(base) for base in cls.__bases__) + hash(cls.__name__)

    def __repr__(cls):
        return f"<class MixedLearner({', '.join(base.__name__ for base in cls.__bases__)})>"


def mix_learners(learners: Sequence[type[Learner | Any]]):
    """Combine multiple learner classes into a single learner class.

    This function creates a new learner class that inherits from all provided
    learner classes, allowing you to compose different learner capabilities.
    It is particularly useful when you want to combine debugging, sample filtering,
    and algorithm-specific features.

    Args:
        learners: A sequence of learner classes to combine. The classes will be
            used as base classes for the new mixed learner in the order provided.
            If a class is already a mixed learner, its base classes will be
            extracted and used instead to avoid deep inheritance hierarchies.

    Returns:
        A new learner class that inherits from all provided learner classes.
        If only one learner is provided, returns that learner unchanged.

    Example:
        Combine debugging with a specific algorithm learner::

        >>> from ray_utilities.learners import mix_learners
        >>> from ray_utilities.learners.leaner_with_debug_connector import LearnerWithDebugConnectors
        >>> from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
        >>>
        >>> DebugPPOLearner = mix_learners([LearnerWithDebugConnectors, PPOTorchLearner])

        Complex composition with multiple features::

        >>> CustomLearner = mix_learners([LearnerWithDebugConnectors, RemoveMaskedSamplesLearner, PPOTorchLearner])

    Note:
        - The order of learners matters for method resolution order (MRO)
        - Duplicate learner classes are automatically removed
        - If a learner is already mixed, its constituent base classes are used
          instead to keep the inheritance hierarchy flat
        - The resulting class uses :class:`_MixedLearnerMeta` for proper
          comparison and serialization behavior

    See Also:
        :class:`_MixedLearnerMeta`: Metaclass used for mixed learner classes
        :class:`ray_utilities.learners.leaner_with_debug_connector.LearnerWithDebugConnectors`:
            Example debugging learner mixin
    """
    assert learners, "At least one learner class must be provided."
    if len(learners) == 1:
        return learners[0]

    # when a learner is already a MixedLearner use its bases
    base_learners = []
    for learner in learners:
        if isinstance(learner, _MixedLearnerMeta):
            for base in learner.__bases__:
                if base not in base_learners:
                    base_learners.append(base)
        elif learner not in base_learners:
            base_learners.append(learner)

    class MixedLearner(*base_learners, metaclass=_MixedLearnerMeta):
        pass

    return MixedLearner
