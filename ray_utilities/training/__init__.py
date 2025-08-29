"""Training utilities and trainable classes for Ray Tune experiments.

Provides base trainable classes and functional training utilities for Ray Tune
integration with proper checkpoint/restore functionality.

Main Components:
    - :class:`DefaultTrainable`: Base trainable with checkpoint/restore support
    - :func:`default_trainable`: Functional trainable implementation
    - :func:`create_default_trainable`: Factory for creating trainables
"""

from .default_class import DefaultTrainable
from .functional import create_default_trainable, default_trainable

__all__ = [
    "DefaultTrainable",
    "create_default_trainable",
    "default_trainable",
]
