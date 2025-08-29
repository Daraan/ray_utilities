"""Setup classes for Ray RLlib experiments and hyperparameter tuning.

Provides base classes and concrete implementations for configuring Ray RLlib
experiments, including algorithm setup, experiment management, and hyperparameter
tuning with Ray Tune.

Main Components:
    - :class:`ExperimentSetupBase`: Abstract base class for experiment configuration
    - :class:`PPOSetup`: Concrete PPO algorithm setup implementation  
    - :class:`TunerSetup`: Ray Tune integration for hyperparameter optimization
"""
from .algorithm_setup import PPOSetup
from .experiment_base import ExperimentSetupBase
from .tuner_setup import TunerSetup

__all__ = ["ExperimentSetupBase", "PPOSetup", "TunerSetup"]
