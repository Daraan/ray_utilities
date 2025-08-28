"""Setup classes for Ray RLlib experiments and hyperparameter tuning.

This module provides base classes and concrete implementations for setting up
Ray RLlib experiments, including algorithm configuration, experiment management,
and hyperparameter tuning integration.

Key Components:
    - :class:`ExperimentSetupBase`: Abstract base class for experiment configuration
    - :class:`PPOSetup`: Concrete setup implementation for PPO algorithms  
    - :class:`TunerSetup`: Setup class to initialize the `tune.Tuner` for experiments..

These classes provide a structured approach to configuring and running
reinforcement learning experiments with Ray RLlib and Ray Tune integration.
"""
from .algorithm_setup import PPOSetup
from .experiment_base import ExperimentSetupBase
from .tuner_setup import TunerSetup

__all__ = ["ExperimentSetupBase", "PPOSetup", "TunerSetup"]
