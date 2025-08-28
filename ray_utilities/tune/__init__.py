"""Ray Tune extensions and utilities for hyperparameter optimization.

This module provides custom schedulers, stoppers, and utilities that extend
Ray Tune's capabilities for hyperparameter optimization of reinforcement learning
experiments. It includes specialized components designed for RL training workflows
and integration with Ray RLlib.

Key Components:
    - Custom schedulers for adaptive training schedule management
    - Specialized stoppers for RL-specific stopping criteria  
    - Integration utilities for seamless Ray Tune and RLlib workflows

The extensions in this module are designed to work with the experiment setup
framework and provide additional control over the hyperparameter optimization
process beyond what's available in standard Ray Tune.
"""
