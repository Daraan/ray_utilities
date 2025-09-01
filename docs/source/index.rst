Ray Utilities Documentation
============================

.. warning::

   **This documentation is a work in progress.** Some sections may be incomplete or subject to change.
   Due to AI generated content it might be inaccurate. Please report any issues on the `GitHub repository <https://github.com/Daraan/ray_utilities/issues>`.

**Ray Utilities** is a comprehensive Python library providing utilities, setup frameworks, and extensions for Ray RLlib reinforcement learning experiments. It streamlines the process of configuring, running, and managing RL experiments with Ray Tune's hyperparameter optimization capabilities.

Key Features
------------

🚀 **Experiment Setup Framework**
   Complete lifecycle management for RL experiments from configuration to execution

🔧 **Configuration Management** 
   Type-safe argument parsing and algorithm configuration with sensible defaults

📊 **Hyperparameter Optimization**
   Native Ray Tune integration with advanced schedulers and search algorithms

🔄 **Training Utilities**
   Ready-to-use trainable classes with checkpointing and progress tracking

📈 **Evaluation & Monitoring**
   Built-in evaluation utilities and experiment tracking integration

🔌 **Extensible Components**
   Modular design with connectors, callbacks, and custom utilities

Quick Start
-----------

.. code-block:: python

   from ray_utilities.setup import PPOSetup
   from ray_utilities.runfiles import run_tune

   with PPOSetup() as setup:  
      # Inside the with block modify your configuration
      setup.config.env = "CartPole-v1"
      setup.config.lr = 0.001
   # Now the config is frozen and the setup.trainable is build
      
   # Run hyperparameter optimization
   results = run_tune(setup)

Installation
------------

.. code-block:: bash

   pip install ray_utilities

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

.. toctree::
   :maxdepth: 1
   :caption: Examples:
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`