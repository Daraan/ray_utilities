Ray Utilities Documentation
============================

.. warning::

   **This documentation is a work in progress.** Some sections may be incomplete or subject to change.
   Due to AI generated content it might be inaccurate. Please report any issues on the `GitHub repository <https://github.com/Daraan/ray_utilities/issues>`_.

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

   from ray_utilities.config import DefaultArgumentParser
   from ray_utilities.runfiles import run_tune
   from ray_utilities.setup import PPOSetup

   with (
      DefaultArgumentParser.patch_args(
         # Override default CLI arguments. Passed CLI arguments still have a higher priority.
         "--batch-size", 1024,
      ),
      PPOSetup() as setup
   ):
      # Inside the with block modify your configuration
      # setup.config can be any AlgorithmConfig
      setup.config.environment(env="CartPole-v1")
      setup.config.lr = 0.001
   # After the with block the config is frozen and the setup.trainable is ready to use with Ray Tune

   # Run hyperparameter optimization
   results = run_tune(setup)


Ray Job Submission CLI
----------------------

The script ``ray_submit.py`` provides a command-line interface for submitting and monitoring Ray jobs in bulk using a YAML configuration file.
It leverages the Ray Job Submission API to automate experiment runs and track their status.
See one of the submission scripts in ``experiments/`` for example usage how to define an entrypoint template and replacement variables.

**Basic Usage:**

.. code-block:: bash

   # Basic use
   python ray_submit.py <group|"all"|"monitor"|"restore"> <submissions_file.yaml> [--address <RAY_DASHBOARD_URL>] [--max-jobs <N>] [--test]

   # View the help message for detailed options
   python ray_submit.py --help

**Arguments:**

- ``group``: The group key in the YAML file to run, ``all`` for all groups, or ``monitor`` to only monitor jobs.
  ``restore`` can be used to restore failed jobs. These must be labled with ``status: RESTORE`` in the YAML file.
- ``submissions_file``: Path to the YAML file containing job definitions.
- ``monitor_group``: (Optional) Additional group(s) to monitor.

**Options:**

- ``--address``: Address of the Ray cluster (default: uses ``DASHBOARD_ADDRESS`` env or ``localhost:8265``).
- ``--test``: Run in test mode without submitting jobs.
- ``--failed-only``: Only submit jobs that have previously failed.
- ``--max-jobs``: Maximum number of jobs to submit concurrently.
- ``--excludes ...``: Space-separated key words in the job name to exclude from submission.
- ``--includes ...``: Space-separated key words in the job name to include for submission, excludes all non-matching jobs.

**Example:**

.. code-block:: bash

   python ray_submit.py all submissions.yaml --failed-only

This will submit all jobs defined in ``submissions.yaml`` that have previously failed, using the specified Ray cluster address.

For more details, see the script's docstring or run ``python ray_submit.py --help``.


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
   ray_utilities

.. toctree::
   :maxdepth: 1
   :caption: Examples:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
