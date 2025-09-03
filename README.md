[![test workflow badge](https://github.com/Daraan/ray_utilities/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Daraan/ray_utilities/actions)
[![ReadtheDocs Badge](https://app.readthedocs.org/projects/ray-utilities/badge/?version=latest&style=flat)](https://app.readthedocs.org/projects/ray-utilities/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

# Ray Utilities

## Features

Many Features are stand-alone and can be used independently. The main features include:

- **JAX PPO for RLlib**: A JAX-based implementation of the Proximal Policy Optimization (PPO) algorithm compatible with RLlib Algorithm.
- **Ray + Optuna Grid Search + Optuna Pruners**: Extends ray's `OptunaSearch` to be compatible with 
- **Experiment Framework**: A base class for setting up experiments with dynamic parameters and parameter spaces, easily run via CLI and `ray.tune.Tuner`.
- **Reproducible Environments**: Reproducible environments for experiments using `ray.tune` by using a more sophisticated seeding mechanism.
- **Dynamic Parameter Tuning (WIP)**: Support for dynamic tuning of parameters during experiments.
    `ray.tune.grid_search` and Optuna pruners can work as a `Stopper`.

Furthermore smaller features:

- **Exact Environment Step Sampling**: RLlib sampling is slightly off and contains masked samples. A callback and connector piece correct this.
- **Improved Logger Callbacks**: Improved csv, Tensorboard, Wandb, Comet logger callbacks. Cleaner logs and better video handling.
- **PPO Torch Learner with Gradient Accumulation**: A PPO learner that supports gradient accumulation, useful for training with large batch sizes.

## Installation

### Install from PyPI

```bash
pip install ray_utilities
```

### Install latest version

Clone the repository and install the package using pip:

```bash
git clone https://github.com/Daraan/ray_utilities.git
cd ray_utilities
pip install .
```

## Documentation (Work in Progress)

Hosted on Read the Docs: https://ray-utilities.readthedocs.io/

## Run Experiments via CLI

Simple entry point:

```python
# File: run_experiment.py
from ray_utilities import run_tune
from ray_utilities.setup import PPOSetup

if __name__ == "__main__":
    with PPOSetup() as setup:  # Replace with your own setup class
        # The setup takes care of many settings passed in via the CLI
        # but the config (an rllib.AlgorithmConfig) can be adjusted
        # inside the code as well.
        # Changes made in this with block are tracked for checkpoint reloads
        setup.config.training(num_epochs=10)
    results = run_tune(setup)
```

Run the experiment:

```bash
python run_experiment.py -a MLP
```

> [!NOTE]  
> It is recommended to subclass `AlgorithmSetup` or `ExperimentSetupBase` to define your own setup. Extend `DefaultArgumentParser` to add custom CLI arguments. Above's `PPOSetup` is a very minimalistic example.
