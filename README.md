[![test workflow badge](https://github.com/Daraan/ray_utilities/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Daraan/ray_utilities/actions)
[![ReadtheDocs Badge](https://app.readthedocs.org/projects/ray-utilities/badge/?version=latest&style=flat)](https://app.readthedocs.org/projects/ray-utilities/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

# Ray Utilities

## Quickstart

Train a PPO agent on CartPole-v1 with default settings and log to WandB after the experiment has finished:

```bash
python experiments/default_training.py --env CartPole-v1 --wandb offline+upload
```


## Features


Many features are stand-alone and can be used independently. The main features include:

- **JAX PPO for RLlib**: A JAX-based implementation of the Proximal Policy Optimization (PPO) algorithm compatible with RLlib Algorithm.
- **Ray + Optuna Grid Search + Optuna Pruners**: Extends ray's `OptunaSearch` to be compatible with RLlib and supports advanced pruners.
- **Experiment Framework**: A base class for setting up experiments with dynamic parameters and parameter spaces, easily run via CLI and `ray.tune.Tuner`.
- **Reproducible Environments**: Reproducible environments for experiments using `ray.tune` by using a more sophisticated seeding mechanism.
- **Dynamic Parameter Tuning (WIP)**: Support for dynamic tuning of parameters during experiments. `ray.tune.grid_search` and Optuna pruners can work as a `Stopper`.
- **Trial forking and Experiment Key Management**: Enhanced support for trial forking and experiment key management, including parsing and restoring from forked checkpoints.
This is especially designed for Population Based Training (PBT) and similar use cases and combined with WandB's support for [fork logging](https://docs.wandb.ai/guides/runs/forking/).


## What's New in v0.5.0

### Highlights

- **Config File Integration**: Config files can now be used seamlessly with the argument parser. Specify config files via `-cfg` or `--config_files` and all arguments in the file will be parsed as if passed on the command line.

- **Flexible Tagging via CLI**: Add tags directly from the command line using `--tag:tag_name` (for a tag without value) or `--tag:tag_name=value` (for a tag with a value). Tags are automatically logged to experiment tracking tools (WandB, Comet, etc) and help organize and filter your results.

- **Population Based Training (PBT) and Forking**: The new `TopPBTTrialScheduler` enables advanced population-based training with quantile-based exploitation and flexible mutation strategies. Forked trials are automatically tracked and restored, and experiment lineage is visible in WandB/Comet dashboards.

- **Advanced Comet and WandB Callbacks**: Improved handling for online/offline experiment tracking, including robust upload and sync logic for offline runs.

- **Improved Log Formatting**: All experiment outputs are now more human-readable and less nested. Training and evaluation results are flattened and easier to interpret.

- **Helper Utilities**: New and improved helpers for experiment key generation, argument patching, and test utilities.

---


## Other Features

- **Exact Environment Step Sampling**: Ensures accurate step counts in RLlib.
- **Improved Logger Callbacks**: Cleaner logs and better video handling for CSV, Tensorboard, WandB, and Comet.
- **PPO Torch Learner with Gradient Accumulation**: Efficient training with large batches.

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

Visit https://ray-utilities.readthedocs.io/

## Experiments

### Pick What You Need - Customize Your Experiments

[`ExperimentSetupBase`](https://ray-utilities.readthedocs.io/ray_utilities.setup.html#ray-utilities-setup-experiment-base-module) classes provide a modular way to parse your configuration,
setup trainables, their parameters and a Tuner, executed by [`run_tune`](https://ray-utilities.readthedocs.io/ray_utilities.runfiles.html#ray-utilities-runfiles-run-tune-module).

Simple entry point:


```python
# File: run_experiment.py
from ray_utilities import run_tune
from ray_utilities.setup import PPOSetup

if __name__ == "__main__":
    # Take a default setup or adjust to your needs
    with PPOSetup() as setup:
        # The setup takes care of many settings passed in via the CLI
        # but the config (an rllib.AlgorithmConfig) can be adjusted
        # inside the code as well.
        # Changes made in this with block are tracked for checkpoint reloads
        setup.config.training(num_epochs=10)
    results = run_tune(setup)
```



## Using Config Files and Tags

You can specify experiment parameters in a config file and combine them with CLI arguments. This makes it easy to share and reproduce experiments.

```bash
python run_experiment.py -cfg experiments/models/mlp/default.cfg --tag:baseline --tag:lr=0.001
```

Tags are automatically logged to experiment tracking tools (WandB, Comet, etc) and help organize and filter your results.
## Population Based Training (PBT) and Forking

The new `TopPBTTrialScheduler` enables advanced population-based training with quantile-based exploitation and flexible mutation strategies. Forked trials are automatically tracked and restored, and experiment lineage is visible in WandB/Comet dashboards.

See the documentation for a full example and advanced usage: [Read the Docs](https://ray-utilities.readthedocs.io/)

> [!NOTE]
> It is recommended to subclass `AlgorithmSetup` or [`ExperimentSetupBase`](https://ray-utilities.readthedocs.io/ray_utilities.setup.html#ray-utilities-setup-experiment-base-module) to define your own setup. Extend `DefaultArgumentParser` to add custom CLI arguments. Above's `PPOSetup` is a very minimalistic example.
