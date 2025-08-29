"""Setup classes for Ray RLlib experiments and hyperparameter tuning.

The setup framework is the core of Ray Utilities, providing a comprehensive system
for configuring, running, and managing reinforcement learning experiments. It integrates
Ray RLlib with Ray Tune for scalable hyperparameter optimization while maintaining
clean separation between experiment configuration and execution.

**Core Design Principles:**

- **Configuration Management**: Centralized handling of algorithm configs, environment
  settings, and hyperparameters through type-safe argument parsing
- **Experiment Lifecycle**: Complete management from initialization through training
  to result collection and experiment tracking
- **Hyperparameter Optimization**: Native integration with Ray Tune for distributed
  parameter search with intelligent defaults

**Main Components:**

- :class:`ExperimentSetupBase`: Abstract foundation for all experiment configurations.
  Handles argument parsing, config validation, and trainable creation
- :class:`PPOSetup`: Production-ready PPO algorithm setup with sensible defaults
  and extensive customization options
- :class:`TunerSetup`: Ray Tune integration for hyperparameter optimization with
  support for various schedulers, search algorithms, and stopping criteria

**Typical Workflow:**

1. **Setup Creation**: Instantiate a setup class (e.g., :class:`PPOSetup`)
2. **Configuration**: Modify algorithm config, environment, and hyperparameters
3. **Execution**: Run experiments via :func:`~ray_utilities.runfiles.run_tune.run_tune`
4. **Analysis**: Process results using Ray Tune's analysis tools

**Examples:**

Basic experiment setup:

>>> from ray_utilities.setup import PPOSetup
>>> setup = PPOSetup()
>>> setup.config.env = "CartPole-v1"
>>> setup.config.lr = 0.001

Hyperparameter optimization:

>>> setup.add_tune_config({"lr": tune.grid_search([0.001, 0.01, 0.1]), "gamma": tune.uniform(0.9, 0.99)})

Custom environment with training configuration:

>>> setup.config.env = "MyCustomEnv"
>>> setup.config.training_iteration = 100
>>> setup.config.checkpoint_freq = 10

See Also:
    :mod:`ray_utilities.runfiles.run_tune`: Main execution functions
    :class:`ray.rllib.algorithms.ppo.PPOConfig`: Underlying algorithm configuration
    :class:`ray.tune.Tuner`: Ray Tune hyperparameter optimization
"""

from .algorithm_setup import PPOSetup
from .experiment_base import ExperimentSetupBase
from .tuner_setup import TunerSetup

__all__ = ["ExperimentSetupBase", "PPOSetup", "TunerSetup"]
