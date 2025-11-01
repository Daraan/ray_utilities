#!/usr/bin/env python3
"""DQN training experiment with default configuration.

This script trains a DQN agent on a specified environment using the default
configuration. It demonstrates how to use the DQNMLPSetup class for
Deep Q-Network experiments.
"""

import os

import ray

# Enable shell completion for this file
import default_arguments.PYTHON_ARGCOMPLETE_OK
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.setup.ppo_mlp_setup import DQNMLPSetup

# Replace outputs to be more human readable and less nested
os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

# print all messages from comet when we upload from a worker
os.environ.setdefault("RAY_DEDUP_LOGS_ALLOW_REGEX", "COMET|wandb")

if __name__ == "__main__":
    DQNMLPSetup.PROJECT = "DQN-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    DQNMLPSetup.group_name = "dqn-default-training"  # pyright: ignore
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "--algorithm", "dqn",  # Force DQN algorithm
        "-a", DefaultArgumentParser.agent_type,
        # DQN-specific parameters
        "--target_network_update_freq", "500",
        "--num_steps_sampled_before_learning_starts", "1000",
        "--double_q",
        "--dueling",
        # Meta / less influential arguments for the experiment.
        "--tags", "static", "default", "dqn",  # per default includes "<env_type>", "<agent_type>",
        # constant
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "online",
        "--comment", "DQN default training run",
        "--log_stats", "most",
    ):  # fmt: skip
        # DQN-specific setup
        setup = DQNMLPSetup(
            config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"],
        )
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
