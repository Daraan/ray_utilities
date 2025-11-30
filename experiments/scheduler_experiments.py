#!/usr/bin/env python3
"""Experimental scheduler tests"""

import os

import default_arguments.PYTHON_ARGCOMPLETE_OK
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.setup.scheduled_tuner_setup import DQNMLPWithReTuneSetup, PPOMLPWithReTuneSetup

os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

if __name__ == "__main__":
    # Parse algorithm selection early to determine which setup to use
    parser = DefaultArgumentParser()
    temp_args, _ = parser.parse_known_args()
    algorithm = getattr(temp_args, "algorithm", "ppo")

    # Select setup class based on algorithm
    if algorithm == "dqn":
        SetupClass = DQNMLPWithReTuneSetup
    else:
        SetupClass = PPOMLPWithReTuneSetup

    SetupClass.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    SetupClass.group_name = "test-scheduler"  # pyright: ignore

    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        # Meta / less influential arguments for the experiment.
        "--num_samples", 6,
        "--max_step_size", 2048,
        "--batch_size", 256,
        "--tags", "scheduler", # per default includes "<env_type>", "<agent_type>",
        "--comment", "Default training run with scheduler.",
        "--env_seeding_strategy", "same",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "online",
        "--log_level", "DEBUG",
        "--log_stats", "most",
        "--test",
        "--total_steps", 20_000
    ):  # fmt: skip
        setup = SetupClass(
            config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"],
        )  # Replace with your own setup class
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
