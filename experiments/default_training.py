#!/usr/bin/env python3
import os

# Enable shell completion for this file
import default_arguments.PYTHON_ARGCOMPLETE_OK
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup import PPOSetup
from ray_utilities.setup.ppo_mlp_setup import DQNMLPSetup, PPOMLPSetup

# Replace outputs to be more human readable and less nested
# env_runners -> training
# evaluation.env_runners -> evaluation
# if only "__all_modules__" and "default_policy" are present in learners, merges them.
# remember, env variables need to be set before ray.init()
os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

# print all messages from comet when we upload from a worker
os.environ.setdefault("RAY_DEDUP_LOGS_ALLOW_REGEX", "COMET|wandb")

if __name__ == "__main__":
    # Parse algorithm selection early to determine which setup to use
    parser = DefaultArgumentParser()
    temp_args, _ = parser.parse_known_args()
    algorithm = getattr(temp_args, "algorithm", "ppo")

    # Select setup class based on algorithm
    if algorithm == "dqn":
        SetupClass = DQNMLPSetup
    else:
        SetupClass = PPOMLPSetup

    SetupClass.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    SetupClass.group_name = "default-training"  # pyright: ignore

    from experiments.create_tune_parameters import (
        default_distributions,
        load_distributions_from_json,
        write_distributions_to_json,
    )

    HYPERPARAMETERS = load_distributions_from_json(write_distributions_to_json(default_distributions))

    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "-a", DefaultArgumentParser.agent_type,
        # Meta / less influential arguments for the experiment.
        # Assure constant total_steps across experiments.
        "--num_samples", 3,
        # Note we do not tune in this file but want to align it with the others
        "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *HYPERPARAMETERS["batch_size"]["grid_search"]), # pyright: ignore
        "--tags", "static", "default",  # per default includes "<env_type>", "<agent_type>",
        # constant
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "online",
        "--comment", "Default training run",
    ):  # fmt: skip
        # Replace with your own setup class
        setup = SetupClass(
            config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"]
        )
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)

