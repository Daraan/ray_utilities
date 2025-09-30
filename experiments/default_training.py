#!/usr/bin/env python3
import os

# Enable shell completion for this file
import default_arguments.PYTHON_ARGCOMPLETE_OK
import ray

from ray_utilities import run_tune, runtime_env
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup import PPOSetup
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

# Replace outputs to be more human readable and less nested
# env_runners -> training
# evaluation.env_runners -> evaluation
# if only "__all_modules__" and "default_policy" are present in learners, merges them.
# remember, env variables need to be set before ray.init()
os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

if __name__ == "__main__":
    ray.init(object_store_memory=4 * 1024**3, runtime_env=runtime_env)  # 4 GB
    PPOMLPSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPSetup.group_name = "default-training"  # pyright: ignore
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "-a", DefaultArgumentParser.agent_type,
        # Meta / less influential arguments for the experiment.
        # Assure constant total_steps across experiments.
        "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *PPOSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--tags", "static", "default",  # per default includes "<env_type>", "<agent_type>",
        # constant
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "online",
        "--comment", "Default training run",
        "--log_stats", "most",
    ):  # fmt: skip
        # Replace with your own setup class
        setup: PPOSetup[DefaultArgumentParser] = PPOMLPSetup(config_files=["experiments/models/mlp/default.cfg"])
        results = run_tune(setup)
