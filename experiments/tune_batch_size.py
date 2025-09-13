#!/usr/bin/env python3
from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

if __name__ == "__main__":
    PPOMLPSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPSetup.group_name = "tune-batch_size"  # pyright: ignore
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "--tune", "batch_size",
        # Meta / less influential arguments for the experiment.
        "--num_samples", len(PPOMLPSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *PPOMLPSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--tags", "tune-batch_size", # per default includes "<env_type>", "<agent_type>",
        "--comment", "Default training run. Tune batch size",
        "--env_seeding_strategy", "same",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "offline+upload",
        "--log_level", "INFO",
        "--log_stats", "most",
    ):  # fmt: skip
        setup = PPOMLPSetup(config_files=["experiments/default.cfg"])  # Replace with your own setup class
        results = run_tune(setup)
