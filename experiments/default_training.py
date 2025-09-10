#!/usr/bin/env python3
from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup import PPOSetup
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

if __name__ == "__main__":
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
        setup: PPOSetup[DefaultArgumentParser] = PPOMLPSetup(config_files=["experiments/default.cfg"])
        results = run_tune(setup)
