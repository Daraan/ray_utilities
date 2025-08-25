#!/usr/bin/env python3
from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup import PPOSetup

if __name__ == "__main__":
    PPOSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOSetup.group_name = "dynamic-rollout_buffer"  # pyright: ignore
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "--dynamic_buffer",
        # Meta / less influential arguments for the experiment.
        "--num_samples", 4,
        "--max_step_size", MAX_DYNAMIC_BATCH_SIZE,
        "--tags", "dynamic", "dynamic:rollout_buffer",  # per default includes "<env_type>", "<agent_type>",
        "--comment", "Default training run. Dynamic rollout buffer",
        "--env_seeding_strategy", "sequential",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "offline+upload",
        "--log_level", "INFO",
        "--log_stats", "most",
    ):  # fmt: skip
        setup: PPOSetup[DefaultArgumentParser] = PPOSetup()  # Replace with your own setup class
        results = run_tune(setup)
