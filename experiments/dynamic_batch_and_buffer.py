#!/usr/bin/env python3
from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

if __name__ == "__main__":
    PPOMLPSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPSetup.group_name = "dynamic:batch_size+rollout_size"  # pyright: ignore
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "--dynamic_batch",
        "--dynamic_rollout",
        # Meta / less influential arguments for the experiment.
        "--num_samples", 8,
        "--max_step_size", MAX_DYNAMIC_BATCH_SIZE,
        "--tags", "dynamic", "dynamic:batch_size", "dynamic:rollout", "dynamic:batch_size+rollout",
        "--comment", "Default training run. Dynamic batch size via gradient accumulation",
        "--env_seeding_strategy", "sequential",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "offline+upload",
        "--log_level", "INFO",
        "--log_stats", "most",
    ):  # fmt: skip
        setup = PPOMLPSetup()  # Replace with your own setup class
        results = run_tune(setup)
