#!/usr/bin/env python3
import ray

from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.misc import extend_trial_name
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

if __name__ == "__main__":
    ray.init(object_store_memory=4 * 1024**3)  # 4 GB
    PPOMLPSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPSetup.group_name = "tune:batch_size"  # pyright: ignore
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
        setup = PPOMLPSetup(
            config_files=["experiments/models/mlp/default.cfg"],
            trial_name_creator=extend_trial_name(insert=["<batch_size>"], prepend="Tune_BatchSize"),
        )
        results = run_tune(setup)
