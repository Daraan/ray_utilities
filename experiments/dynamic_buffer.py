#!/usr/bin/env python3
import os

import ray

from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.misc import extend_trial_name
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

if __name__ == "__main__":
    ray.init(object_store_memory=4 * 1024**3)  # 4 GB
    PPOMLPSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPSetup.group_name = "dynamic:rollout_buffer"  # pyright: ignore
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "--dynamic_buffer",
        # Meta / less influential arguments for the experiment.
        "--num_samples", 5,
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
        setup = PPOMLPSetup(
            config_files=["experiments/models/mlp/default.cfg"],
            trial_name_creator=extend_trial_name(prepend="Dynamic_RolloutBuffer"),
        )
        results = run_tune(setup)
