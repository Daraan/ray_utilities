#!/usr/bin/env python3
import os

import default_arguments.PYTHON_ARGCOMPLETE_OK
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.misc import extend_trial_name
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup

os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

if __name__ == "__main__":
    PPOMLPSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPSetup.GROUP = "dynamic-batch_size"
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "--dynamic_batch",
        "--batch_size", 512,
        # Meta / less influential arguments for the experiment.
        "--num_samples", 5,
        "--max_step_size", MAX_DYNAMIC_BATCH_SIZE,
        "--tags", "dynamic", "dynamic:batch_size",  # per default includes "<env_type>", "<agent_type>",
        "--comment", "Default training run. Dynamic batch size via gradient accumulation",
        "--env_seeding_strategy", "sequential",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "offline+upload",
        "--log_level", "INFO",
    ):  # fmt: skip
        setup = PPOMLPSetup(
            config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"],
            trial_name_creator=extend_trial_name(prepend="Dynamic_GradientAccumulation"),
        )
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
