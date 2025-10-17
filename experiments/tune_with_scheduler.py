#!/usr/bin/env python3
import os

import default_arguments.PYTHON_ARGCOMPLETE_OK
import ray

from ray_utilities import run_tune, runtime_env
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.misc import extend_trial_name
from ray_utilities.setup.scheduled_tuner_setup import PPOMLPWithPBTSetup
from ray_utilities.tune.scheduler.top_pbt_scheduler import KeepMutation

os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

# print all messages from comet when we upload from a worker
os.environ.setdefault("RAY_DEDUP_LOGS_ALLOW_REGEX", "COMET|wandb")

if __name__ == "__main__":
    PPOMLPWithPBTSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPWithPBTSetup.group_name = "pbt:batch_size"  # pyright: ignore
    PPOMLPWithPBTSetup.batch_size_sample_space = {"grid_search": [64, 128, 256, 512, 1024, 2048, 4096, 8192]}
    with (
        ray.init(num_cpus=11, object_store_memory=4 * 1024**3, runtime_env=runtime_env),   # 4 GB
        DefaultArgumentParser.patch_args(
            # main args for this experiment
            "--tune", "batch_size",
            # Meta / less influential arguments for the experiment.
            "--num_samples", 1, # NOTE: is multiplied by grid_search samples
            "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *PPOMLPWithPBTSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
            "--tags", "pbt:batch_size", # per default includes "<env_type>", "<agent_type>",
            "--comment", "Tune with Top PBT scheduler over different batch sizes.",
            "--env_seeding_strategy", "same",
            # constant
            "-a", DefaultArgumentParser.agent_type,
            "--seed", "42",
            "--wandb", "online",
            "--comet", "online",
            "--log_level", "INFO",
            "--log_stats", "learners",
            # PBT arguments at the end
            "pbt",
            "--perturbation_interval", 1/10,
            config_files=["experiments/pbt.cfg"]
        )
    ):  # fmt: skip
        setup = PPOMLPWithPBTSetup(
            config_files=["experiments/pbt.cfg", "experiments/models/mlp/default.cfg"],
            # TODO: Trials are reused, trial name might be wrong then
            trial_name_creator=extend_trial_name(insert=["<batch_size>"], prepend="Tune_BatchSize_WithScheduler"),
        )
        setup.args.command.set_hyperparam_mutations(
            {
                "train_batch_size_per_learner": KeepMutation(),
            }
        )
        results = run_tune(setup)
