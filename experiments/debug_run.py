#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

# Enable shell completion for this file
from experiments.create_tune_parameters import default_distributions, write_distributions_to_json
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.setup.extensions import load_distributions_from_json
from ray_utilities.setup.ppo_mlp_setup import MLPSetup
from ray_utilities.setup.scheduled_tuner_setup import MLPPBTSetup
from ray_utilities.tune import update_hyperparameters
from ray_utilities.tune.scheduler.top_pbt_scheduler import KeepMutation

if TYPE_CHECKING:
    from ray_utilities.config.parser.mlp_argument_parser import MLPArgumentParser
    from ray_utilities.config.parser.pbt_scheduler_parser import PopulationBasedTrainingParser

# Replace outputs to be more human readable and less nested
# env_runners -> training
# evaluation.env_runners -> evaluation
# if only "__all_modules__" and "default_policy" are present in learners, merges them.
# remember, env variables need to be set before ray.init()
os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

# print all messages from comet when we upload from a worker
os.environ.setdefault("RAY_DEDUP_LOGS_ALLOW_REGEX", "COMET|wandb")

if __name__ == "__main__":
    HYPERPARAMETERS = load_distributions_from_json(
        write_distributions_to_json(default_distributions, MLPPBTSetup.TUNE_PARAMETER_FILE)
    )
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "-a", DefaultArgumentParser.agent_type,
        "--test",
        # Meta / less influential arguments for the experiment.
        # Assure constant total_steps across experiments.
        "--num_samples", 1,
        "--num_jobs", 2,
        "--max_step_size", 4096,
        "--total_steps", 4096 * 5,
        "--tags", "dev", "test",
        # constant
        "--seed", "42",
        "--wandb", "0",
        "--comet", "0",
        "--comment", "Debug - delete me",
        "--log_level", "DEBUG",
        "--offline_loggers", "0",
    ):  # fmt: skip
        setup: MLPSetup[MLPArgumentParser[None] | MLPArgumentParser[PopulationBasedTrainingParser]]
        if "pbt" in sys.argv:
            setup = MLPPBTSetup(
                config_files=["experiments/pbt.cfg", "experiments/default.cfg", "experiments/models/mlp/default.cfg"],
                # TODO: Trials are reused, trial name might be wrong then
            )
            setup.PROJECT = "dev-workspace"  # Upper category on Comet / WandB
            setup.group_name = "debugging"  # pyright: ignore
            assert setup.args.tune, "Use --tune ... when using PBT setup"
            hyperparameters = {k: HYPERPARAMETERS[k] for k in setup.args.tune}
            hyperparameters = update_hyperparameters(
                setup.param_space,
                hyperparameters,
                setup.args.tune,
                num_grid_samples=setup.args.num_samples,
                train_batch_size_per_learner=setup.args.train_batch_size_per_learner,
            )

            mutations: dict[str, KeepMutation[object]] = {k: KeepMutation() for k in hyperparameters.keys()}
            setup.args.command.set_hyperparam_mutations(mutations)  # pyright: ignore[reportArgumentType]
        else:
            setup = MLPSetup(config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"])
        os.environ["RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING"] = "1"
        setup.base_storage_path = "./outputs/experiments/TESTING/"
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
