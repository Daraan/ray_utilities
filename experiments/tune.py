#!/usr/bin/env python3
import os

import default_arguments.PYTHON_ARGCOMPLETE_OK
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.misc import extend_trial_name
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup
from ray_utilities.tune import update_hyperparameters

os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

if __name__ == "__main__":
    PPOMLPSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    from experiments.create_tune_parameters import (
        default_distributions,
        load_distributions_from_json,
        write_distributions_to_json,
    )

    HYPERPARAMETERS = load_distributions_from_json(write_distributions_to_json(default_distributions))
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        # --tune should use
        "--num_samples", 8,
        # Meta / less influential arguments for the experiment.
        "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *HYPERPARAMETERS["batch_size"]["grid_search"]), # pyright: ignore
        "--tags", "tune", # per default includes "<env_type>", "<agent_type>",
        "--comment", "Default training run. Tune batch size",
        "--env_seeding_strategy", "same",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--seed", "42",
        "--wandb", "offline+upload@end",
        "--comet", "offline+upload",
        "--log_level", "INFO",
    ):  # fmt: skip
        with PPOMLPSetup(
            config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"],
            trial_name_creator=extend_trial_name(insert=["<batch_size>"], prepend="Tune_BatchSize"),
        ) as setup:
            # Set hyperparameters to tune
            assert setup.args.tune
            setup.GROUP = "tune-" + "_".join(setup.args.tune)  # pyright: ignore
        hyperparameters = update_hyperparameters(
            setup.param_space,
            {k: HYPERPARAMETERS[k] for k in setup.args.tune},
            setup.args.tune,
            num_grid_samples=setup.args.num_samples,
            train_batch_size_per_learner=setup.args.train_batch_size_per_learner,
        )

        # Update group name
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
