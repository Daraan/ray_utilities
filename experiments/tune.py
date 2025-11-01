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
        "--max_step_size", MAX_DYNAMIC_BATCH_SIZE,
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
        setup = PPOMLPSetup(
            config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"],
            trial_name_creator=extend_trial_name(insert=["<batch_size>"], prepend="Tune_BatchSize"),
        )
        # Set hyperparameters to tune
        assert setup.args.tune
        hyperparameters = {k: HYPERPARAMETERS[k] for k in setup.args.tune}
        # TODO: Should put below logic into the Setup
        if "batch_size" in setup.args.tune:  # convenience key
            hyperparameters["train_batch_size_per_learner"] = hyperparameters.pop("batch_size")
        # Check grid search length and fix minibatch_size
        if (
            len(hyperparameters) == 1
            and isinstance(param := next(iter(hyperparameters.values())), dict)
            and "grid_search" in param
        ):
            # If only tuning batch size with cyclic mutation, also tune minibatch size accordingly
            if "minibatch_size" in hyperparameters:
                # Limit grid to be <= train_batch_size_per_learner
                param["grid_search"] = [v for v in param["grid_search"] if v <= setup.args.train_batch_size_per_learner]
            if len(param["grid_search"]) < setup.args.num_samples:
                # enlarge cyclic grid search values, Optuna shuffles
                param["grid_search"] = (
                    list(param["grid_search"]) * ((setup.args.num_samples // len(param["grid_search"])) + 1)
                )[: setup.args.num_samples]

        setup.param_space.update(hyperparameters)

        # Update group name
        PPOMLPSetup.group_name = "tune:" + "_".join(setup.args.tune)  # pyright: ignore
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
