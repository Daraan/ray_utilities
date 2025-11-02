#!/usr/bin/env python3
import os

import exceptiongroup  # noqa: F401


import default_arguments.PYTHON_ARGCOMPLETE_OK  # fmt: skip
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.misc import extend_trial_name
from ray_utilities.setup.scheduled_tuner_setup import PPOMLPWithPBTSetup
from ray_utilities.tune import validate_hyperparameters
from ray_utilities.tune.scheduler.top_pbt_scheduler import KeepMutation

os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

# print all messages from comet when we upload from a worker
os.environ.setdefault("RAY_DEDUP_LOGS_ALLOW_REGEX", "COMET|wandb")


if __name__ == "__main__":
    from experiments.create_tune_parameters import (
        default_distributions,
        load_distributions_from_json,
        write_distributions_to_json,
    )

    PPOMLPWithPBTSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPWithPBTSetup.group_name = "pbt:batch_size"  # pyright: ignore
    HYPERPARAMETERS = load_distributions_from_json(write_distributions_to_json(default_distributions))
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "--tune", "batch_size",
        # Meta / less influential arguments for the experiment.
        "--num_samples", 1, # NOTE: is multiplied by grid_search samples
        # TODO: DO not use sample space
        "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *PPOMLPWithPBTSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--tags", "pbt:batch_size", # per default includes "<env_type>", "<agent_type>",
        "--comment", "Tune with Top PBT scheduler over different batch sizes.",
        "--env_seeding_strategy", "same",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--evaluate_every_n_steps_before_step", "(512, 32784)",
        "--seed", "42",
        "--wandb", "online",
        "--comet", "online",
        "--log_level", "INFO",
        "--log_stats", "learners",
        # PBT arguments at the end
        "pbt",
        "--perturbation_interval", 1/8,
        config_files=["experiments/pbt.cfg"]
    ):  # fmt: skip
        setup = PPOMLPWithPBTSetup(
            config_files=["experiments/pbt.cfg", "experiments/default.cfg", "experiments/models/mlp/default.cfg"],
            # TODO: Trials are reused, trial name might be wrong then
            trial_name_creator=extend_trial_name(insert=["<batch_size>"], prepend="Tune_BatchSize_WithScheduler"),
        )
        assert setup.args.tune
        hyperparameters = {k: HYPERPARAMETERS[k] for k in setup.args.tune}
        validate_hyperparameters(
            hyperparameters,
            setup.args.tune,
            num_grid_samples=setup.args.num_samples,
            train_batch_size_per_learner=setup.args.train_batch_size_per_learner,
        )
        mutations: dict[str, KeepMutation[object]] = {k: KeepMutation() for k in hyperparameters.keys()}
        setup.param_space.update(hyperparameters)

        setup.args.command.set_hyperparam_mutations(mutations)  # pyright: ignore[reportArgumentType]
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
