#!/usr/bin/env python3
import os

import default_arguments.PYTHON_ARGCOMPLETE_OK  # fmt: skip  # noqa: F401
import exceptiongroup  # noqa: F401
from experiments.ray_init_helper import init_ray_with_setup

from ray_utilities import get_runtime_env, run_tune
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.misc import extend_trial_name
from ray_utilities.setup.scheduled_tuner_setup import MLPPBTSetup
from ray_utilities.tune.scheduler.top_pbt_scheduler import KeepMutation
from sympol.rllib_port.core.sympol_pbt_setup import SympolPBTSetup
from sympol.rllib_port.extended_args import SympolArgumentParser

os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

# print all messages from comet when we upload from a worker
os.environ.setdefault("RAY_DEDUP_LOGS_ALLOW_REGEX", "COMET|wandb")


if __name__ == "__main__":
    from experiments.create_tune_parameters import (
        default_distributions,
        load_distributions_from_json,
        write_distributions_to_json,
    )

    HYPERPARAMETERS = load_distributions_from_json(
        write_distributions_to_json(default_distributions, MLPPBTSetup.TUNE_PARAMETER_FILE)
    )
    with SympolArgumentParser.patch_args(
        # main args for this experiment
        "--tune", "batch_size",
        # Meta / less influential arguments for the experiment.
        "--num_samples", 1, # NOTE: is multiplied by grid_search samples
        # TODO: DO not use sample space
        "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *HYPERPARAMETERS["batch_size"]["grid_search"]), # pyright: ignore[reportIndexIssue] # TODO: Support Domain here
        "--tags", "pbt:batch_size", # per default includes "<env_type>", "<agent_type>",
        "--comment", "Tune with Top PBT scheduler over different batch sizes.",
        "--env_seeding_strategy", "same",
        # constant
        "-a", "sympol",
        "--evaluate_every_n_steps_before_step", "(512, 32784)",
        "--seed", "42",
        "--wandb", "online",
        "--comet", "online",
        "--log_level", "INFO",
        "--log_stats", "learners",
        # PBT arguments at the end
        "pbt", "--perturbation_interval", 1/8,
        config_files=["experiments/pbt.cfg"]
    ):  # fmt: skip
        with SympolPBTSetup(
            config_files=["experiments/pbt.cfg", "experiments/default.cfg"],
        ) as setup:
            setup.PROJECT = "SYMPOL-<critic>-<env_type>"  # Upper category on Comet / WandB, parent directory
            setup.GROUP = "pbt-<tune>"  # group on Comet / WandB, sub directory
            assert setup.args.tune
            setup._tune_trial_name_creator = extend_trial_name(
                append=[f"<{k}>" for k in setup.args.tune], prepend="PBT_" + "_".join(setup.args.tune)
            )
        mutations: dict[str, KeepMutation[object]] = {k: KeepMutation() for k in setup.args.tune}

        assert setup.args.actor == "sympol"
        setup.args.command.set_hyperparam_mutations(mutations)  # pyright: ignore[reportArgumentType]
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
