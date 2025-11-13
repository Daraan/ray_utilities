#!/usr/bin/env python3
import os
import sys

# Enable shell completion for this file
import default_arguments.PYTHON_ARGCOMPLETE_OK
from experiments.create_tune_parameters import default_distributions, write_distributions_to_json
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.setup import PPOSetup
from ray_utilities.setup.extensions import load_distributions_from_json
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup
from ray_utilities.setup.scheduled_tuner_setup import PPOMLPWithPBTSetup
from ray_utilities.tune import update_hyperparameters
from ray_utilities.tune.scheduler.top_pbt_scheduler import KeepMutation

# Replace outputs to be more human readable and less nested
# env_runners -> training
# evaluation.env_runners -> evaluation
# if only "__all_modules__" and "default_policy" are present in learners, merges them.
# remember, env variables need to be set before ray.init()
os.environ.setdefault("RAY_UTILITIES_NEW_LOG_FORMAT", "1")

# print all messages from comet when we upload from a worker
os.environ.setdefault("RAY_DEDUP_LOGS_ALLOW_REGEX", "COMET|wandb")

if __name__ == "__main__":
    PPOMLPSetup.PROJECT = "dev-workspace"  # Upper category on Comet / WandB
    PPOMLPSetup.group_name = "debugging"  # pyright: ignore
    HYPERPARAMETERS = load_distributions_from_json(
        write_distributions_to_json(default_distributions, PPOMLPWithPBTSetup.TUNE_PARAMETER_FILE)
    )
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "-a", DefaultArgumentParser.agent_type,
        "--test",
        # Meta / less influential arguments for the experiment.
        # Assure constant total_steps across experiments.
        "--num_samples", 2,
        "--num_jobs", 2,
        "--max_step_size", 4096,
        "--total_steps", 4096 * 5,
        "--tags", "dev", "test",
        # constant
        "--seed", "42",
        "--wandb", "0",
        "--comet", "0",
        "--comment", "Debug - delete me",
        "--offline_loggers", "0",
    ):  # fmt: skip
        if "pbt" in sys.argv:
            setup = PPOMLPWithPBTSetup(
                config_files=["experiments/pbt.cfg", "experiments/default.cfg", "experiments/models/mlp/default.cfg"],
                # TODO: Trials are reused, trial name might be wrong then
            )
            assert setup.args.tune
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
            setup: PPOSetup[DefaultArgumentParser] = PPOMLPSetup(
                config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"]
            )
        os.environ["RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING"] = "1"
        setup.storage_path = "./outputs/experiments/TESTING/"
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
