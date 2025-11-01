#!/usr/bin/env python3
import os
import sys

# Enable shell completion for this file
import default_arguments.PYTHON_ARGCOMPLETE_OK
from experiments.create_tune_parameters import default_distributions, write_distributions_to_json
from experiments.ray_init_helper import init_ray_with_setup
from ray_utilities import get_runtime_env, run_tune
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup import PPOSetup
from ray_utilities.setup.extensions import load_distributions_from_json
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup
from ray_utilities.setup.scheduled_tuner_setup import PPOMLPWithPBTSetup
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
    HYPERPARAMETERS = load_distributions_from_json(write_distributions_to_json(default_distributions))
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
        # Replace with your own setup class
        if "pbt" in sys.argv:
            setup = PPOMLPWithPBTSetup(
                config_files=["experiments/pbt.cfg", "experiments/default.cfg", "experiments/models/mlp/default.cfg"],
                # TODO: Trials are reused, trial name might be wrong then
            )
            assert setup.args.tune
            hyperparameters = {k: HYPERPARAMETERS[k] for k in setup.args.tune}
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
                    param["grid_search"] = [
                        v for v in param["grid_search"][2:-2] if v <= setup.args.train_batch_size_per_learner
                    ]
                if len(param["grid_search"]) < setup.args.num_samples:
                    # enlarge cyclic grid search values, Optuna shuffles
                    param["grid_search"] = (
                        list(param["grid_search"]) * ((setup.args.num_samples // len(param["grid_search"][2:-2])) + 1)
                    )[: setup.args.num_samples]

            mutations: dict[str, KeepMutation[object]] = {k: KeepMutation() for k in hyperparameters.keys()}
            setup.param_space.update(hyperparameters)

            setup.args.command.set_hyperparam_mutations(mutations)  # pyright: ignore[reportArgumentType]
        else:
            setup: PPOSetup[DefaultArgumentParser] = PPOMLPSetup(
                config_files=["experiments/default.cfg", "experiments/models/mlp/default.cfg"]
            )
        os.environ["RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING"] = "1"
        setup.storage_path = "./outputs/experiments/TESTING/"
        with init_ray_with_setup(setup, runtime_env=get_runtime_env()):
            results = run_tune(setup)
