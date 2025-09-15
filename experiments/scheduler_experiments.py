#!/usr/bin/env python3
"""Experimental scheduler tests"""

import ray
from ray.tune.tuner import Tuner

from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup
from ray_utilities.setup.tuner_setup import ScheduledTunerSetup

if __name__ == "__main__":
    ray.init(object_store_memory=4 * 1024**3)  # 4 GB
    PPOMLPSetup.PROJECT = "Default-<agent_type>-<env_type>"  # Upper category on Comet / WandB
    PPOMLPSetup.group_name = "test-scheduler"  # pyright: ignore

    class SchedulerSetup(PPOMLPSetup):
        def create_tuner(self) -> Tuner:
            return ScheduledTunerSetup(
                setup=self, eval_metric=EVAL_METRIC_RETURN_MEAN, eval_metric_order="max"
            ).create_tuner()

    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        # Meta / less influential arguments for the experiment.
        "--num_samples", 6,
        "--max_step_size", 2048,
        "--batch_size", 256,
        "--tags", "scheduler", # per default includes "<env_type>", "<agent_type>",
        "--comment", "Default training run with scheduler.",
        "--env_seeding_strategy", "same",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "online",
        "--log_level", "DEBUG",
        "--log_stats", "most",
        "--test",
        "--total_steps", 20_000
    ):  # fmt: skip
        setup = SchedulerSetup(config_files=["experiments/default.cfg"])  # Replace with your own setup class
        results = run_tune(setup)

        # TODO: Should not restore seed
