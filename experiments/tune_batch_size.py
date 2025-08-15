# File: run_experiment.py
from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.setup import PPOSetup
from ray_utilities.testing_utils import patch_args

if __name__ == "__main__":
    PPOSetup.PROJECT = "DefaultTraining"
    with patch_args(
        "-a", DefaultArgumentParser.agent_type,
        "--tune", "batch_size",
        "--wandb", "offline+upload",
        "--comet", "offline+upload",
        "--seed", "42",
        "--num_samples", len(PPOSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--max_step_size", max(PPOSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--tags", "tune-batch_size", "mlp",
        "--comment", "Default training run. Tune batch size",
        "--env_seeding_strategy", "same",
        "--log_level", "INFO",
        extend=True,
    ):  # fmt: skip
        setup = PPOSetup()  # Replace with your own setup class
        results = run_tune(setup)
