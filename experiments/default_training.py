# File: run_experiment.py
from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.setup import PPOSetup
from ray_utilities.testing_utils import patch_args

if __name__ == "__main__":
    PPOSetup.PROJECT = "DefaultTraining"
    with patch_args(
        "-a", DefaultArgumentParser.agent_type,
        "--test",
        "--total_steps", 200_000,
        "--wandb", "offline+upload",
        "--comet", "offline+upload",
        extend=True,
    ):  # fmt: skip
        setup = PPOSetup()  # Replace with your own setup class
        results = run_tune(setup)
