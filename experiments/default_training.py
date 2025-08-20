# File: run_experiment.py
from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup import PPOSetup

if __name__ == "__main__":
    PPOSetup.PROJECT = "Default-MLP"  # Upper category on Comet / WandB
    PPOSetup.group_name = "default-training"  # pyright: ignore
    with DefaultArgumentParser.patch_args(
        # main args for this experiment
        "-a", DefaultArgumentParser.agent_type,
        # Meta / less influential arguments for the experiment.
        # Assure constant total_steps accross experiments.
        "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *PPOSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--tags", "static", "default",
        # constant
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "online",
        "--comment", "Default training run",
    ):  # fmt: skip
        setup = PPOSetup()  # Replace with your own setup class
        results = run_tune(setup)
