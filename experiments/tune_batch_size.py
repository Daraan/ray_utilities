# File: run_experiment.py
from ray_utilities import run_tune
from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup import PPOSetup
from ray_utilities.testing_utils import patch_args

if __name__ == "__main__":
    PPOSetup.PROJECT = "Default-MLP"  # Upper category on Comet / WandB
    PPOSetup.group_name = "tune-batch_size"  # pyright: ignore
    with patch_args(
        # main args for this experiment
        "--tune", "batch_size",
        # Meta / less influential arguments for the experiment.
        "--num_samples", len(PPOSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--max_step_size", max(MAX_DYNAMIC_BATCH_SIZE, *PPOSetup.batch_size_sample_space["grid_search"]), # pyright: ignore
        "--tags", "tune-batch_size", "mlp",
        "--comment", "Default training run. Tune batch size",
        "--env_seeding_strategy", "same",
        # constant
        "-a", DefaultArgumentParser.agent_type,
        "--seed", "42",
        "--wandb", "offline+upload",
        "--comet", "offline+upload",
        "--log_level", "INFO",
        extend_argv=True,
    ):  # fmt: skip
        setup = PPOSetup()  # Replace with your own setup class
        results = run_tune(setup)
