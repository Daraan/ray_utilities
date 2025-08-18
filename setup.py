from setuptools import find_packages, setup
import os
import time

version: str = "0.2.2"
if "DEV_VERSION" in os.environ and "RC_VERSION" in os.environ:
    raise ValueError("Cannot set both DEV_VERSION and RC_VERSION at the same time.")
if "DEV_VERSION" in os.environ:
    version += f".dev{int(time.time())}"  # Append a timestamp for development versions
elif "RC_VERSION" in os.environ:
    version += f".rc{int(time.time())}"


# TODO: make some installations optional -> todo in pyproject.toml
setup(
    name=r"ray_utilities",
    version=version,
    package_dir={"ray_utilities": "ray_utilities"},
    packages=find_packages(),
    include=["ray_utilities*"],
    install_requires=[
        "ray[tune,rllib]",
        "typed-argument-parser",
        "tqdm",
        "colorlog",
        "opencv-python",
        "comet_ml",
        "wandb",
        "optuna",
        "python-dotenv",
        "torch",
        "jax",
    ],
    extras_require={
        "test": ["pytest", "pytest-timeout", "pytest-subtests", "debugpy"],
        "wandb": ["wandb"],
        "comet": ["comet_ml"],
        "optuna": ["optuna"],
        "jax": ["jax", "flax"],
    },
)
