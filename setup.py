from setuptools import find_packages, setup

setup(
    name=r"ray_utilities",
    version="0.0.4",
    package_dir={"ray_utilities": "ray_utilities"},
    packages=find_packages(),
    include=["ray_utilities*"],
    install_requires=[
        "ray[tuner,rllib]",
        "typed-argument-parser",
        "tqdm",
        "opencv-python",
        "comet_ml",
        "wandb",
        "optuna",
        "dotenv",
        "torch",
        "jax",
    ],
)
