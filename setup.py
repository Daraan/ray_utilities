from setuptools import find_packages, setup

# TODO: make some installations optional
setup(
    name=r"ray_utilities",
    version="0.0.11.dev",
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
)
