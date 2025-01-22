from setuptools import setup, find_packages

setup(
    name=r"ray_utilities",
    version="0.0.2",
    package_dir={"ray_utilities": "ray_utilities"},
    packages=find_packages(),
    include=["ray_utilities*"],
    install_requires=["ray[tuner,rllib]"],
)
