from setuptools import setup, find_packages

setup(
    name=r"ray_utilities",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["ray[tuner,rllib]"],
)
