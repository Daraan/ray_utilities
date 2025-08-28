from setuptools import find_packages, setup

# TODO: make some installations optional
setup(
    name=r"ray_utilities",
    package_dir={"ray_utilities": "ray_utilities"},
    packages=find_packages(),
    include=["ray_utilities*"],
)
