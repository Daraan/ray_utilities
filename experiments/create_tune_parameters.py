""" "
Define distributions for hyperparameter tuning and save them to a JSON file.
Distributions are defined using Optuna's distribution classes and can be converted to
Ray Domains for use in Ray Tune.
"""

# ruff: noqa: F401
from __future__ import annotations

import json
from math import log2
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    distribution_to_json,
)

from ray_utilities.config.parser.default_argument_parser import DefaultArgumentParser
from ray_utilities.dynamic_config.dynamic_buffer_update import MAX_DYNAMIC_BATCH_SIZE
from ray_utilities.setup.extensions import load_distributions_from_json
import time

__all__ = [
    "default_distributions",
    "load_distributions_from_json",
    "write_distributions_to_json",
]

GridSearch = Mapping[Literal["grid_search"], list[float | int | str]]

DistributionDefinition = BaseDistribution | GridSearch | Mapping[str, Mapping[str, float | bool]]

ACCUMULATION_BATCH_SIZE_BASE = DefaultArgumentParser.minibatch_size

base_exp = log2(ACCUMULATION_BATCH_SIZE_BASE)
assert base_exp.is_integer()
base_exp = int(base_exp)
max_exp = log2(MAX_DYNAMIC_BATCH_SIZE)
assert max_exp.is_integer()
max_exp = int(max_exp)

default_distributions: dict[str, DistributionDefinition] = {
    # "lr": {"qloguniform": {"lower": 5e-5, "upper": 1e-1, "q": 5e-5}},
    # qloguniform does not sample that well, samples that are close by and not spreading over the whole range
    # pure random would be nicer if a setting is totally not usable
    "lr": {"grid_search": sorted([*np.round(np.logspace(np.log2(5e-8), np.log2(0.0015), num=10, base=2), 8), 1e-4])},
    "batch_size": {"grid_search": [128, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2]},
    # NOTE: Upperbound of accumulate_gradients_every num_epochs * train_batch_size_per_learner / minibatch_size
    "accumulate_gradients_every": {
        "grid_search": [2**i for i in list(range(int(log2(MAX_DYNAMIC_BATCH_SIZE / ACCUMULATION_BATCH_SIZE_BASE)) + 1))]
    },  # assume 128 as base
    "minibatch_size": {"grid_search": [2**i for i in range(base_exp, max_exp + 1)]},
    # Need to skip too small minibatches, i.e. min 32, skip/resample. Alternatively use FloatDistribution
    "minibatch_scale": {"grid_search": [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]},
    # For high num_envs_per_env runner should adjust num_env_runners accordingly
    "num_envs_per_env_runner": {"grid_search": [1, 2, 4, 8, 16, 32]},
}


def write_distributions_to_json(
    distributions: dict[str, DistributionDefinition] | None,
    output_file: Path | str | None = None,
) -> Path:
    """Write the given distributions to a JSON file."""
    json_distributions: dict[str, Mapping] = {}
    if distributions is None:
        distributions = default_distributions

    for key, dist in distributions.items():
        if isinstance(dist, BaseDistribution):
            json_distributions[key] = json.loads(distribution_to_json(dist))
        else:
            json_distributions[key] = dist

    if output_file is None:
        output_file = Path(__file__).parent / "tune_parameters.json"
    elif isinstance(output_file, str):
        output_file = Path(output_file)

    lock_file = output_file.with_suffix(output_file.suffix + ".lock")
    while lock_file.exists():
        time.sleep(0.1)
    while True:
        try:
            lock_file.touch(exist_ok=False)
            break
        except FileExistsError:
            lock_file.unlink()
            time.sleep(0.1)
    try:
        with output_file.open("w") as f:
            json.dump(json_distributions, f, indent=2)
            f.write("\n")  # newline at end of file
    finally:
        if lock_file.exists():
            lock_file.unlink()
    return output_file


if __name__ == "__main__":
    write_distributions_to_json(default_distributions)
