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
from typing import Literal

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

__all__ = [
    "default_distributions",
    "load_distributions_from_json",
    "write_distributions_to_json",
]

GridSearch = dict[Literal["grid_search"], list[float | int | str]]

ACCUMULATION_BATCH_SIZE_BASE = DefaultArgumentParser.minibatch_size

base_exp = log2(ACCUMULATION_BATCH_SIZE_BASE)
assert base_exp.is_integer()
base_exp = int(base_exp)
max_exp = log2(MAX_DYNAMIC_BATCH_SIZE)
assert max_exp.is_integer()
max_exp = int(max_exp)

default_distributions: dict[str, BaseDistribution | GridSearch] = {
    # TODO log and step cannot be used togher
    # "lr": FloatDistribution(5e-5, 1e-1, log=True, step=5e-5),
    "batch_size": {"grid_search": [128, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2]},
    # NOTE: Upperbound of gradient_accumulation num_epochs * train_batch_size_per_learner / minibatch_size
    "gradient_accumulation": {"grid_search": list(range(1, max_exp + 1))},  # assume 512 as base
    "minibatch_size": {"grid_search": [2**i for i in range(base_exp, max_exp + 1)]},
    # Need to skip too small minibatches, i.e. min 32, skip/resample. Alternatively use FloatDistribution
    "minibatch_scale": {"grid_search": [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]},
}


def write_distributions_to_json(
    distributions: dict[str, BaseDistribution | GridSearch] | None,
    output_file: Path | None = None,
) -> Path:
    """Write the given distributions to a JSON file."""
    json_distributions: dict[str, dict] = {}
    if distributions is None:
        distributions = default_distributions

    for key, dist in distributions.items():
        if isinstance(dist, BaseDistribution):
            json_distributions[key] = json.loads(distribution_to_json(dist))
        else:
            json_distributions[key] = {"grid_search": dist["grid_search"]}

    if output_file is None:
        output_file = Path(__file__).parent / "tune_parameters.json"

    with output_file.open("w") as f:
        json.dump(json_distributions, f, indent=2)
        f.write("\n")  # newline at end of file
    return output_file


if __name__ == "__main__":
    write_distributions_to_json(default_distributions)
