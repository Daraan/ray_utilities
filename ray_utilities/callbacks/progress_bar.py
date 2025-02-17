from __future__ import annotations
import math
from typing import Optional, TypedDict, TYPE_CHECKING

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from tqdm import tqdm
    from ray.experimental import tqdm_ray


class TrainRewardMetrics(TypedDict, total=False):
    mean: float
    max: float
    roll: float


class EvalRewardMetrics(TypedDict):
    mean: float
    roll: NotRequired[float]


class DiscreteEvalRewardMetrics(TypedDict):
    mean: float
    roll: NotRequired[float]


def update_pbar(
    pbar: "tqdm_ray.tqdm | tqdm",
    *,
    eval_results: EvalRewardMetrics,
    train_results: Optional[TrainRewardMetrics] = None,
    discrete_eval_results: Optional[DiscreteEvalRewardMetrics] = None,
):
    try:
        if train_results:
            # Remember float("nan") != float("nan")
            train_mean = train_results.get("mean", float("nan"))
            train_max = train_results.get("max", float("nan"))

            if train_mean == train_max or (math.isnan(train_mean) and math.isnan(train_max)):
                train_results = train_results.copy()
                train_results.pop("max")
            lines = [
                f"Train R {key}: {value:>6.1f}" if key != "max" else f"Train R {key}: {value:>4.0f}"
                for key, value in train_results.items()
            ]
        else:
            lines = []
        lines += [f"Eval R {key}: {value:>6.1f}" for key, value in eval_results.items()]
        if discrete_eval_results:
            lines += [f"Disc Eval R {key}: {value:>6.1f}" for key, value in discrete_eval_results.items()]
        description = " |".join(lines)
    except KeyError as e:
        description = ""
        print("KeyError in update_pbar", e)
    pbar.set_description(description)
