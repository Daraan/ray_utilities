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


def _unit_division(amount: int) -> tuple[int, str]:
    """Divides the amount by 1_000_000 or 1_000 and returns the unit."""
    if amount >= 1_000_000:
        return amount // 1_000_000, "M"
    if amount >= 1_000:
        return amount // 1_000, "K"
    return amount, ""


def update_pbar(
    pbar: "tqdm_ray.tqdm | tqdm",
    *,
    eval_results: EvalRewardMetrics,
    train_results: Optional[TrainRewardMetrics] = None,
    discrete_eval_results: Optional[DiscreteEvalRewardMetrics] = None,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
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
                f"Train {key}: {value:>5.1f}" if key != "max" else f"Train {key}: {value:>4.0f}"
                for key, value in train_results.items()
            ]
        else:
            lines = []
        lines += [f"Eval {key}: {value:>5.1f}" for key, value in eval_results.items()]
        if discrete_eval_results:
            lines += [f"Disc Eval {key}: {value:>5.1f}" for key, value in discrete_eval_results.items()]
        if current_step is not None:
            current_step, step_unit = _unit_division(current_step)
            step_count = f"Step {current_step:>3d}{step_unit}"
            if total_steps is not None:
                total_steps, total_step_unit = _unit_division(total_steps)
                step_count += f"/{total_steps}{total_step_unit}"
            else:
                step_count += "/?"
            lines.append(step_count)
        description = " |".join(lines)
    except KeyError as e:
        description = ""
        print("KeyError in update_pbar", e)
    pbar.set_description(description)
