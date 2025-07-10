from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional, TypedDict, overload

from ray.experimental import tqdm_ray
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN
from tqdm import tqdm
from typing_extensions import NotRequired

if TYPE_CHECKING:

    from ray_utilities import StrictAlgorithmReturnData
    from ray_utilities.typing import LogMetricsDict, RewardsDict


logger = logging.getLogger(__name__)

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

@overload
def update_pbar(
    pbar: "tqdm_ray.tqdm | tqdm",
    *,
    eval_results: EvalRewardMetrics,
    train_results: Optional[TrainRewardMetrics] = None,
    discrete_eval_results: Optional[DiscreteEvalRewardMetrics] = None,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> None: ...

@overload
def update_pbar(
    pbar: "tqdm_ray.tqdm | tqdm",
    *,
    rewards: RewardsDict,
    metrics: LogMetricsDict,
    result: StrictAlgorithmReturnData ,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> None: ...

@overload
def update_pbar(
    pbar: "tqdm_ray.tqdm | tqdm",
    *,
    rewards: RewardsDict,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> None: ...

def update_pbar(
    pbar: "tqdm_ray.tqdm | tqdm",
    *,
    rewards: Optional[RewardsDict] = None,
    metrics: Optional[LogMetricsDict] = None,
    result: Optional[StrictAlgorithmReturnData] = None,
    eval_results: Optional[EvalRewardMetrics] = None,
    train_results: Optional[TrainRewardMetrics] = None,
    discrete_eval_results: Optional[DiscreteEvalRewardMetrics] = None,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
):
    """Updates the progress bar with the latest training and evaluation metrics."""
    if metrics is not None and result is not None and rewards is not None:
        train_results = {
            "mean": metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN],
            "max": result[ENV_RUNNER_RESULTS].get("episode_return_max", float("nan")),
            "roll": rewards["running_reward"],
        }
    if rewards:
        if eval_results is not None:
            logger.warning(
                "Both eval_results and rewards are provided. "
                "Using eval_results for evaluation metrics."
            )
        else:
            eval_results = {
                "mean": rewards["eval_mean"],
                "roll": rewards["running_eval_reward"],
            }
        if discrete_eval_results is not None:
            logger.warning(
                "Both discrete_eval_results and rewards are provided. "
                "Using discrete_eval_results for discrete evaluation metrics."
            )
        else:
            discrete_eval_results = (
                {
                    "mean": rewards["disc_eval_mean"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    "roll": rewards["disc_running_eval_reward"],  # pyright: ignore[reportTypedDictNotRequiredAccess]
                }
                if rewards.get("disc_eval_mean") is not None and rewards.get("disc_running_eval_reward") is not None
                else None
            )
    elif eval_results is None:
        raise ValueError("Either eval_results or rewards must be provided to update the progress bar.")

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
        logger.error("KeyError while updating progress bar: %s.", e)
    pbar.set_description(description)


_TotalValue = int | None
RangeState = tuple[int, int, int]
TqdmState = tuple[int, _TotalValue]
RayTqdmState = dict[str, Any]


@overload
def save_pbar_state(pbar: "range", iteration: int) -> RangeState: ...


@overload
def save_pbar_state(pbar: "tqdm", iteration: Optional[int] = None) -> TqdmState: ...
@overload
def save_pbar_state(pbar: "tqdm_ray.tqdm", iteration: Optional[int] = None) -> RayTqdmState: ...

def save_pbar_state(
    pbar: "tqdm_ray.tqdm | tqdm | range", iteration: Optional[int] = None
) -> tuple[int, int | None] | RayTqdmState | tuple[int, int, int]:
    if isinstance(pbar, range):
        if not iteration:
            raise ValueError("Iteration must be provided when saving a range progress bar state.")
        return (iteration, pbar.stop, pbar.step)
    if isinstance(pbar, tqdm_ray.tqdm):
        return pbar._get_state()
    if iteration is not None and pbar.n != iteration:
        logger.error(
            "Progress bar n (%d) does not match the provided iteration (%d). "
            "Saving the progress bar state with the current n value.",
            pbar.n, iteration
        )
    return (pbar.n, pbar.total)

def restore_pbar(state: TqdmState | RayTqdmState | RangeState) -> "tqdm_ray.tqdm | tqdm | range":
    """Restores the progress bar from a saved state, returns a new object"""
    if isinstance(state, dict): # ray tqdm state
        pbar = tqdm_ray.tqdm(range(state["x"], state["total"]))
        pbar._unit = state["unit"]
        return pbar
    if len(state) > 2:  # range state
        return range(*state)
    start, stop = state
    if stop is None:
        raise ValueError("Cannot restore a progress bar with no total value.")
    return tqdm(range(start, stop))
