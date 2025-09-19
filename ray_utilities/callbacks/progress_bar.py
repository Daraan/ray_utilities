"""Progress bar utilities for Ray Tune experiments and training monitoring.

This module provides enhanced progress bar functionality for monitoring Ray Tune
experiments and training progress. It includes utilities for displaying training
and evaluation metrics in real-time with formatted progress indicators.

The progress bars integrate with both Ray's distributed progress tracking and
standard tqdm progress bars, providing a unified interface for experiment monitoring.

Key Features:
    - Real-time training and evaluation metric display
    - Automatic unit formatting (K, M for large numbers)
    - Support for both local and distributed progress tracking
    - Integration with Ray Tune's progress reporting
    - Colored and formatted metric displays

Example:
    Basic progress bar usage with training metrics::

        from ray_utilities.callbacks.progress_bar import update_pbar
        from tqdm import tqdm

        pbar = tqdm(total=1000000, desc="Training")

        # Update with training and evaluation results
        update_pbar(pbar, eval_results={"mean": 150.0}, train_results={"mean": 100.0, "max": 200.0})

Functions:
    :func:`update_pbar`: Update progress bar with training/evaluation metrics

Type Definitions:
    :class:`TrainRewardMetrics`: Training reward metric structure
    :class:`EvalRewardMetrics`: Evaluation reward metric structure
    :class:`DiscreteEvalRewardMetrics`: Discrete evaluation reward metrics

See Also:
    :mod:`ray.experimental.tqdm_ray`: Ray's distributed progress bars
    :mod:`tqdm`: Standard progress bar library
    :mod:`ray_utilities.misc`: Related utility functions for metric processing
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional, TypedDict, overload

from ray.experimental import tqdm_ray
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, EVALUATION_RESULTS
from tqdm import tqdm
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from ray_utilities import StrictAlgorithmReturnData
    from ray_utilities.typing import LogMetricsDict, RewardsDict


logger = logging.getLogger(__name__)


class TrainRewardMetrics(TypedDict, total=False):
    """Training reward metrics for progress bar display.

    Attributes:
        mean: Mean episode return during training
        max: Maximum episode return observed
        roll: Rolling average of episode returns (optional)
    """

    mean: float
    max: float
    roll: float


class EvalRewardMetrics(TypedDict):
    """Evaluation reward metrics for progress bar display.

    Attributes:
        mean: Mean episode return during evaluation
        roll: Rolling average of evaluation returns (optional)
    """

    mean: float
    roll: NotRequired[float]


class DiscreteEvalRewardMetrics(TypedDict):
    """Discrete evaluation reward metrics for progress bar display.

    Attributes:
        mean: Mean episode return during discrete evaluation
        roll: Rolling average of discrete evaluation returns (optional)
    """

    mean: float
    roll: NotRequired[float]


def _unit_division(amount: int) -> tuple[int, str]:
    """Convert large numbers to human-readable format with unit suffixes.

    Args:
        amount: The number to format

    Returns:
        A tuple of (scaled_amount, unit_suffix) where unit_suffix is one of
        "", "K" (thousands), or "M" (millions).

    Example:
        >>> _unit_division(1500)
        (1, "K")
        >>> _unit_division(2500000)
        (2, "M")
        >>> _unit_division(500)
        (500, "")
    """
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
    result: StrictAlgorithmReturnData,
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
    eval_constructed = False
    if metrics is not None and result is not None and rewards is not None:
        train_results = {
            "mean": metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN],
            "max": result[ENV_RUNNER_RESULTS].get("episode_return_max", float("nan")),
            "roll": rewards["running_reward"],
        }
    if (
        eval_results is None
        and rewards is not None
        and (
            (metrics and (eval_runner_results := metrics.get(EVALUATION_RESULTS, None)))
            or (result and (eval_runner_results := result.get(EVALUATION_RESULTS, None)))
        )
    ):
        eval_results = {
            "mean": eval_runner_results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN],
            "roll": rewards["running_eval_reward"],
        }
        eval_constructed = True
    if rewards:
        if eval_results is not None:
            if not eval_constructed:
                logger.warning("Both eval_results and rewards are provided. Using eval_results for evaluation metrics.")
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
            train_results = train_results.copy()
            train_mean = train_results.get("mean", float("nan"))
            train_max = train_results.get("max", float("nan"))

            # combine mean and roll
            if train_results.get("roll", None) is not None:
                mean_roll_value = f"{train_mean:.1f} (roll: {train_results.pop('roll'):.1f})"  # pyright: ignore[reportTypedDictNotRequiredAccess]
                train_results["mean"] = mean_roll_value  # pyright: ignore[reportGeneralTypeIssues]
            if train_mean == train_max or (math.isnan(train_mean) and math.isnan(train_max)):
                train_results.pop("max")
            train_results_str = {
                k: v if isinstance(v, str) else f"{v:>5.1f}" if k != "max" else f"{v:>4.0f}"
                for k, v in train_results.items()
            }
            lines = [f"Train {key}: {value}" for key, value in train_results_str.items()]
        else:
            lines = []
        if eval_results.get("mean") is not None and eval_results.get("roll") is not None:  # pyright: ignore[reportUnnecessaryComparison]
            eval_results = eval_results.copy()
            mean_roll_value = f"{eval_results['mean']:.1f} (roll: {eval_results.pop('roll'):.1f})"  # pyright: ignore[reportTypedDictNotRequiredAccess]
            eval_results["mean"] = mean_roll_value  # pyright: ignore[reportGeneralTypeIssues]
        lines += [
            f"Eval {key}: {value:>5.1f}" if not isinstance(value, str) else f"Eval {key}: {value}"
            for key, value in eval_results.items()
        ]
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
            pbar.n,
            iteration,
        )
    return (pbar.n, pbar.total)


def restore_pbar(state: TqdmState | RayTqdmState | RangeState) -> "tqdm_ray.tqdm | tqdm | range":
    """Restores the progress bar from a saved state, returns a new object"""
    if isinstance(state, dict):  # ray tqdm state
        pbar = tqdm_ray.tqdm(range(state["x"], state["total"]), total=state["total"])
        pbar._unit = state["unit"]
        pbar._x = state["x"]
        return pbar
    if len(state) > 2:  # range state
        return range(*state)
    start, stop = state
    if stop is None:
        raise ValueError("Cannot restore a progress bar with no total value.")
    return tqdm(range(start, stop))
