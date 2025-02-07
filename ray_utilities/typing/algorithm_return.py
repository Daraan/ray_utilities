# Currently cannot use variables, even if final or literal, with TypedDict
# Uses PEP 728 not yet released in typing_extensions
# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Never, NotRequired, Required, TypedDict

from . import _PEP_728_AVAILABLE, ExtraItems

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "AlgorithmReturnData",
    "EnvRunnersResultsDict",
    "EvaluationResultsDict",
    "StrictAlgorithmReturnData",
]


class EnvRunnersResultsDict(TypedDict, closed=False):
    episode_return_mean: float
    episode_return_max: float
    episode_return_min: float


if _PEP_728_AVAILABLE or TYPE_CHECKING:

    class EvalEnvRunnersResultsDict(EnvRunnersResultsDict, total=False, extra_items=ExtraItems):
        episode_videos_best: list[NDArray]
        """
        List, likely with on entry, of a 5D array

        # array is shape=3D -> An image (c, h, w).
        # array is shape=4D -> A batch of images (B, c, h, w).
        # array is shape=5D -> A video (1, L, c, h, w), where L is the length of the
        """
        episode_videos_worst: list[NDArray]
        """
        List, likely with on entry, of a 5D array

        # array is shape=3D -> An image (c, h, w).
        # array is shape=4D -> A batch of images (B, c, h, w).
        # array is shape=5D -> A video (1, L, c, h, w), where L is the length of the
        """
else:

    class EvalEnvRunnersResultsDict(EnvRunnersResultsDict, total=False):
        episode_videos_best: list[NDArray]
        episode_videos_worst: list[NDArray]


if _PEP_728_AVAILABLE or TYPE_CHECKING:

    class _EvaluationNoDiscreteDict(TypedDict, extra_items=ExtraItems):
        env_runners: EvalEnvRunnersResultsDict
        discrete: NotRequired[Never]

    class EvaluationResultsDict(TypedDict, total=False, extra_items=ExtraItems):
        env_runners: Required[EvalEnvRunnersResultsDict]
        discrete: _EvaluationNoDiscreteDict
else:

    class _EvaluationNoDiscreteDict(TypedDict):
        env_runners: EvalEnvRunnersResultsDict
        discrete: NotRequired[Never]

    class EvaluationResultsDict(TypedDict, total=False):
        env_runners: Required[EvalEnvRunnersResultsDict]
        discrete: _EvaluationNoDiscreteDict


class _RequiredEnvRunners(TypedDict, total=False, closed=False):
    env_runners: Required[EnvRunnersResultsDict]


class _NotRequiredEnvRunners(TypedDict, total=False, closed=False):
    env_runners: NotRequired[EnvRunnersResultsDict]


if _PEP_728_AVAILABLE or TYPE_CHECKING:

    class _AlgoReturnDataWithoutEnvRunners(TypedDict, total=False, extra_items=ExtraItems):
        done: Required[bool]
        evaluation: EvaluationResultsDict
        env_runners: Required[EnvRunnersResultsDict] | NotRequired[EnvRunnersResultsDict]
        # Present in rllib results
        training_iteration: Required[int]
        """The number of times train.report() has been called"""

        should_checkpoint: bool

        comment: str
        trial_id: Required[int | str]
        episodes_total: int
        episodes_this_iter: int
        learners: dict[str, dict[str, float | int]]

        # Times
        timers: dict[str, float]
        timestamp: int
        time_total_s: float
        time_this_iter_s: float
        """
        Runtime of the current training iteration in seconds
        i.e. one call to the trainable function or to _train() in the class API.
        """

        # System results
        date: str
        node_ip: str
        hostname: str
        pid: int

        # Restore
        iterations_since_restore: int
        """The number of times train.report has been called after restoring the worker from a checkpoint"""

        time_since_restore: int
        """Time in seconds since restoring from a checkpoint."""

        timesteps_since_restore: int
        """Number of timesteps since restoring from a checkpoint"""

    class AlgorithmReturnData(
        _AlgoReturnDataWithoutEnvRunners, _NotRequiredEnvRunners, total=False, extra_items=ExtraItems
    ):
        """
        See Also:
            - https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        """

    class StrictAlgorithmReturnData(  # pyright: ignore[reportIncompatibleVariableOverride]
        _AlgoReturnDataWithoutEnvRunners, _RequiredEnvRunners, total=False, extra_items=ExtraItems
    ):
        """
        Return data with env_runners present

        See Also:
            - https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        """
else:
    # PEP 728 not yet released in typing_extensions
    AlgorithmReturnData = dict
    StrictAlgorithmReturnData = dict
    _AlgoReturnDataWithoutEnvRunners = dict
