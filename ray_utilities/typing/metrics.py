"""
Type definitions for postprocessing metrics.

For algorithm return data, see `ray_utilities.typing.algorithm_return`
"""

# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, TypeGuard

from typing_extensions import Never, NotRequired, Required, TypedDict

from .algorithm_return import EvaluationResultsDict, _EvaluationNoDiscreteDict
from .common import BaseEnvRunnersResultsDict, BaseEvaluationResultsDict, CommonVideoTypes

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from ray.rllib.utils.typing import AgentID, ModuleID
    from wandb import Video  # pyright: ignore[reportMissingImports] # TODO: can we use this as log type as well?

__all__ = [
    "LogMetricsDict",
]

Shape4D = CommonVideoTypes.Shape4D
Array4D = CommonVideoTypes.Array4D
Shape5D = CommonVideoTypes.Shape5D
Array5D = CommonVideoTypes.Array5D

LOG_METRICS_VIDEO_TYPES: TypeAlias = CommonVideoTypes.LogVideoTypes
"""
Log types for videos in LogMetricsDict.

Either a :class:`wandb.Video` object, a string pointing to a video file to upload,
a 5D Numpy array representing a video, or a list of 4D or 5D Numpy arrays representing videos.
The list should either be a single 5D array (N, T, C, H, W) or multiple 4D arrays (B, C, H, W)
to be stacked to a 5D array.
"""


class VideoMetricsDict(TypedDict, closed=True):
    video: LOG_METRICS_VIDEO_TYPES
    """
    A 5D numpy array representing a video; or a string pointing to a video file to upload.

    Should be a list of a 5D numpy video array representing a video.
    """
    reward: float
    video_path: NotRequired[str]
    """If a video file already exists for re-use, this is the path to the video file."""


class _WarnVideosToEnvRunners(TypedDict):
    episode_videos_best: NotRequired[Annotated[Never, "needs to be in env_runners"]]
    episode_videos_worst: NotRequired[Annotated[Never, "needs to be in env_runners"]]


class _LogMetricsEnvRunnersResultsDict(BaseEnvRunnersResultsDict):
    """Environment runner results optimized for logging metrics.
    
    Extends the base type with additional optional fields used in logging,
    such as module and agent-specific step counts.
    """
    episode_return_mean: float  # Keep required for logging
    num_module_steps_sampled: NotRequired[dict[ModuleID, int]]
    num_module_steps_sampled_lifetime: NotRequired[dict[ModuleID, int]]
    num_agent_steps_sampled: NotRequired[dict[AgentID, int]]
    num_agent_steps_sampled_lifetime: NotRequired[dict[AgentID, int]]


class _LogMetricsEvalEnvRunnersResultsDict(_LogMetricsEnvRunnersResultsDict, total=False):
    """
    Either a 5D Numpy array representing a video, or a dict with "video" and "reward" keys,
    representing the video, or a string pointing to a video file to upload.
    """

    episode_videos_best: LOG_METRICS_VIDEO_TYPES | VideoMetricsDict
    episode_videos_worst: LOG_METRICS_VIDEO_TYPES | VideoMetricsDict


class _LogMetricsEvaluationResultsWithoutDiscreteDict(_EvaluationNoDiscreteDict, _WarnVideosToEnvRunners):
    env_runners: Required[_LogMetricsEvalEnvRunnersResultsDict]  # pyright: ignore[reportIncompatibleVariableOverride]
    discrete: NotRequired[Never]


class _LogMetricsEvaluationResultsDict(EvaluationResultsDict, _WarnVideosToEnvRunners):
    env_runners: _LogMetricsEvalEnvRunnersResultsDict  # pyright: ignore[reportIncompatibleVariableOverride]
    discrete: NotRequired[_LogMetricsEvaluationResultsWithoutDiscreteDict]  # pyright: ignore[reportIncompatibleVariableOverride]


class _LogMetricsBase(TypedDict):
    training_iteration: int
    """The number of times train.report() has been called"""

    current_step: int
    """
    The current step in the training process, usually the number of environment steps sampled.

    For exact sampling use:

        - "learners/(__all_modules__ | default_policy)/num_env_steps_passed_to_learner_lifetime"
            Requires: ``RemoveMaskedSamplesConnector`` (+ ``exact_sampling_callback`` at best)
        - "env_runners/num_env_steps_sampled_lifetime"
            Requires: ``exact_sampling_callback``

    Otherwise use::

            env_runners/num_env_steps_sampled_lifetime
    """

    done: bool
    timers: NotRequired[dict[str, dict[str, Any] | Any]]
    learners: NotRequired[dict[ModuleID | Literal["__all_modules__"], dict[str, Any] | Any]]

    fault_tolerance: NotRequired[Any]
    env_runner_group: NotRequired[Any]
    num_env_steps_sampled_lifetime: NotRequired[int]
    num_env_steps_sampled_lifetime_throughput: NotRequired[float]
    """Mean time in seconds between two logging calls to num_env_steps_sampled_lifetime"""

    should_checkpoint: NotRequired[bool]
    """If True, the tuner should checkpoint this step."""

    batch_size: NotRequired[int]
    """Current train_batch_size_per_learner. Should be logged in experiments were it can change."""

    num_training_step_calls_per_iteration: NotRequired[int]
    """How training_steps was called between two train.report() calls."""

    config: NotRequired[dict[str, Any]]
    """Algorithm config used for this training step."""


class LogMetricsDict(_LogMetricsBase):
    """Stays true to RLlib's naming."""

    env_runners: _LogMetricsEnvRunnersResultsDict
    evaluation: _LogMetricsEvaluationResultsDict


class AutoExtendedLogMetricsDict(LogMetricsDict):
    """
    Auto filled in keys after train.report.

    Use this in Callbacks

    See Also:
        - https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        - Trial.last_result
    """

    done: bool
    training_iteration: int  # auto filled in
    """The number of times train.report() has been called"""
    trial_id: int | str  # auto filled in


FlatLogMetricsDict = TypedDict(
    "FlatLogMetricsDict",
    {
        "training_iteration": int,
        "env_runners/episode_return_mean": float,
        "evaluation/env_runners/episode_return_mean": float,
        "evaluation/env_runners/episode_videos_best": NotRequired["str | NDArray"],
        "evaluation/env_runners/episode_videos_best/reward": NotRequired[float],
        "evaluation/env_runners/episode_videos_best/video": NotRequired["NDArray"],
        "evaluation/env_runners/episode_videos_best/video_path": NotRequired["str"],
        "evaluation/env_runners/episode_videos_worst": NotRequired["NDArray | str"],
        "evaluation/env_runners/episode_videos_worst/reward": NotRequired[float],
        "evaluation/env_runners/episode_videos_worst/video": NotRequired["NDArray"],
        "evaluation/env_runners/episode_videos_worst/video_path": NotRequired["str"],
        "evaluation/discrete/env_runners/episode_return_mean": float,
        "evaluation/discrete/env_runners/episode_videos_best": NotRequired["str | NDArray"],
        "evaluation/discrete/env_runners/episode_videos_best/reward": NotRequired[float],
        "evaluation/discrete/env_runners/episode_videos_best/video": NotRequired["NDArray"],
        "evaluation/discrete/env_runners/episode_videos_best/video_path": NotRequired["str"],
        "evaluation/discrete/env_runners/episode_videos_worst": NotRequired["str | NDArray"],
        "evaluation/discrete/env_runners/episode_videos_worst/reward": NotRequired[float],
        "evaluation/discrete/env_runners/episode_videos_worst/video": NotRequired["NDArray"],
        "evaluation/discrete/env_runners/episode_videos_worst/video_path": NotRequired["str"],
    },
    closed=False,
)


class _NewLogMetricsEvaluationResultsWithoutDiscreteDict(_LogMetricsEvalEnvRunnersResultsDict):
    discrete: NotRequired[Never]


class _NewLogMetricsEvaluationResultsDict(_LogMetricsEvalEnvRunnersResultsDict):
    discrete: NotRequired[_NewLogMetricsEvaluationResultsWithoutDiscreteDict]


class NewLogMetricsDict(_LogMetricsBase):
    """
    Changes:
        env_runners -> training
        evaluation.env_runners -> evaluation

        if only "__all_modules__" and "default_policy" are present in learners,
        merges them.
    """

    training: _LogMetricsEnvRunnersResultsDict
    evaluation: _NewLogMetricsEvaluationResultsDict


AnyLogMetricsDict = LogMetricsDict | NewLogMetricsDict


class NewAutoExtendedLogMetricsDict(NewLogMetricsDict):
    """
    Auto filled in keys after train.report.

    Use this in Callbacks

    See Also:
        - https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        - Trial.last_result
    """

    done: bool
    training_iteration: int  # auto filled in
    """The number of times train.report() has been called"""
    trial_id: int | str  # auto filled in


AnyAutoExtendedLogMetricsDict = AutoExtendedLogMetricsDict | NewAutoExtendedLogMetricsDict

NewFlatLogMetricsDict = TypedDict(
    "NewFlatLogMetricsDict",
    {
        "training_iteration": int,
        "training/episode_return_mean": float,
        "evaluation/episode_return_mean": float,
        "evaluation/episode_videos_best": NotRequired["str | NDArray"],
        "evaluation/episode_videos_best/reward": NotRequired[float],
        "evaluation/episode_videos_best/video": NotRequired["NDArray"],
        "evaluation/episode_videos_best/video_path": NotRequired["str"],
        "evaluation/episode_videos_worst": NotRequired["NDArray | str"],
        "evaluation/episode_videos_worst/reward": NotRequired[float],
        "evaluation/episode_videos_worst/video": NotRequired["NDArray"],
        "evaluation/episode_videos_worst/video_path": NotRequired["str"],
        "evaluation/discrete/episode_return_mean": float,
        "evaluation/discrete/episode_videos_best": NotRequired["str | NDArray"],
        "evaluation/discrete/episode_videos_best/reward": NotRequired[float],
        "evaluation/discrete/episode_videos_best/video": NotRequired["NDArray"],
        "evaluation/discrete/episode_videos_best/video_path": NotRequired["str"],
        "evaluation/discrete/episode_videos_worst": NotRequired["str | NDArray"],
        "evaluation/discrete/episode_videos_worst/reward": NotRequired[float],
        "evaluation/discrete/episode_videos_worst/video": NotRequired["NDArray"],
        "evaluation/discrete/episode_videos_worst/video_path": NotRequired["str"],
    },
    closed=False,
)

AnyFlatLogMetricsDict = FlatLogMetricsDict | NewFlatLogMetricsDict

# region TypeGuards


def has_video_key(
    dir: dict | _LogMetricsEvalEnvRunnersResultsDict, video_key: Literal["episode_videos_best", "episode_videos_worst"]
) -> TypeGuard[_LogMetricsEvalEnvRunnersResultsDict]:
    return video_key in dir


# endregion TypeGuards
