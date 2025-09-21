"""
Type definitions for postprocessing metrics.

For algorithm return data, see `ray_utilities.typing.algorithm_return`
"""

# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeGuard

from typing_extensions import Never, NotRequired, Required, TypedDict

from .algorithm_return import EvaluationResultsDict, _EvaluationNoDiscreteDict
from .common import BaseEnvRunnersResultsDict, VideoTypes

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ray.rllib.utils.typing import AgentID, ModuleID

__all__ = [
    "LogMetricsDict",
]


class VideoMetricsDict(TypedDict, closed=True):
    video: VideoTypes.LogVideoTypes
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

    Extends the base type with additional optional fields used in logging.
    Most fields are NotRequired in metrics context as they may not always be available.

    See Also:
        :class:`BaseEnvRunnersResultsDict`: Common base type
        :data:`ray.rllib.utils.metrics.ENV_RUNNER_RESULTS`: Ray RLlib env runner metrics
    """

    episode_return_max: NotRequired[float]
    episode_return_min: NotRequired[float]
    num_env_steps_sampled_lifetime: NotRequired[int]
    num_env_steps_sampled: NotRequired[int]
    num_env_steps_passed_to_learner: NotRequired[int]
    """Custom key for exact current_steps added by exact_sampling_callback"""
    num_env_steps_passed_to_learner_lifetime: NotRequired[int]
    """Custom key for exact current_steps added by exact_sampling_callback"""
    episode_duration_sec_mean: NotRequired[float]
    num_module_steps_sampled: NotRequired[dict[ModuleID, int]]
    num_module_steps_sampled_lifetime: NotRequired[dict[ModuleID, int]]
    num_agent_steps_sampled: NotRequired[dict[AgentID, int]]
    num_agent_steps_sampled_lifetime: NotRequired[dict[AgentID, int]]


class _LogMetricsEvalEnvRunnersResultsDict(_LogMetricsEnvRunnersResultsDict, total=False):
    """Environment runner results for evaluation with video support."""

    episode_videos_best: VideoTypes.LogVideoTypes | VideoMetricsDict
    episode_videos_worst: VideoTypes.LogVideoTypes | VideoMetricsDict


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
