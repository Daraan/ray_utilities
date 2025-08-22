# Currently cannot use variables, even if final or literal, with TypedDict
# Uses PEP 728 not yet released in typing_extensions
# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import Never, NotRequired, ReadOnly, Required, TypedDict

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
    num_env_steps_sampled_lifetime: int
    """Amount of sampling steps taken for the training of the agent"""
    num_env_steps_sampled: int
    """Amount of sampling steps taken for the training of the agent in this iteration"""
    num_env_steps_passed_to_learner: NotRequired[int]
    """
    Amount of steps passed to the learner in this iteration

    Custom key added by exact_sampling_callback.
    """
    num_env_steps_passed_to_learner_lifetime: NotRequired[int]
    """
    Amount of steps passed to the learner in this iteration

    Custom key added by exact_sampling_callback.
    """


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

    class EvaluationResultsDict(TypedDict, extra_items=ExtraItems):
        env_runners: EvalEnvRunnersResultsDict
        discrete: NotRequired[_EvaluationNoDiscreteDict]
        """Custom key - evaluation results for discrete actions"""
        evaluated_this_step: NotRequired[bool]
        """Custom key"""

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


class _LearnerResults(TypedDict, extra_items=ReadOnly["int | float | _LearnerResults"]): ...


class LearnerAllModulesDict(_LearnerResults):
    num_env_steps_passed_to_learner: NotRequired[int]
    """Key added by exact_sampling_callback"""
    num_env_steps_passed_to_learner_lifetime: NotRequired[int]
    """Key added by exact_sampling_callback"""


class LearnerModuleDict(_LearnerResults):
    num_env_steps_passed_to_learner: NotRequired[int]
    """Key added by exact_sampling_callback"""
    num_env_steps_passed_to_learner_lifetime: NotRequired[int]
    """Key added by exact_sampling_callback"""


class LearnersMetricsDict(_LearnerResults):
    __all_modules__: LearnerAllModulesDict
    default_policy: LearnerModuleDict


if _PEP_728_AVAILABLE or TYPE_CHECKING:

    class _AlgoReturnDataWithoutEnvRunners(TypedDict, total=False, extra_items=ExtraItems):
        done: Required[bool]
        evaluation: EvaluationResultsDict
        env_runners: Required[EnvRunnersResultsDict] | NotRequired[EnvRunnersResultsDict]
        learners: Required[LearnersMetricsDict]
        # Present in rllib results
        training_iteration: Required[int]
        """The number of times train.report() has been called"""

        config: Required[dict[str, Any]]

        should_checkpoint: bool

        comment: str
        trial_id: Required[int | str]
        episodes_total: int
        episodes_this_iter: int

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

    class AlgorithmReturnData(_AlgoReturnDataWithoutEnvRunners, _NotRequiredEnvRunners, extra_items=ExtraItems):
        """
        See Also:
            - https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html#tune-autofilled-metrics
        """

    class StrictAlgorithmReturnData(  # pyright: ignore[reportIncompatibleVariableOverride]
        _AlgoReturnDataWithoutEnvRunners, _RequiredEnvRunners, extra_items=ExtraItems
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
