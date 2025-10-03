# pyright: enableExperimentalFeatures=true
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import Never, NotRequired, ReadOnly, Required, TypedDict

from .common import BaseEnvRunnersResultsDict, BaseEvaluationResultsDict, ExtraItems

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "AlgorithmReturnData",
    "EnvRunnersResultsDict",
    "EvalEnvRunnersResultsDict",
    "EvaluationResultsDict",
    "StrictAlgorithmReturnData",
]


class EnvRunnersResultsDict(BaseEnvRunnersResultsDict, closed=False):
    """Environment runner results from Ray RLlib algorithm training.

    See Also:
        :data:`ray.rllib.utils.metrics.ENV_RUNNER_RESULTS`: Ray RLlib env runner metrics
        :class:`BaseEnvRunnersResultsDict`: Common base type
    """

    episode_return_max: NotRequired[float]
    """Maximum episode return in this iteration"""
    episode_return_min: NotRequired[float]
    """Minimum episode return in this iteration"""
    num_env_steps_sampled_lifetime: int
    """Amount of sampling steps taken for the training of the agent"""
    num_env_steps_sampled: int
    """Amount of sampling steps taken for the training of the agent in this iteration"""
    num_env_steps_passed_to_learner: NotRequired[int]
    """Amount of steps passed to the learner in this iteration. Custom key added by exact_sampling_callback"""
    num_env_steps_passed_to_learner_lifetime: NotRequired[int]
    """Amount of steps passed to the learner in this iteration. Custom key added by exact_sampling_callback"""

    # Additional Ray RLlib fields based on metrics imports
    episode_duration_sec_mean: NotRequired[float]
    """Mean duration of episodes in seconds"""
    num_module_steps_sampled: NotRequired[dict[str, int]]
    """Number of steps sampled per module"""
    num_module_steps_sampled_lifetime: NotRequired[dict[str, int]]
    """Lifetime number of steps sampled per module"""
    num_agent_steps_sampled: NotRequired[dict[str, int]]
    """Number of steps sampled per agent"""
    num_agent_steps_sampled_lifetime: NotRequired[dict[str, int]]
    """Lifetime number of steps sampled per agent"""

    environments: NotRequired[dict[str, Any]]
    """Custom key - environments info"""


class EvalEnvRunnersResultsDict(EnvRunnersResultsDict, total=False, extra_items=ExtraItems):
    episode_videos_best: NotRequired[list[NDArray]]
    """
    List, likely with on entry, of a 5D array

    # array is shape=3D -> An image (c, h, w).
    # array is shape=4D -> A batch of images (B, c, h, w).
    # array is shape=5D -> A video (1, L, c, h, w), where L is the length of the video
    """

    episode_videos_worst: NotRequired[list[NDArray]]
    """
    List, likely with on entry, of a 5D array

    # array is shape=3D -> An image (c, h, w).
    # array is shape=4D -> A batch of images (B, c, h, w).
    # array is shape=5D -> A video (1, L, c, h, w), where L is the length of the video
    """


class _EvaluationNoDiscreteDict(TypedDict, extra_items=ExtraItems):
    env_runners: EvalEnvRunnersResultsDict
    discrete: NotRequired[Never]


class EvaluationResultsDict(BaseEvaluationResultsDict, extra_items=ExtraItems):
    """Evaluation results structure for algorithm return data.

    See Also:
        :data:`ray.rllib.utils.metrics.EVALUATION_RESULTS`: Ray RLlib evaluation metrics
        :class:`BaseEvaluationResultsDict`: Common base type
    """

    env_runners: EvalEnvRunnersResultsDict
    discrete: NotRequired[_EvaluationNoDiscreteDict]
    """Custom key - evaluation results for discrete actions"""


class _RequiredEnvRunners(TypedDict, total=False, closed=False):
    env_runners: Required[EnvRunnersResultsDict]


class _NotRequiredEnvRunners(TypedDict, total=False, closed=False):
    env_runners: NotRequired[EnvRunnersResultsDict]


class _LearnerResults(TypedDict, extra_items=ReadOnly["int | float | _LearnerResults"]): ...


class LearnerAllModulesDict(_LearnerResults):
    num_env_steps_passed_to_learner: NotRequired[int]
    """Key added by custom connector"""
    num_env_steps_passed_to_learner_lifetime: NotRequired[int]
    """Key added by custom connector"""


class LearnerModuleDict(_LearnerResults):
    num_env_steps_passed_to_learner: NotRequired[int]
    """Key added by custom connector"""
    num_env_steps_passed_to_learner_lifetime: NotRequired[int]
    """Key added by custom connector"""


class LearnersMetricsDict(_LearnerResults):
    __all_modules__: LearnerAllModulesDict
    default_policy: LearnerModuleDict


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
