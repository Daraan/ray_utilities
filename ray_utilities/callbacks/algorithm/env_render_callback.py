from __future__ import annotations

# ruff: noqa: ARG001,ARG002
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.images import resize

from ray_utilities.constants import EPISODE_BEST_VIDEO, EPISODE_WORST_VIDEO, GYM_V1

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium.vector import VectorEnv

    if GYM_V1:
        from gymnasium.wrappers.vector import DictInfoToList  # pyright: ignore[reportMissingImports]
    else:
        from gymnasium.wrappers.vector_list_info import (  # pyright: ignore[reportMissingImports]
            VectorListInfo as DictInfoToList,
        )
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from ray.rllib.utils.typing import EpisodeType
    from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

_EnvType: TypeAlias = "gym.Env | DictInfoToList | VectorEnv"

_Condition: TypeAlias = (
    # With optional EnvRunner
    Callable[
        [
            "AdvEnvRenderCallback",
            "EpisodeType",
            Optional["EnvRunner"],
            _EnvType,
            Optional["MetricsLogger"],
            Optional["RLModule"],
        ],
        bool | None,
    ]
    # Without EnvRunner
    | Callable[
        [
            "AdvEnvRenderCallback",
            "EpisodeType",
            "EnvRunner",
            _EnvType,
            Optional["MetricsLogger"],
            Optional["RLModule"],
        ],
        bool | None,
    ]
)
_C = TypeVar("_C", bound=_Condition)


def _check_condition(func: _C) -> _C:
    """Decorator to check if a function's signature satisfies the _Condition type alias."""
    return func


@_check_condition
def always_render(
    callback: AdvEnvRenderCallback,
    episode: EpisodeType,
    env_runner: Optional["EnvRunner"],
    env: _EnvType,
    metrics_logger: Optional[MetricsLogger],
    rl_module: Optional[RLModule],
) -> Literal[True]:
    """Always evaluates to True and saves the video for the current episode."""
    return True


@_check_condition
def render_during_evaluation(
    callback: AdvEnvRenderCallback,
    episode: EpisodeType,
    env_runner: EnvRunner,
    env: _EnvType,
    metrics_logger: Optional[MetricsLogger],
    rl_module: Optional[RLModule],
) -> bool | None:
    return env_runner.config.in_evaluation


def make_render_callback(
    *,
    save_condition: _Condition = always_render,
    render_condition: _Condition = render_during_evaluation,
) -> type[AdvEnvRenderCallback]:
    """
    Creates a `AdvEnvRenderCallback` with custom trigger condition.

    Args:
        save_condition: Evaluated at episode end if the video should be saved in the metrics logger.
        render_condition: Evaluated at episode start if the environment should be rendered this episode.
    """

    class AdvEnvRenderCallbackWithTrigger(AdvEnvRenderCallback):
        episode_save_condition = save_condition
        episode_render_condition = render_condition

    return AdvEnvRenderCallbackWithTrigger


class AdvEnvRenderCallback(DefaultCallbacks):
    """A custom callback to render the environment.

    This can be used to create videos of the episodes for some or all EnvRunners
    and some or all env indices (in a vectorized env). These videos can then
    be sent to e.g. WandB as shown in this example script here.

    We override the `on_episode_step` method to create a single ts render image
    and temporarily store it in the Episode object.
    """

    episode_render_condition: ClassVar[_Condition] = render_during_evaluation
    """
    Required condition to render the environment on each episode step.
    Should only depend on episode-constant values for example depend on the episode number.

    Note:
        If this condition is False resources will be saved.

    Raises:
        ValueError: If the condition changes during an episode
    """

    episode_save_condition: ClassVar[_Condition] = always_render
    """
    Condition evaluated at the end of an episode if the saved video should be saved in the metrics logger.

    """

    def __init__(self, env_runner_indices: Optional[Sequence[int]] = None):
        # TODO: add some specification for image size
        super().__init__()
        # Only render and record on certain EnvRunner indices?
        self.env_runner_indices = env_runner_indices
        # Per sample round (on this EnvRunner), we want to only log the best- and
        # worst performing episode's videos in the custom metrics. Otherwise, too much
        # data would be sent to WandB.
        self.best_episode_and_return = (None, float("-inf"))
        self.worst_episode_and_return = (None, float("inf"))
        self._is_evaluating = None
        self._render_this_episode = None

    def on_episode_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        episode: EpisodeType,
        env_runner: EnvRunner = None,  # pyright: ignore[reportArgumentType]
        metrics_logger: Optional[MetricsLogger] = None,
        env: gym.Env = None,  # pyright: ignore[reportArgumentType]
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        """Resets the temporary render image storage for the episode."""
        # Reset the temporary render image storage.
        render_cond = self.episode_render_condition(episode, env_runner, env, metrics_logger, rl_module)
        if render_cond is not None:
            self._render_this_episode = render_cond

    def on_episode_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        episode: EpisodeType,
        env_runner: Optional[EnvRunner] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: gym.Env | VectorEnv | Any = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        """On each env.step(), we add the render image to our Episode instance.

        Note that this would work with MultiAgentEpisodes as well.
        """
        if not self._render_this_episode:
            return
        if self.env_runner_indices is not None and env_runner.worker_index not in self.env_runner_indices:  # type: ignore[attr-defined]
            return
        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, SyncVectorEnv):
            assert not TYPE_CHECKING or isinstance(env, SyncVectorEnv)
            image = env.envs[0].render()
        else:
            image = env.render()
        if image is None:
            logger.warning("Env.render() returned None. Skipping rendering.")
            return
        if isinstance(env.unwrapped, AsyncVectorEnv):
            # tuple of images
            image = image[0]

        # Original render images for CartPole are 400x600 (hxw). We'll downsize here to
        # a very small dimension (to save space and bandwidth).
        image = resize(image, 64, 96)  # pyright: ignore[reportArgumentType]
        # For WandB videos, we need to put channels first.
        image = np.transpose(image, axes=[2, 0, 1])
        # Add the compiled single-step image as temp. data to our Episode object.
        # Once the episode is done, we'll compile the video from all logged images
        # and log the video with the EnvRunner's `MetricsLogger.log_...()` APIs.
        # See below:
        # `on_episode_end()`: We compile the video and maybe store it).
        # `on_sample_end()` We log the best and worst video to the `MetricsLogger`.
        try:
            episode.add_temporary_timestep_data("render_images", image)  # pyright: ignore[reportCallIssue]
        except (ValueError, TypeError):  # deprecated in 2.47+
            if "render_images" not in episode.custom_data:
                episode.custom_data["render_images"] = []
            episode.custom_data["render_images"].append(image)

    def on_episode_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        episode: EpisodeType,
        env_runner: EnvRunner = None,  # pyright: ignore[reportArgumentType]
        metrics_logger: Optional[MetricsLogger] = None,
        env: gym.Env | DictInfoToList | VectorEnv | Any = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        """Computes episode's return and compiles a video, iff best/worst in this iter.

        Note that the actual logging to the EnvRunner's MetricsLogger only happens
        at the very env of sampling (when we know, which episode was the best and
        worst). See `on_sample_end` for the implemented logging logic.
        """
        if not self._render_this_episode:
            return
        if not self.episode_save_condition(episode, env_runner, env, metrics_logger, rl_module):
            return
        # Get the episode's return.
        episode_return = episode.get_return()

        # Better than the best or worse than worst Episode thus far?
        if episode_return > self.best_episode_and_return[1] or episode_return < self.worst_episode_and_return[1]:
            # Pull all images from the temp. data of the episode.
            try:
                images = episode.get_temporary_timestep_data("render_images")  # pyright: ignore[reportCallIssue]
            except (ValueError, TypeError):  # deprecated in 2.47+
                images = episode.custom_data.get("render_images", [])
            # `images` is now a list of 3D ndarrays

            # Create a video from the images by simply stacking them AND
            # adding an extra B=1 dimension. Note that Tune's WandB logger currently
            # knows how to log the different data types by the following rules:
            # array is shape=3D -> An image (c, h, w).
            # array is shape=4D -> A batch of images (B, c, h, w).
            # array is shape=5D -> A video (1, L, c, h, w), where L is the length of the
            # video.
            # -> Make our video ndarray a 5D one.
            video = np.expand_dims(np.stack(images, axis=0), axis=0)

            # `video` is from the best episode in this cycle (iteration).
            if episode_return > self.best_episode_and_return[1]:
                self.best_episode_and_return = (video, episode_return)
            # `video` is worst in this cycle (iteration).
            else:
                self.worst_episode_and_return = (video, episode_return)

    def on_sample_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: MetricsLogger = None,  # type: ignore
        samples: Union[SampleBatch, list[EpisodeType]],
        **kwargs,
    ) -> None:
        """Logs the best and worst video to this EnvRunner's MetricsLogger."""
        self._render_this_episode = None
        # Best video.
        if self.best_episode_and_return[0] is not None:
            metrics_logger.log_value(
                EPISODE_BEST_VIDEO,
                self.best_episode_and_return[0],
                # Do not reduce the videos (across the various parallel EnvRunners).
                # This would not make sense (mean over the pixels?). Instead, we want to
                # log all best videos of all EnvRunners per iteration.
                reduce=None,
                # B/c we do NOT reduce over the video data (mean/min/max), we need to
                # make sure the list of videos in our MetricsLogger does not grow
                # infinitely and gets cleared after each `reduce()` operation, meaning
                # every time, the EnvRunner is asked to send its logged metrics.
                clear_on_reduce=True,
            )
            self.best_episode_and_return = (None, float("-inf"))
        # Worst video.
        if self.worst_episode_and_return[0] is not None:
            metrics_logger.log_value(
                EPISODE_WORST_VIDEO,
                self.worst_episode_and_return[0],
                # Same logging options as above.
                reduce=None,
                clear_on_reduce=True,
            )
            self.worst_episode_and_return = (None, float("inf"))
