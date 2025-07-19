from __future__ import annotations

import logging
from inspect import isclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

try:
    from ray.tune.callback import _CallbackMeta
except ImportError:
    from abc import ABCMeta as _CallbackMeta  # in case meta is removed in future versions


if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.env.env_context import EnvContext
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
    from typing_extensions import TypeIs


def _is_vector_env(env) -> TypeIs[gym.vector.VectorEnv]:
    """Check if the environment is a vectorized environment."""
    return hasattr(env, "num_envs")


def _is_async(env) -> TypeIs[gym.vector.AsyncVectorEnv]:
    """Check if the environment is a vectorized environment."""
    return hasattr(env, "set_attr")


logger = logging.getLogger(__name__)


class _SeededEnvCallbackMeta(_CallbackMeta):  # pyright: ignore[reportGeneralTypeIssues]  # base is union type
    env_seed: ClassVar[int | None] = 0

    def __eq__(cls, value):  # pyright: ignore[reportSelfClsParameterName]
        if not isclass(value):
            return False
        if SeedEnvsCallback in value.__bases__:
            return cls.env_seed == value.env_seed
        return False

    def __hash__(cls):  # pyright: ignore[reportSelfClsParameterName]
        # Jonas
        return hash(DefaultCallbacks) + hash(cls.env_seed) + hash(cls.__name__)


class SeedEnvsCallback(DefaultCallbacks):
    """
    Attributes:
        env_seed: A common seed that is used for all workers and vector indices.
            If None, the environment will not be seeded. Rendering this callback useless.

    Use make_seeded_env_callback(None) for pure randomness.
    Use make_seeded_env_callback(fixed_seed) to create reproducible runs.
    make_seeded_env_callback(0) is equivalent to using this class directly.
    """

    env_seed: ClassVar[int | None] = 0
    """If None env will not be seeded"""

    def on_environment_created(
        self,
        *,
        env_runner: EnvRunner,  # noqa: ARG002
        metrics_logger: Optional[MetricsLogger] = None,
        env: gym.vector.AsyncVectorEnv | gym.vector.VectorEnv | gym.Env,
        env_context: EnvContext,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Callback run when a new environment object has been created.

        Note: This only applies to the new API stack. The env used is usually a
        gym.Env (or more specifically a gym.vector.Env).

        Args:
            env_runner: Reference to the current EnvRunner instance.
            metrics_logger: The MetricsLogger object inside the `env_runner`. Can be
                used to log custom metrics after environment creation.
            env: The environment object that has been created on `env_runner`. This is
                usually a gym.Env (or a gym.vector.Env) object.
            env_context: The `EnvContext` object that has been passed to the
                `gym.make()` call as kwargs (and to the gym.Env as `config`). It should
                have all the config key/value pairs in it as well as the
                EnvContext-typical properties: `worker_index`, `num_workers`, and
                `remote`.
            kwargs: Forward compatibility placeholder.
        """
        env_seed = self.env_seed
        if env_seed is None:
            return
        seeds = np.random.SeedSequence(
            env_seed,
            spawn_key=(env_context.worker_index, env_context.vector_index),  # type: ignore[attr-defined]
        ).generate_state(env.num_envs if _is_async(env) else 1)
        logger.debug(
            "Seeding envs with %s from env_seed=%s and worker_index %s/%s; vector=%s",
            seeds,
            env_seed,
            env_context.worker_index,
            env_context.num_workers,
            env_context.vector_index,
        )
        rngs = [np.random.Generator(np.random.PCG64(seed)) for seed in seeds]
        if _is_async(env=env):
            env.set_attr("np_random", rngs)
        else:
            env.np_random = rngs[0]

        # NOTE: Could log seeds in metrics_logger
        if metrics_logger:
            metrics_logger.log_value(
                ("environments", "seeds"), tuple(seeds.tolist()), clear_on_reduce=True, reduce=None
            )

    def __call__(self, **kwargs):
        """This is a no-op. The class is used as a callback."""
        return self.on_environment_created(**kwargs)

    def __init__(self, **kwargs):  # treat like a callback function
        if "env_context" in kwargs:
            self.on_environment_created(**kwargs)

    def __eq__(self, other):
        """Equality check for the callback."""
        return isinstance(other, SeedEnvsCallback) and self.env_seed == other.env_seed


def make_seeded_env_callback(env_seed_: int | None) -> type[SeedEnvsCallback]:
    """Create a callback that seeds the environment."""
    if env_seed_ is None:
        logger.info(
            "Using None as env_seed, this will create non-reproducible runs. The callback is deactivated.", stacklevel=2
        )

    class FixedSeedEnvsCallback(SeedEnvsCallback, metaclass=_SeededEnvCallbackMeta):
        env_seed = env_seed_

    return FixedSeedEnvsCallback
