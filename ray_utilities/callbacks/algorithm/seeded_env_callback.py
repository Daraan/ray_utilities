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

NUM_ENV_RUNNERS_0_1_EQUAL = True
FIX_EVAL_SEED = True
"""If True, this is closer to original EnvRunner behavior, but each evaluation will use the same seeds."""


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

    __logged_env_seed_none = False

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

        Note:
            This callback sets the EnvRunner's seed to `None`. This changes how env.reset works.
            In vanilla RLlib the reset is *always* to the same key of the EnvRunner, i.e. during evaluations
            the same initial state is used. This is not the case for this callback.
        """
        env_seed = self.env_seed
        if env_seed is None:
            if not self.__logged_env_seed_none:
                logger.debug("Environment not seeded, env_seed is None. Callback is deactivated.")
                self.__logged_env_seed_none = True
            return
        if env_context.recreated_worker:
            # Worker restart, potentially add flag to seed
            logger.warning("Recreated worker detected. Will be seeded with initial seed, potentially change seed.")
        # Trick to make num_env_runners=0 and num_env_runners=1 equal:
        if NUM_ENV_RUNNERS_0_1_EQUAL and (
            env_context.worker_index == 0 and env_context.num_workers == 0 and env_context.vector_index == 0
        ):
            worker_index = 1
            suffix = " (changed worker_index from 0/0 to 1 to be equal to num_env_runners=1)"
        else:
            worker_index = env_context.worker_index
            suffix = ""
        seed_sequence = np.random.SeedSequence(
            env_seed,
            spawn_key=(worker_index, env_context.vector_index, env_runner.config.in_evaluation),
        )
        seeds = seed_sequence.generate_state(env.num_envs if _is_async(env) else 1)
        rng = np.random.default_rng(seed_sequence)
        rngs = rng.spawn(env.num_envs if _is_async(env) else 1)
        logger.debug(
            "Seeding envs with seed=%s - "
            "created from env_seed=%s, worker_index %s/%s, evaluation=%s, vector_index=%s.%s",
            seed_sequence,
            env_seed,
            worker_index,
            env_context.num_workers,  # not used for seed
            env_runner.config.in_evaluation,
            env_context.vector_index,
            suffix,
        )
        # rngs = [np.random.Generator(np.random.PCG64(seed)) for seed in seeds]
        # Set random generators for the environments
        if _is_async(env=env):
            env.set_attr("np_random", rngs)
        else:
            env.np_random = rngs[0]

        # NOTE: Could log seeds in metrics_logger
        if metrics_logger:
            # HACK: Set clear_on_reduce=True and remove window again when https://github.com/ray-project/ray/issues/54324 is solved  # noqa: E501
            metrics_logger.log_value(
                ("environments", "seeds", "seed_sequence"),
                seeds.tolist(),
                clear_on_reduce=False,
                reduce=None,
                # HACK 2: clear_on_reduce=True is forced when no window is provided
                window=len(seeds) + 1,  # remove when bug is fixed
            )
        # NOTE: Need to set env_runner._seed to None for the custom seeds to be used.
        if env_runner.config.in_evaluation and FIX_EVAL_SEED:
            env_runner._seed = rng.integers(0, 2**31 - 1, size=env.num_envs if _is_async(env) else 1).tolist()
        else:
            env_runner._seed = None
        logger.debug("Setting EnvRunner seed to None, to use seed of %s", type(self).__name__)

    def __call__(self, **kwargs):
        """Instance is used as a callback."""
        return self.on_environment_created(**kwargs)

    def __init__(self, **kwargs):  # treat like a callback function
        if "env_context" in kwargs:  # Instance called on_environment_created
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
