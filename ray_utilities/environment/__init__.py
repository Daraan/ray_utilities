from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from ray.tune.registry import register_env
from typing_extensions import deprecated

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.rllib.env.env_context import EnvContext

__all__ = [
    "create_env",
    "env_creator_with_seed",
    "parse_env_name",
]


env_short_names = {
    "lunar": "LunarLander-v2",
    "cart": "CartPole-v1",
    "cartpole": "CartPole-v1",
}

_logger = logging.getLogger(__name__)


def parse_env_name(name: str) -> str:
    return env_short_names.get(name, name)


def create_env(name: str, **kwargs) -> gym.Env:
    if name in env_short_names:
        return gym.make(env_short_names[name], **kwargs)
    return gym.make(name, **kwargs)


_seed_counter = 0


@deprecated("in favor of callback")
def env_creator_with_seed(config: EnvContext):
    """
    Creates an environment with seed

    Deprecated in favor of callback.
    """
    # NOTE: DO NOT MODIFY CONFIG; reused for VectorEnv
    this_env_config = config.copy()
    seed: int = this_env_config.pop("seed")
    env_type: str = this_env_config.pop("env_type")

    # If using multiple workers, use different seeds for workers with a higher index
    global _seed_counter  # noqa: PLW0603
    if config.num_workers and config.worker_index:
        mixed_seed = np.random.SeedSequence(
            seed,
            spawn_key=(config.worker_index, _seed_counter),  # type: ignore[attr-defined]
        ).generate_state(1)[0]
    else:
        mixed_seed = np.random.SeedSequence(
            seed,
            spawn_key=(0, _seed_counter),
        ).generate_state(1)[0]
    _seed_counter += 1
    _logger.info("Environment seed: %s", mixed_seed)
    geni = np.random.Generator(np.random.PCG64(mixed_seed))

    env = gym.make(env_type, **this_env_config)

    # TODO: Create vector env here

    env.np_random = geni
    _logger.debug(
        "Creating env with seed %s from env_seed=%s for worker idx %s/%s; count=%s.",
        # "Sample obs %s",
        mixed_seed,
        seed,
        config.worker_index,
        config.num_workers,
        _seed_counter,
        # env.reset(),
    )
    return env


register_env("seeded_env", env_creator_with_seed)


def create_env_for_config(config: AlgorithmConfig, env_spec: str | gym.Env):
    """
    Creates an initial environment for the given config.env.

    If it is a `seeded_env` it will create a config from `env_spec` instead.
    """
    if isinstance(config.env, str) and config.env != "seeded_env":
        init_env = gym.make(config.env)
    elif config.env == "seeded_env":
        if isinstance(env_spec, str):
            init_env = gym.make(env_spec)
        else:
            init_env = env_spec
    else:
        assert not TYPE_CHECKING or config.env
        init_env = gym.make(config.env.unwrapped.spec.id)  # pyright: ignore[reportOptionalMemberAccess]
    return init_env
