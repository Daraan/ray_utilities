import logging

import gymnasium as gym
import numpy as np
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env

__all__ = [
    "create_env",
    "env_creator_with_seed",
    "parse_env_name",
]

_logger = logging.getLogger(__name__)

env_short_names = {
    "lunar": "LunarLander-v2",
    "cart": "CartPole-v1",
    "cartpole": "CartPole-v1",
}


def parse_env_name(name: str) -> str:
    return env_short_names.get(name, name)


def create_env(name: str, **kwargs) -> gym.Env:
    if name in env_short_names:
        return gym.make(env_short_names[name], **kwargs)
    return gym.make(name, **kwargs)


def env_creator_with_seed(config: EnvContext):
    """"""
    seed: int = config.pop("seed")
    env_type: str = config.pop("env_type")

    # If using multiple workers, use different seeds for workers with a higher index
    if config.num_workers and config.worker_index:
        mixed_seed = np.random.SeedSequence(
            seed,
            spawn_key=(config.worker_index,),  # type: ignore[attr-defined]
        ).generate_state(1)[0]
    else:
        mixed_seed = seed

    _logger.info("Environment seed: %s", mixed_seed)
    env = gym.make(env_type, **config)

    geni = np.random.Generator(np.random.PCG64(mixed_seed))
    env.np_random = geni
    return env

register_env("seeded_env", env_creator_with_seed)
