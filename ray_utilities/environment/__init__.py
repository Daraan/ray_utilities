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


_seed_counter = 0


def env_creator_with_seed(config: EnvContext):
    """"""
    # NOTE: DO NOT MODIFY CONFIG; reused for VectorEnv
    this_env_config = config.copy()
    seed: int = this_env_config.pop("seed")
    env_type: str = this_env_config.pop("env_type")

    # If using multiple workers, use different seeds for workers with a higher index
    global _seed_counter
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
