from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing_extensions import deprecated

import gymnasium as gym
import numpy as np
from ray.tune.registry import register_env

from ray_utilities.callbacks.algorithm.seeded_env_callback import SeedEnvsCallback, make_seeded_env_callback

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


def seed_environments_for_config(config: AlgorithmConfig, env_seed: int | None):
    """
    Adds/replaces a common deterministic seeding that is used to seed all environments created when this config is build.

    Choose One:
    - Same environment seeding across trials:
        seed_environments_for_config(config, constant_seed)
    - Different, but deterministic, seeding across trials:
        seed_environments_for_config(config, run_seed)
    """
    seed_envs_cb = make_seeded_env_callback(env_seed)
    # NOTE: Needs NEW API
    if config.callbacks_on_environment_created:
        if callable(config.callbacks_on_environment_created):
            config.callbacks(on_environment_created=[config.callbacks_on_environment_created, seed_envs_cb])
        else:  # assume another iterable
            try:
                l_before = len(config.callbacks_on_environment_created)
            except (TypeError, Exception):  # Some iterable without len?
                l_before = None
            config.callbacks(
                on_environment_created=[
                    *(
                        cb
                        for cb in config.callbacks_on_environment_created
                        if not isinstance(cb, SeedEnvsCallback)
                        or (isinstance(cb, type) and issubclass(cb, SeedEnvsCallback))
                    ),
                    seed_envs_cb,
                ]
            )
            if l_before == len(config.callbacks_on_environment_created):
                _logger.info("A SeedEnvsCallback was replaced by calling seed_environments_for_config.")
    else:
        config.callbacks(on_environment_created=seed_envs_cb)
