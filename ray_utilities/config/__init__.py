"""Ray AlgorithmConfig and argument parsing utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray_utilities.callbacks.algorithm.seeded_env_callback import SeedEnvsCallback, make_seeded_env_callback

from .typed_argument_parser import DefaultArgumentParser

if TYPE_CHECKING:
    from typing import Any, Callable

    from ray.rllib.algorithms import AlgorithmConfig
    from ray.rllib.callbacks.callbacks import RLlibCallback

__all__ = [
    "DefaultArgumentParser",
    "add_callbacks_to_config",
    "seed_environments_for_config",
]
logger = logging.getLogger(__name__)


def add_callbacks_to_config(
    config: AlgorithmConfig,
    callbacks: Optional[
        type[RLlibCallback] | list[type[RLlibCallback]] | dict[str, Callable[..., Any] | list[Callable[..., Any]]]
    ] = None,
    *,
    remove_existing: Callable[[Any], bool] = lambda cb: False,  # noqa: ARG005
    **kwargs,
):
    """
    Add the callbacks to the config.

    Args:
        config: The config to add the callback to.
        callback: The callback to add to the config.
        remove_existing: Remove existing callbacks that do match the filter.
    """
    if callbacks is not None and kwargs:
        raise ValueError("Specify either 'callbacks' or keyword arguments, not both.")
    if callbacks is None:
        callbacks = kwargs
    if not callbacks:
        return
    if isinstance(callbacks, dict):
        for event, callback in callbacks.items():
            assert event != "callbacks_class", "Pass types and not a dictionary."
            if present_callbacks := getattr(config, "callbacks_" + event):
                # add  multiple or a single new one to existing one or multiple ones
                callback_list = [callback] if callable(callback) else callback
                if callable(present_callbacks):
                    if remove_existing(present_callbacks):
                        logger.debug(
                            "Replacing existing callback %s with new one %s for event %s",
                            present_callbacks,
                            callback_list,
                            event,
                        )
                        config.callbacks(**{event: callback})  # pyright: ignore[reportArgumentType]; cannot assign to callback_class
                    else:
                        config.callbacks(**{event: [present_callbacks, *callback_list]})
                else:
                    num_old = len(present_callbacks)
                    num_new = len(callback_list)
                    new_cb = [*(cb for cb in present_callbacks if not remove_existing(cb)), *callback_list]
                    if len(new_cb) < num_old + num_new:
                        removed = [cb for cb in present_callbacks if remove_existing(cb)]
                        logger.debug(
                            "Removing existing callbacks that match the filter for event %s:\n- %s\n+ %s",
                            event,
                            removed,
                            new_cb,
                        )
                    config.callbacks(**{event: new_cb})
            else:
                config.callbacks(**{event: callback})  # pyright: ignore[reportArgumentType]
        return
    if not isinstance(callbacks, (list, tuple)):
        callbacks = [callbacks]
    if isinstance(config.callbacks_class, list):
        # filter out and extend
        present_callbacks = config.callbacks_class
        num_old = len(present_callbacks)
        num_new = len(callbacks)
        new_cb = [*(cb for cb in present_callbacks if not remove_existing(cb)), *callbacks]
        if len(new_cb) < num_old + num_new:
            removed = [cb for cb in present_callbacks if remove_existing(cb)]
            logger.debug(
                "Replacing existing callback_class list with filtered new list:\n- %s\n+ %s",
                removed,
                new_cb,
            )
        config.callbacks(callbacks_class=new_cb)
        return
    if getattr(config.callbacks_class, "IS_CALLBACK_CONTAINER", False):
        # Deprecated Multi callback
        logger.warning("Using deprecated MultiCallbacks API, cannot add efficient callbacks to it. Use the new API.")
    if remove_existing(config.callbacks_class):
        logger.debug(
            "Replacing existing callback_class %s with new one %s",
            config.callbacks_class,
            callbacks,
        )
        config.callbacks(callbacks_class=callbacks)
        return
    config.callbacks(callbacks_class=[config.callbacks_class, *callbacks])


if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig


def _remove_existing_seeded_envs(cb: Any) -> bool:
    """Returns True if the passed callback is a SeedEnvsCallback or a subclass of it."""
    return isinstance(cb, SeedEnvsCallback) or (isinstance(cb, type) and issubclass(cb, SeedEnvsCallback))


def seed_environments_for_config(config: AlgorithmConfig, env_seed: int | None):
    """
    Adds/replaces a common deterministic seeding that is used to seed all environments created
    when config is build.

    Choose One:
    - Same environment seeding across trials, workers have different constant seeds:

        seed_environments_for_config(config, constant_seed)  # <-- constant across trials

    - Different, but deterministic, seeding across trials:

        seed_environments_for_config(config, env_seed)  # <-- sampled by tune

    - Random seeding across trials:

        seed_environments_for_config(config, None)  # <-- always random
    """
    seed_envs_cb = make_seeded_env_callback(env_seed)
    add_callbacks_to_config(config, on_environment_created=seed_envs_cb, remove_existing=_remove_existing_seeded_envs)
