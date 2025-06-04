from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .experiment_base import ExperimentSetupBase
from .tuner_setup import TunerSetup
from .typed_argument_parser import DefaultArgumentParser

if TYPE_CHECKING:
    from typing import Any, Callable

    from ray.rllib.algorithms import AlgorithmConfig
    from ray.rllib.callbacks.callbacks import RLlibCallback

__all__ = ["DefaultArgumentParser", "ExperimentSetupBase", "TunerSetup", "add_callbacks_to_config"]
logger = logging.getLogger(__name__)


def add_callbacks_to_config(
    config: AlgorithmConfig,
    callbacks: Optional[
        type[RLlibCallback] | list[type[RLlibCallback]] | dict[str, Callable[..., Any] | list[Callable[..., Any]]]
    ] = None,
    **kwargs,
):
    """
    Add the callbacks to the config.

    Args:
        config: The config to add the callback to.
        callback: The callback to add to the config.
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
                    config.callbacks(**{event: [present_callbacks, *callback_list]})
                else:
                    config.callbacks(**{event: [*present_callbacks, *callback_list]})
            else:
                config.callbacks(**{event: callback})  # pyright: ignore[reportArgumentType]
        return
    if not isinstance(callbacks, (list, tuple)):
        callbacks = [callbacks]
    if isinstance(config.callbacks_class, list):
        config.callbacks_class.extend(callbacks)
        return
    if getattr(config.callbacks_class, "IS_CALLBACK_CONTAINER", False):
        # Deprecated Multi callback
        logger.warning("Using deprecated MultiCallbacks API, cannot add efficient callbacks to it. Use the new API.")
    config.callbacks(callbacks_class=[config.callbacks_class, *callbacks])
