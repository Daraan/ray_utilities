from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ray.tune import Callback

from .adv_comet_callback import AdvCometLoggerCallback
from .adv_csv_callback import AdvCSVLoggerCallback
from .adv_json_logger_callback import AdvJsonLoggerCallback
from .adv_tbx_logger_callback import AdvTBXLoggerCallback
from .adv_wandb_callback import AdvWandbLoggerCallback
from .sync_config_files_callback import SyncConfigFilesCallback

if TYPE_CHECKING:
    from ray.tune.callback import Callback

__all__ = [
    "AdvCSVLoggerCallback",
    "AdvCometLoggerCallback",
    "AdvJsonLoggerCallback",
    "AdvTBXLoggerCallback",
    "AdvWandbLoggerCallback",
    "SyncConfigFilesCallback",
]


DEFAULT_TUNER_CALLBACKS_NO_RENDER: list[type["Callback"]] = []
"""
Default callbacks to use when neither needing render_mode nor advanced loggers.

Note:
    AdvCometLoggerCallback is not included
"""

DEFAULT_TUNER_CALLBACKS_RENDER: list[type["Callback"]] = [
    AdvJsonLoggerCallback,
    AdvTBXLoggerCallback,
    AdvCSVLoggerCallback,
]
"""Default callbacks to use when needing render_mode"""

DEFAULT_ADV_TUNER_CALLBACKS = DEFAULT_TUNER_CALLBACKS_RENDER.copy()
"""
List of advanced tuner callbacks to use if the advanced variants should be used.
Recommended when using schedulers working with :const:`FORK_FROM`.

A copy of :obj:`DEFAULT_TUNER_CALLBACKS_RENDER`.
"""


def create_tuner_callbacks(
    *, adv_loggers: bool, offline_loggers: Optional[bool | list[str | Any]] = None, json=True, tbx=True, csv=True
) -> list["Callback"]:
    callbacks = []
    if offline_loggers is not None:
        if isinstance(offline_loggers, bool):
            json = json and offline_loggers
            tbx = tbx and offline_loggers
            csv = csv and offline_loggers
        elif isinstance(offline_loggers, list):
            # "all" should be converted to True by parser
            json = json and ("json" in offline_loggers or "all" in offline_loggers)
            tbx = tbx and (
                "tensorboard" in offline_loggers
                or "tb" in offline_loggers
                or "tbx" in offline_loggers
                or "all" in offline_loggers
            )
            csv = csv and ("csv" in offline_loggers or "all" in offline_loggers)
    if adv_loggers:
        for cb in DEFAULT_ADV_TUNER_CALLBACKS:
            if (
                (cb is AdvJsonLoggerCallback and not json)
                or (cb is AdvTBXLoggerCallback and not tbx)
                or (cb is AdvCSVLoggerCallback and not csv)
            ):
                continue
            callbacks.append(cb())
        return callbacks
    for cb in DEFAULT_TUNER_CALLBACKS_NO_RENDER:
        if (
            (cb is AdvJsonLoggerCallback and not json)
            or (cb is AdvTBXLoggerCallback and not tbx)
            or (cb is AdvCSVLoggerCallback and not csv)
        ):
            continue
        callbacks.append(cb())
    return callbacks
