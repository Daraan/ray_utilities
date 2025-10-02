from __future__ import annotations

from typing import TYPE_CHECKING

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


def create_tuner_callbacks(*, adv_loggers: bool) -> list["Callback"]:
    if adv_loggers:
        return [cb() for cb in DEFAULT_ADV_TUNER_CALLBACKS]
    return [cb() for cb in DEFAULT_TUNER_CALLBACKS_NO_RENDER]
