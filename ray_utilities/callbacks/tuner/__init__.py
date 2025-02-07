from __future__ import annotations

from typing import TYPE_CHECKING

from ray.tune import Callback

from .adv_comet_callback import AdvCometLoggerCallback
from .adv_csv_callback import AdvCSVLoggerCallback
from .adv_json_logger_callback import AdvJsonLoggerCallback
from .adv_tbx_logger_callback import AdvTBXLoggerCallback
from .adv_wandb_callback import AdvWandbLoggerCallback

if TYPE_CHECKING:
    from ray.tune.callback import Callback

__all__ = [
    "AdvCSVLoggerCallback",
    "AdvCometLoggerCallback",
    "AdvJsonLoggerCallback",
    "AdvTBXLoggerCallback",
    "AdvWandbLoggerCallback",
]


DEFAULT_TUNER_CALLBACKS_NO_RENDER: list[type["Callback"]] = []
"""
Default callbacks to use when not needing render_mode

Note:
    AdvCometLoggerCallback is not included
"""

DEFAULT_TUNER_CALLBACKS_RENDER: list[type["Callback"]] = [
    AdvJsonLoggerCallback,
    AdvTBXLoggerCallback,
    AdvCSVLoggerCallback,
]
"""Default callbacks to use when needing render_mode"""


def create_tuner_callbacks(*, render: bool) -> list["Callback"]:
    if render:
        return [cb() for cb in DEFAULT_TUNER_CALLBACKS_RENDER]
    return [cb() for cb in DEFAULT_TUNER_CALLBACKS_NO_RENDER]
