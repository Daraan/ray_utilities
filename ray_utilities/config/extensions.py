from __future__ import annotations

from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray_utilities.config.typed_argument_parser import OptionalExtensionsArgs


class SetupWithDynamicBuffer:
    @classmethod
    def _get_callbacks_from_args(cls, args: OptionalExtensionsArgs | argparse.Namespace) -> list[type[RLlibCallback]]:
        # When used as a mixin, can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)  # pyright: ignore[reportAttributeAccessIssue]
        if callbacks is None:
            callbacks = []
        if args.dynamic_buffer:
            callbacks.append(DynamicBufferUpdate)
        return callbacks
