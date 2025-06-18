from __future__ import annotations

from typing import TYPE_CHECKING

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import DynamicEvalInterval

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
            if all(not issubclass(cb, DynamicEvalInterval) for cb in callbacks):
                callbacks.append(DynamicEvalInterval)
        return callbacks


class SetupWithDynamicBatchSize:
    @classmethod
    def _get_callbacks_from_args(cls, args: OptionalExtensionsArgs | argparse.Namespace) -> list[type[RLlibCallback]]:
        # When used as a mixin, can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)  # pyright: ignore[reportAttributeAccessIssue]
        if callbacks is None:
            callbacks = []
        if args.dynamic_batch:
            callbacks.append(DynamicGradientAccumulation)
            if all(not issubclass(cb, DynamicEvalInterval) for cb in callbacks):
                callbacks.append(DynamicEvalInterval)
        return callbacks
