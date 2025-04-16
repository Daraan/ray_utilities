from __future__ import annotations
import argparse

from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate

from typing import TYPE_CHECKING

from .experiment_base import ExperimentSetupBase, NamespaceType, ParserType, _ConfigType_co, _AlgorithmType_co

if TYPE_CHECKING:
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray_utilities.config.typed_argument_parser import OptionalExtenionsArgs


class SetupWithDynamicBuffer:

    @classmethod
    def _get_callbacks_from_args(cls, args: OptionalExtenionsArgs | argparse.Namespace) -> list[type[RLlibCallback]]:
        # When used as a mixin, can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)  # pyright: ignore[reportAttributeAccessIssue]
        if callbacks is None:
            callbacks = []
        if args.dynamic_buffer:
            callbacks.append(DynamicBufferUpdate)
        return callbacks
