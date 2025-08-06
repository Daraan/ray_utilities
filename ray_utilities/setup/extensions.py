from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ray import tune

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import DynamicEvalInterval
from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co, ExperimentSetupBase, ParserType_co

if TYPE_CHECKING:
    from ray.rllib.callbacks.callbacks import RLlibCallback

_logger = logging.getLogger(__name__)


class SetupForDynamicTuning(ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]):
    ...
    # base class that can be expanded in the future


class SetupWithDynamicBuffer(SetupForDynamicTuning[ParserType_co, ConfigType_co, AlgorithmType_co]):
    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        # Can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)
        if args.dynamic_buffer:
            callbacks.append(DynamicBufferUpdate)
            if all(not issubclass(cb, DynamicEvalInterval) for cb in callbacks):
                callbacks.append(DynamicEvalInterval)
        return callbacks

    @staticmethod
    def _create_dynamic_buffer_params():
        return tune.grid_search([16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 16384])

    def create_param_space(self) -> dict[str, Any]:
        self._set_dynamic_parameters_to_tune()  # sets _dynamic_parameters_to_tune
        if not self.args.tune or not (
            (add_all := "all" in self._dynamic_parameters_to_tune) or "rollout_size" in self._dynamic_parameters_to_tune
        ):
            return super().create_param_space()
        if not add_all:
            self._dynamic_parameters_to_tune.remove(
                "rollout_size"
            )  # remove before calling super().create_param_space()
        param_space = super().create_param_space()
        # TODO: # FIXME "rollout_size" is not used anywhere
        # however train_batch_size_per_learner is used with the DynamicBatchSize Setup
        # which uses in ints dynamic variant gradient accumulation.
        param_space["rollout_size"] = self._create_dynamic_buffer_params()
        return param_space


class SetupWithDynamicBatchSize(SetupForDynamicTuning[ParserType_co, ConfigType_co, AlgorithmType_co]):
    """
    Note:
        When using the dynamic_batch callback the setup uses gradient accumulation to control the batch size.
        It does not increase the train_batch_size_per_learner. For that select the dynamic rollout size setup.
    """

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        # When used as a mixin, can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)
        if args.dynamic_batch:
            callbacks.append(DynamicGradientAccumulation)
            if all(not issubclass(cb, DynamicEvalInterval) for cb in callbacks):
                callbacks.append(DynamicEvalInterval)
        return callbacks

    @staticmethod
    def _create_dynamic_batch_size_params():
        """
        Attentions:
            Adds batch_size and not values for gradient accumulation!
        """
        # TODO: control this somehow via args
        return tune.grid_search([16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 16384])

    def create_param_space(self) -> dict[str, Any]:
        self._set_dynamic_parameters_to_tune()  # sets _dynamic_parameters_to_tune
        if not self.args.tune or not (
            (add_all := "all" in self._dynamic_parameters_to_tune) or "batch_size" in self._dynamic_parameters_to_tune
        ):
            return super().create_param_space()
        if not add_all:
            self._dynamic_parameters_to_tune.remove("batch_size")  # remove before calling super().create_param_space()
        param_space = super().create_param_space()
        param_space["train_batch_size_per_learner"] = self._create_dynamic_batch_size_params()
        return param_space
