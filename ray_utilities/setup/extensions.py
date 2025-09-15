"""Extension mixins for dynamic configuration in Ray RLlib experiment setups.

This module provides mixin classes that extend the base experiment setup functionality
with dynamic configuration capabilities. These mixins can be composed with base setup
classes to add features like dynamic buffer sizing, batch size adjustment, and
adaptive evaluation intervals during training.

The mixins are designed to work together and with the base
:class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase` class through
multiple inheritance, providing modular functionality that can be mixed and matched
based on experiment requirements.

Key Components:
    - :class:`SetupWithDynamicBuffer`: Dynamic experience buffer sizing
    - :class:`SetupWithDynamicBatchSize`: Dynamic batch size and gradient accumulation
    - :class:`SetupForDynamicTuning`: Base class for dynamic configuration mixins

These extensions integrate with Ray Tune's parameter spaces and RLlib callbacks
to provide adaptive behavior during training and hyperparameter optimization.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar

from ray import tune

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.dynamic_evaluation_callback import add_dynamic_eval_callback_if_missing
from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co, ExperimentSetupBase, ParserType_co

if TYPE_CHECKING:
    from ray.rllib.callbacks.callbacks import RLlibCallback

    from ray_utilities.typing import ParameterSpace

_logger = logging.getLogger(__name__)


class SetupForDynamicTuning(ExperimentSetupBase[ParserType_co, ConfigType_co, AlgorithmType_co]):
    ...
    # base class that can be expanded in the future


class SetupWithDynamicBuffer(SetupForDynamicTuning[ParserType_co, ConfigType_co, AlgorithmType_co]):
    """Mixin class for dynamic experience buffer sizing in RLlib experiments.

    This mixin adds the capability to dynamically adjust the experience buffer
    (rollout) size during training or hyperparameter optimization. It provides
    both predefined parameter spaces for tuning and automatic callback integration
    for adaptive buffer management.

    The mixin automatically adds the :class:`~ray_utilities.callbacks.algorithm.dynamic_buffer_callback.DynamicBufferUpdate`
    callback when dynamic buffer sizing is enabled, and includes evaluation
    interval adjustment to work properly with the dynamic buffer updates.

    Features:
        - Predefined rollout size parameter space for Ray Tune optimization
        - Automatic callback registration for dynamic buffer updates
        - Integration with dynamic evaluation intervals
        - Supports both grid search and adaptive parameter selection

    Class Attributes:
        rollout_size_sample_space: Ray Tune parameter space with common rollout sizes
            ranging from 32 to 8192 steps, suitable for grid search optimization.

    Note:
        This mixin should be used before other setups that add
        :class:`~ray_utilities.callbacks.algorithm.dynamic_evaluation_callback.DynamicEvalInterval`
        to avoid error logs about missing configuration keys.

    Example:
        class MySetup(SetupWithDynamicBuffer, ExperimentSetupBase):
            config_class = PPOConfig
            algo_class = PPO

    See Also:
        :class:`~ray_utilities.callbacks.algorithm.dynamic_buffer_callback.DynamicBufferUpdate`: Buffer update callback
        :class:`~ray_utilities.callbacks.algorithm.dynamic_evaluation_callback.DynamicEvalInterval`: Evaluation callback
        :class:`SetupWithDynamicBatchSize`: Companion mixin for batch size dynamics
    """

    rollout_size_sample_space: ClassVar[ParameterSpace[int]] = tune.grid_search(
        [32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 2048 * 3, 8192]  # 4096 * 3, 16384]
    )

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        # Can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)
        if args.dynamic_buffer:
            callbacks.append(DynamicBufferUpdate)
            add_dynamic_eval_callback_if_missing(callbacks)
        return callbacks

    @classmethod
    def _create_dynamic_buffer_params(cls):
        return deepcopy(cls.rollout_size_sample_space)

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
    """Mixin class for dynamic batch size adjustment through gradient
    accumulation.

    This mixin enables dynamic batch size control during training by
    utilizing gradient accumulation rather than directly modifying the
    ``train_batch_size_per_learner``. It integrates the
    :class:`~ray_utilities.callbacks.algorithm.dynamic_batch_size.DynamicGradientAccumulation`
    callback when dynamic batch sizing is enabled so the effective batch
    size can be adjusted during training or tuning.

    Features:
        - Dynamic batch size control via gradient accumulation
        - Predefined batch size parameter space for Ray Tune optimization
        - Automatic callback registration for gradient accumulation
        - Integration with dynamic evaluation intervals

    Note:
        Use :class:`SetupWithDynamicBuffer` for direct tuning of rollout
        sizes. This mixin controls effective batch size via gradient
        accumulation.

    Warning:
        The ``batch_size`` tuning values refer to effective batch sizes
        achieved via gradient accumulation, not direct accumulation
        multipliers.

    Examples:
        .. code-block:: python

            class MySetup(SetupWithDynamicBatchSize, ExperimentSetupBase):
                config_class = PPOConfig
                algo_class = PPO

                def create_config(self, args):
                    config = super().create_config(args)
                    if args.dynamic_batch:
                        # Gradient accumulation will be handled automatically
                        pass
                    return config

    See Also:
        :class:`~ray_utilities.callbacks.algorithm.dynamic_batch_size.DynamicGradientAccumulation`:
            Gradient accumulation callback
        :class:`SetupWithDynamicBuffer`:
            Companion mixin for buffer size dynamics
        :class:`~ray_utilities.callbacks.algorithm.dynamic_evaluation_callback.DynamicEvalInterval`:
            Evaluation callback
    """

    batch_size_sample_space: ClassVar[ParameterSpace[int]] = tune.grid_search(
        [32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 2048 * 3, 8192]  # 4096 * 3, 16384]
    )
    """
    Tune parameter space with batch sizes from 32 to 16384.
    """

    @classmethod
    def _get_callbacks_from_args(cls, args) -> list[type[RLlibCallback]]:
        # When used as a mixin, can call the parent; might be None for ExperimentSetupBase
        callbacks = super()._get_callbacks_from_args(args)
        if args.dynamic_batch:
            callbacks.append(DynamicGradientAccumulation)
            add_dynamic_eval_callback_if_missing(callbacks)
        return callbacks

    @classmethod
    def _create_dynamic_batch_size_params(cls):
        """Create parameter space for dynamic batch size tuning.

        Returns a deep copy of the class's batch size sample space for use in
        hyperparameter optimization. The returned parameter space contains
        batch size values that will be achieved through gradient accumulation.

        Returns:
            A Ray Tune parameter space containing batch size values for optimization.

        Note:
            The returned parameters represent effective batch sizes achieved through
            gradient accumulation, not the gradient accumulation multiplier values directly.

        Warning:
            This method adds ``batch_size`` parameters to the tuning space, not
            values for direct gradient accumulation control.
        """
        # TODO: control this somehow via args
        return deepcopy(cls.batch_size_sample_space)

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
