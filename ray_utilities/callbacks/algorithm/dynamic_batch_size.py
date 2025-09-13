# ruff: noqa: ARG002
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray_utilities.callbacks.algorithm.callback_mixins import BudgetMixin, StepCounterMixin
from ray_utilities.callbacks.algorithm.dynamic_hyperparameter import DynamicHyperparameterCallback, UpdateFunction
from ray_utilities.dynamic_config.dynamic_buffer_update import update_buffer_and_rollout_size

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


logger = logging.getLogger(__name__)


class DynamicGradientAccumulation(StepCounterMixin, BudgetMixin, DynamicHyperparameterCallback):
    """
    Attributes:
        updater
    """

    # TODO:  As body is rather simple can just pass function as updater
    def _update_gradient_accumulation(
        self,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        *,
        global_step: int,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        assert algorithm.config
        assert metrics_logger
        # Need to modify algorithm.config.train_batch_size_per_learner
        learner_config_dict = algorithm.config.learner_config_dict
        assert self._budget
        batch_size, accumulate_gradients_every, _env_steps = update_buffer_and_rollout_size(
            total_steps=learner_config_dict["total_steps"],  # test if budget total_steps is fine as well
            dynamic_buffer=learner_config_dict["dynamic_buffer"],
            dynamic_batch=learner_config_dict["dynamic_batch"],
            global_step=global_step,
            accumulate_gradients_every_initial=self._accumulate_gradients_every_initial,
            initial_steps=self._budget["step_sizes"][0],
            num_increase_factors=len(self._budget["step_sizes"]),
            n_envs=1,
        )
        # Batch Size
        if accumulate_gradients_every != self._accumulate_gradients_every_current:
            logger.debug(
                "Accumulating gradients update after n iterations changed from %s to %s at iteration %s - step %s. "
                "effective batch size is %s",
                self._accumulate_gradients_every_current,
                accumulate_gradients_every,
                metrics_logger.peek("training_iteration", default=self._training_iterations),
                global_step,
                # Might be wrong if batch size is updated at same step by a later callback
                algorithm.config.train_batch_size_per_learner * accumulate_gradients_every,
            )
            self._batch_size_current = batch_size
            self._accumulate_gradients_every_current = accumulate_gradients_every
            self._update_learner_config(algorithm, accumulate_gradients_every=accumulate_gradients_every)

    def __init__(
        self, update_function: UpdateFunction | None = None, learner_config_dict: dict[Any, Any] | None = None
    ):
        """

        Args:
            update_func: Function to update the buffer and batch size.
            learner_config_dict: Configuration dictionary for the learner. At best this is the same object as
                `algorithm.config.learner_config_dict` to ensure that the values are updated correctly.
        """
        super().__init__(update_function or self._update_gradient_accumulation, "TBA - DynamicBufferUpdate")
        self._set_budget_on__init__(learner_config_dict=learner_config_dict)
        # Set on algorithm init
        self._batch_size_current: int = None
        self._accumulate_gradients_every_current: int = None
        self._accumulate_gradients_every_initial: int = None

    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        # TODO: What to do when using checkpoints?
        assert algorithm.config
        assert algorithm.config.minibatch_size is not None
        self._set_step_counter_on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger)
        self._set_budget_on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
        logger.debug("Initial rollout size for DynamicBuffer %s", self._budget["step_sizes"][0])
        # legacy only
        self._initial_minibatch_size = algorithm.config.minibatch_size
        interval = algorithm.config.learner_config_dict.get("accumulate_gradients_every", None)
        if interval is None:
            logger.warning("No accumulate_gradients_every found in learner_config_dict. Using default value of 1. ")
            interval = 1
        self._accumulate_gradients_every_current = interval
        self._accumulate_gradients_every_initial = interval
        super().on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger)

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        result: dict,
        **kwargs,
    ) -> None:
        assert metrics_logger
        # Safer way to get correct steps:
        self._set_step_counter_on_train_result(algorithm=algorithm, metrics_logger=metrics_logger)
        self._updater(algorithm, metrics_logger, global_step=self._planned_current_step)  # pyright: ignore[reportArgumentType]
        # Exact way to update steps:
        # self._updater(algorithm, metrics_logger, global_step=self._get_global_step(metrics_logger))

    def on_checkpoint_loaded(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        self._set_budget_on_checkpoint_loaded(algorithm=algorithm, **kwargs)
        assert metrics_logger
        self._set_step_counter_on_checkpoint_loaded(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
        super().on_checkpoint_loaded(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)  # calls update
        # TODO: self._training_iterations = 0
