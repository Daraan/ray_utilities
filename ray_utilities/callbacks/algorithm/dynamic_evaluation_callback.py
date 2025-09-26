# ruff: noqa: ARG002
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray_utilities.callbacks.algorithm.callback_mixins import BudgetMixin, StepCounterMixin
from ray_utilities.callbacks.algorithm.dynamic_hyperparameter import DynamicHyperparameterCallback, UpdateFunction
from ray_utilities.dynamic_config.dynamic_buffer_update import get_dynamic_evaluation_intervals
from ray_utilities.misc import AutoInt

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


logger = logging.getLogger(__name__)


class DynamicEvalInterval(StepCounterMixin, BudgetMixin, DynamicHyperparameterCallback):
    """
    Attributes:
        updater
    """

    def _update_eval_interval(
        self,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        *,
        global_step: int,
        **kwargs: Any,
    ) -> None:
        assert algorithm.config
        env_steps = algorithm.config.train_batch_size_per_learner
        new_eval_interval = self._evaluation_intervals.get(env_steps, None)
        if new_eval_interval is None:
            logger.error(
                "No evaluation interval for current step %s in %s. "
                "Expected a value in the dictionary for global step %s. Not changing it, stays: %d",
                env_steps,
                self._evaluation_intervals,
                global_step,
                algorithm.config.evaluation_interval,
            )
            # do not change it
            return
        if algorithm.config.evaluation_interval == new_eval_interval:
            return
        # Change evaluation interval
        assert metrics_logger
        logger.info(
            "Evaluation interval changed from %s to %s at iteration %s - step %s",
            algorithm.config.evaluation_interval,
            new_eval_interval,
            metrics_logger.peek("training_iteration", default=self._training_iterations),
            global_step,
        )
        # Likely do not need to update learners and env runners here.
        self._update_algorithm(
            algorithm,
            key="evaluation_interval",
            value=AutoInt(new_eval_interval),
            update_learner=False,
            update_env_runners=False,
        )

    def __init__(
        self, update_function: UpdateFunction | None = None, learner_config_dict: dict[Any, Any] | None = None
    ):
        """

        Args:
            update_func: Function to update the buffer and batch size.
            learner_config_dict: Configuration dictionary for the learner. At best this is the same object as
                `algorithm.config.learner_config_dict` to ensure that the values are updated correctly.
        """
        self._set_budget_on__init__(learner_config_dict=learner_config_dict)
        super().__init__(update_function or self._update_eval_interval, "TBA - DynamicBufferUpdate")

    def _set_evaluation_intervals(self, algorithm: Algorithm) -> None:
        """Sets: self._evaluation_intervals"""
        # Add intervals that are also used during tuning
        step_sizes = list({*self._budget["step_sizes"], *(3072, 4096, 2048 * 3)})
        self._evaluation_intervals: dict[int, int] = dict(
            zip(
                step_sizes,
                (
                    get_dynamic_evaluation_intervals(
                        step_sizes,
                        eval_freq=self._original_interval,
                        batch_size=algorithm.config.train_batch_size_per_learner,  # pyright: ignore[reportOptionalMemberAccess]
                        take_root=True,
                    )
                    # 0 for no evaluation
                    if self._original_interval  # pyright: ignore[reportOptionalMemberAccess]
                    else [0] * len(step_sizes)
                ),
            )
        )
        for step_size, iterations in zip(self._budget["step_sizes"], self._budget["iterations_per_step_size"]):
            if iterations <= 2 and self._evaluation_intervals[step_size] > 1:
                # when doing not many iterations between step changes, assure that the evaluation interval is at least 1
                logger.debug(
                    "Setting evaluation interval for %s steps to 1, because iterations are %s",
                    step_size,
                    iterations,
                )
                self._evaluation_intervals[step_size] = 1
        logger.info("Dynamic evaluation intervals: %s", self._evaluation_intervals)

    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        self._set_budget_on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
        assert self._budget
        assert algorithm.config
        # TODO: When loading checkpoints and original differs from loaded, this is a problem
        self._original_interval = algorithm.config.evaluation_interval
        self._set_evaluation_intervals(algorithm=algorithm)
        self._set_step_counter_on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger)
        super().on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger)

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        assert metrics_logger
        self._set_step_counter_on_train_result(algorithm=algorithm, metrics_logger=metrics_logger)
        # self._planned_current_step likely safer way to get correct step, instead of using metrics_logger
        self._updater(algorithm, metrics_logger, global_step=self._planned_current_step)

    # Note this is not executed in the get_set_state tests
    def on_checkpoint_loaded(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        assert metrics_logger
        self._set_step_counter_on_checkpoint_loaded(algorithm=algorithm, metrics_logger=metrics_logger)
        self._set_budget_on_checkpoint_loaded(algorithm=algorithm, **kwargs)
        # TODO: problem when loading config.evaluation_interval is already based on the loaded config
        self._set_evaluation_intervals(algorithm=algorithm)
        super().on_checkpoint_loaded(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)  # calls updater


def add_dynamic_eval_callback_if_missing(callbacks: list[type[RLlibCallback]]):
    """Adds a DynamicEvalInterval callback if it's not already present"""
    if all(not issubclass(cb, DynamicEvalInterval) for cb in callbacks):
        # if any(issubclass(cb, AutoEvalIntervalOnInit) for cb in callbacks):
        #    logger.info("Removing a AutoEvalIntervalOnInit callback in favor of DynamicEvalInterval")
        # callbacks = [cb for cb in callbacks if not issubclass(cb, AutoEvalIntervalOnInit)]
        callbacks.append(DynamicEvalInterval)
