# ruff: noqa: ARG002
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray_utilities.callbacks.algorithm.callback_mixins import BudgetMixin, StepCounterMixin
from ray_utilities.callbacks.algorithm.dynamic_hyperparameter import DynamicHyperparameterCallback, UpdateFunction
from ray_utilities.dynamic_config.dynamic_buffer_update import get_dynamic_evaluation_intervals

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
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
                "No evaluation interval for %s steps in %s. Expected a value in the dictionary. Steps: %s",
                env_steps,
                self._evaluation_intervals,
                global_step,
            )
            # do not change it
            return
        if algorithm.config.evaluation_interval == new_eval_interval:
            return
        # Change evaluation interval
        assert metrics_logger
        logger.debug(
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
            value=new_eval_interval,
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
        self._evaluation_intervals: dict[int, int] = dict(
            zip(
                self._budget["step_sizes"],
                (
                    get_dynamic_evaluation_intervals(
                        self._budget["step_sizes"],
                        eval_freq=algorithm.config.evaluation_interval,  # pyright: ignore[reportOptionalMemberAccess]
                        batch_size=algorithm.config.train_batch_size_per_learner,  # pyright: ignore[reportOptionalMemberAccess]
                        take_root=True,
                    )
                    # 0 for no evaluation
                    if algorithm.config.evaluation_interval  # pyright: ignore[reportOptionalMemberAccess]
                    else [0] * len(self._budget["step_sizes"])
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
        # TODO: What when using checkpoint?
        self._set_budget_on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
        assert self._budget
        assert algorithm.config
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
        self._updater(algorithm, None, global_step="???" or self._planned_current_step)
        # TODO: self._training_iterations = 0
