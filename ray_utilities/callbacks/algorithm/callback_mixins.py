from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, TypeVar

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    LEARNER_RESULTS,
    ENV_RUNNER_RESULTS,
)  # pyright: ignore[reportPrivateImportUsage]

from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
from ray_utilities.dynamic_config.dynamic_buffer_update import SplitBudgetReturnDict, split_timestep_budget

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

_logger = logging.getLogger(__name__)

__all__ = ["BudgetMixin"]

T = TypeVar("T", None, Mapping[str, Any])


class BudgetMixin(Generic[T]):
    """
    Attributes:
        _budget: dict[str, int] | None
            A dictionary containing the budget for dynamic buffer size and rollout size.
            It is initialized from the learner_config_dict if provided, or set to None.
            The dictionary contains keys: 'min_dynamic_buffer_size', 'max_dynamic_buffer_size', and 'total_steps'.

    Methods:
        _set_budget_on__init__(learner_config_dict: dict[Any, Any] | None = None, **kwargs: Any) -> None
        _set_budget_on_algorithm_init(algorithm: Algorithm, **kwargs: Any) -> None
            Initializes the budget based on the learner_config_dict from the algorithm's config.
        _set_budget_on_checkpoint_loaded(algorithm: Algorithm, **kwargs: Any) -> None
    """

    def _set_budget_on__init__(self, learner_config_dict: dict[Any, Any] | T = None):
        self._budget: SplitBudgetReturnDict | T
        if learner_config_dict:
            # NOTE: Current pyright error could be a bug T not correctly narrowed.
            if "total_steps" not in learner_config_dict:
                _logger.warning(
                    "learner_config_dict must contain 'total_steps' key. Possibly the config is not set yet."
                )
            try:
                self._budget = split_timestep_budget(
                    total_steps=learner_config_dict["total_steps"],
                    min_size=learner_config_dict["min_dynamic_buffer_size"],
                    max_size=learner_config_dict["max_dynamic_buffer_size"],
                    assure_even=True,
                )
            except KeyError as e:
                _logger.warning(
                    "Missing key in learner_config_dict: %s during creation of %s. "
                    "Potentially this callback is created before setting the learner_config_dict. "
                    "If the key is not present during the algorithm initialization, "
                    "this will raise an error later.",
                    e,
                    self.__class__.__name__,
                )
        else:
            self._budget = None

    def _set_budget_on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,  # noqa: ARG002
    ) -> None:
        assert algorithm.config
        learner_config_dict = algorithm.config.learner_config_dict
        try:
            budget = split_timestep_budget(
                total_steps=learner_config_dict["total_steps"],
                min_size=learner_config_dict["min_dynamic_buffer_size"],
                max_size=learner_config_dict["max_dynamic_buffer_size"],
                assure_even=True,
            )
        except KeyError as e:
            _logger.error("Missing key in learner_config_dict: %s", e)
            raise
        if self._budget is not None:
            assert budget == self._budget, "Budget dict changed since initialization."
        self._budget = budget

    def _set_budget_on_checkpoint_loaded(self, *, algorithm: Algorithm, **kwargs) -> None:
        if self._budget is None:
            _logger.error("BudgetMixin._budget is None. Need to recreate.")
        # FIXME


class GetGlobalStepMixin:
    """
    Methods:
        _get_global_step(metrics_logger: MetricsLogger) -> int

    Returns the global step from the metrics logger.
    This is used to track the number of environment steps passed to the learner.

    Note:
        Requires the custom key `NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME`
        to be present in the metrics logger stats.
    """

    @staticmethod
    def _get_global_step(metrics_logger: MetricsLogger) -> int:
        """Assumes metrics_logger.stats is not empty and contains necessary keys."""
        # other possible keys are num_module_steps_sampled_lifetime/default_policy
        # or num_agent_steps_sampled_lifetime/default_agent
        if LEARNER_RESULTS in metrics_logger.stats:
            gs = metrics_logger.stats[LEARNER_RESULTS][ALL_MODULES][
                NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
            ].peek()  # NOTE: Custom key
        else:
            gs = metrics_logger.stats[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME].peek()
        # gs = metrics_logger.stats[ENV_RUNNERS][NUM_ENV_STEPS_SAMPLED_LIFETIME"].peek()
        # logger.debug("Global step %s", gs)
        return gs


class StepCounterMixin(GetGlobalStepMixin):
    """
    Attributes:
        _planned_current_step
        _training_iterations
        _get_global_step
    """

    def _set_step_counter_on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
    ) -> None:
        assert algorithm.config
        self._planned_current_step = 0
        self._training_iterations = 0
        # training_iterations is most likely not yet logged - only after the learner, therefore increase attr manually
        self._training_iterations = metrics_logger.peek("training_iteration", default=0) if metrics_logger else 0
        self._planned_current_step = (
            self._get_global_step(metrics_logger) if metrics_logger and metrics_logger.stats else 0
        )

    def _set_step_counter_on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: MetricsLogger,
    ) -> None:
        self._training_iterations += 1
        self._planned_current_step += (
            algorithm.config.total_train_batch_size  # pyright: ignore[reportOptionalMemberAccess]
        )  # pyright: ignore[reportOptionalMemberAccess]
        if self._planned_current_step != self._get_global_step(metrics_logger):
            _logger.error(
                "Expected step %d (%d + %d) but got %d instead. "
                "Expected step should at least be smaller but not larger.",
                self._planned_current_step,
                self._planned_current_step - algorithm.config.total_train_batch_size,  # pyright: ignore[reportOptionalMemberAccess]
                algorithm.config.total_train_batch_size,  # pyright: ignore[reportOptionalMemberAccess]
                self._get_global_step(metrics_logger),
            )

    def _set_step_counter_on_checkpoint_loaded(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: MetricsLogger,
    ) -> None:
        # TODO
        ...
