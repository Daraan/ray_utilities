from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, Mapping, Optional, TypeVar

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.metrics import (
    ALL_MODULES,  # pyright: ignore[reportPrivateImportUsage]
    ENV_RUNNER_RESULTS,
    LEARNER_RESULTS,
)

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
        ``_set_budget_on__init__(learner_config_dict: dict[Any, Any] | None = None, **kwargs: Any) -> None``
        ``_set_budget_on_algorithm_init(algorithm: Algorithm, **kwargs: Any) -> None``
            Initializes the budget based on the learner_config_dict from the algorithm's config.
        ``_set_budget_on_checkpoint_loaded(algorithm: Algorithm, **kwargs: Any) -> None``
    """

    def _set_budget_on__init__(self, learner_config_dict: dict[Any, Any] | T = None):
        self._budget: SplitBudgetReturnDict | T
        if learner_config_dict:
            # NOTE: Currently pyright 1.1.402+ has a bug that does not narrow T on truthy.
            assert learner_config_dict is not None
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
        algorithm: Algorithm,
        allow_budget_change: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """on_load_checkpoint set allow_budget_change=True"""
        assert algorithm.config
        learner_config_dict = algorithm.config.learner_config_dict
        if algorithm.config.train_batch_size_per_learner < learner_config_dict["min_dynamic_buffer_size"] or (
            algorithm.config.train_batch_size_per_learner > learner_config_dict["max_dynamic_buffer_size"]
        ):
            _logger.warning(
                "train_batch_size_per_learner (%d) is outside of the dynamic buffer size range [%d, %d]. "
                "This might lead to unexpected behavior.",
                algorithm.config.train_batch_size_per_learner,
                learner_config_dict["min_dynamic_buffer_size"],
                learner_config_dict["max_dynamic_buffer_size"],
            )
        try:
            batch_size = algorithm.config.train_batch_size_per_learner
            budget = split_timestep_budget(
                total_steps=learner_config_dict["total_steps"],
                min_size=min(batch_size, learner_config_dict["min_dynamic_buffer_size"]),
                max_size=max(batch_size, learner_config_dict["max_dynamic_buffer_size"]),
                assure_even=True,
            )
        except KeyError as e:
            _logger.error("Missing key in learner_config_dict: %s", e)
            raise
        if not allow_budget_change and self._budget is not None:
            assert budget == self._budget, "Budget dict changed since initialization."
        self._budget = budget

    def _set_budget_on_checkpoint_loaded(self, *, algorithm: Algorithm, **kwargs) -> None:
        if self._budget is None:
            _logger.error("BudgetMixin._budget is None. Need to recreate.")
        self._set_budget_on_algorithm_init(algorithm=algorithm, allow_budget_change=True)


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
        # look for our custom keys
        gs = metrics_logger.peek((LEARNER_RESULTS, ALL_MODULES, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), default=-1)
        if gs == -1:  # alternative custom key
            gs = metrics_logger.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), default=-1)
        if gs == -1:
            assert metrics_logger.peek("training_iteration", default=0) == 0
            gs = 0
        # For now to not fall back to rllib key to reveal errors.
        # gs = metrics_logger.stats[ENV_RUNNERS][NUM_ENV_STEPS_SAMPLED_LIFETIME"].peek()
        # logger.debug("Global step %s", gs)
        return int(gs)


class StepCounterMixin(GetGlobalStepMixin):
    """
    Attributes:
        _planned_current_step
        _training_iterations
        _get_global_step
    """

    _batch_size_changes: Final[dict[int, tuple[int, int]]] = {}

    @classmethod
    def batch_size_changed(cls, *, iteration: int, old_batch_size: int, new_batch_size: int):
        StepCounterMixin._batch_size_changes[iteration] = (old_batch_size, new_batch_size)

    @classmethod
    def _did_batch_size_change(cls, iteration: int) -> Literal[False] | tuple[int, int]:
        if iteration in cls._batch_size_changes:
            return cls._batch_size_changes[iteration]
        return False

    def _set_step_counter_on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
    ) -> None:
        assert algorithm.config
        self._training_iterations: int = 0
        # training_iterations is most likely not yet logged - only after the learner, therefore increase attr manually
        self._training_iterations = metrics_logger.peek("training_iteration", default=0) if metrics_logger else 0
        self._planned_current_step: int = (
            self._get_global_step(metrics_logger) if metrics_logger and metrics_logger.stats else 0
        )

    def _set_step_counter_on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger,
    ) -> None:
        """
        Increment the step counter _training_iteration by one and update the planned current step.
        Note:
            If the updater changes the total_train_batch_size, the planned_current_step needs to
            be adjusted *after* (or in) the updater call.
            Such an updater should be the first callback in the list of callbacks.
        """
        self._training_iterations += 1
        # NOTE: When updating the total_train_batch_size this step; then planned_current_step needs to be adjusted
        if size_change := self._did_batch_size_change(self._training_iterations):
            # update with old batch size to match the logger result, new is used for the next step
            self._planned_current_step += size_change[0]
        else:
            self._planned_current_step += algorithm.config.total_train_batch_size  # pyright: ignore[reportOptionalMemberAccess]
        if self._planned_current_step != (global_step := self._get_global_step(metrics_logger)):
            _logger.error(
                "%s: Expected step %d (%d + %d) but got %d (difference: %d) instead. Iteration: %d. "
                "Expected step should at least be smaller but not larger: %s",
                self.__class__.__name__,
                self._planned_current_step,
                self._planned_current_step - algorithm.config.total_train_batch_size,  # pyright: ignore[reportOptionalMemberAccess]
                algorithm.config.total_train_batch_size,  # pyright: ignore[reportOptionalMemberAccess]
                global_step,
                global_step - self._planned_current_step,
                self._training_iterations,
                "OK" if self._planned_current_step < global_step else "NOT OK",
                stacklevel=2,
            )
        # else:
        #    _logger.info(
        #        "%s: Step counter as expected: %d==%d (training_iterations: %d, total_train_batch_size: %d)",
        #        self.__class__.__name__,
        #        self._planned_current_step,
        #        global_step,
        #        self._training_iterations,
        #        algorithm.config.total_train_batch_size,  # pyright: ignore[reportOptionalMemberAccess]
        #    )

    def _set_step_counter_on_checkpoint_loaded(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger,
    ) -> None:
        self._planned_current_step = metrics_logger.peek(
            (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME), default=-1
        )
        if self._planned_current_step == -1:
            # this should only happen if 0 training steps have been done:
            assert metrics_logger.peek("training_iteration", default=0) == 0
            self._planned_current_step = 0
