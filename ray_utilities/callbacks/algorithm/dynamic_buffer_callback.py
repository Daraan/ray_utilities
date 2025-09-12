# ruff: noqa: ARG002
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ray_utilities.callbacks.algorithm.callback_mixins import BudgetMixin, StepCounterMixin
from ray_utilities.callbacks.algorithm.dynamic_hyperparameter import DynamicHyperparameterCallback, UpdateFunction
from ray_utilities.dynamic_config.dynamic_buffer_update import update_buffer_and_rollout_size
from ray_utilities.warn import warn_if_batch_size_not_divisible

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


logger = logging.getLogger(__name__)


class DynamicBufferUpdate(StepCounterMixin, BudgetMixin, DynamicHyperparameterCallback):
    """
    Attributes:
        updater
    """

    def _calculate_buffer_and_batch_size(
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
        # budget = split_timestep_budget(
        #    total_steps=args.total_steps,
        #    min_size=learner_config_dict["min_dynamic_buffer_size"],
        #    max_size=learner_config_dict["max_dynamic_buffer_size"],
        #    assure_even=True,
        # )
        # these are Stats objects
        batch_size, _accumulate_gradients_every, env_steps = update_buffer_and_rollout_size(
            total_steps=learner_config_dict["total_steps"],  # test if budget total_steps is fine as well
            dynamic_buffer=learner_config_dict["dynamic_buffer"],
            dynamic_batch=learner_config_dict["dynamic_batch"],
            global_step=global_step,
            accumulate_gradients_every_initial=1,
            initial_steps=self._budget["step_sizes"][0],
            num_increase_factors=len(self._budget["step_sizes"]),
            n_envs=1,
        )
        # TEST:  # TODO remove
        step_index = 0
        iterations = self._budget["iterations_per_step_size"][0]
        import numpy as np

        np.cumsum(np.multiply(self._budget["step_sizes"], self._budget["iterations_per_step_size"]))
        for step_index, iterations in enumerate(  # noqa: B007
            np.cumsum(np.multiply(self._budget["step_sizes"], self._budget["iterations_per_step_size"]))
        ):
            if global_step < iterations:
                break
        assert env_steps == self._budget["step_sizes"][step_index]
        # Log current behavior
        if self._training_iterations % 4 == 0:
            logger.debug(
                "Step %s & Iteration %s: batch size %s, env_steps %s",
                global_step,
                self._training_iterations,
                batch_size,
                env_steps,
            )
        # Batch Size
        if batch_size != self._batch_size_current:
            logger.debug(
                "Batch size changed from %s to %s at iteration %s - step %s",
                self._batch_size_current,
                batch_size,
                metrics_logger.peek("training_iteration", default=self._training_iterations),
                global_step,
            )
            self._batch_size_current = batch_size
            assert env_steps == batch_size
            # TODO: Both are currently the same, batch size should likely be minibatch size.
            # object.__setattr__(algorithm.config, "_train_batch_size_per_learner", n_steps)
        # Rollout Size
        if env_steps != self._env_steps_current:
            logger.debug(
                "Rollout size changed from %s to %s at iteration %s - step %s",
                self._env_steps_current,
                env_steps,
                metrics_logger.peek("training_iteration", default=self._training_iterations),
                global_step,
            )
            self._env_steps_current = env_steps
            assert algorithm.config
            assert algorithm.config.num_envs_per_env_runner is not None
            # Warn if steps are not divisible by num_envs_per_env_runner
            warn_if_batch_size_not_divisible(
                batch_size=env_steps, num_envs_per_env_runner=algorithm.config.num_envs_per_env_runner
            )
            self._update_algorithm(algorithm, key="_train_batch_size_per_learner", value=env_steps)
            # decrease minibatch size if necessary to minibatch == batch_size
            if env_steps < self._initial_minibatch_size:
                logger.debug(
                    "Minibatch size changed from %s to %s; as it is larger than the new batch size %d. "
                    "Note: The first noted value might be incorrect if adjusted after the callback init",
                    self._initial_minibatch_size,
                    env_steps,
                    env_steps,
                )
                self._update_algorithm(algorithm, key="minibatch_size", value=env_steps)
            elif algorithm.config.minibatch_size != self._initial_minibatch_size:
                logger.debug("Resetting minibatch_size to %s", self._initial_minibatch_size)
                self._update_algorithm(algorithm, key="minibatch_size", value=self._initial_minibatch_size)
        # maybe need to also adjust epochs -> cycled over, same amount of minibatches until rollout > batch_size;
        # then whole rollout is consumed
        # theoretically also increase minibatch size.

        # In legacy batch_size = int(args.n_envs * n_steps)

    def __init__(
        self, update_function: UpdateFunction | None = None, learner_config_dict: dict[Any, Any] | None = None
    ):
        """

        Args:
            update_func: Function to update the buffer and batch size.
            learner_config_dict: Configuration dictionary for the learner. At best this is the same object as
                `algorithm.config.learner_config_dict` to ensure that the values are updated correctly.
        """
        super().__init__(update_function or self._calculate_buffer_and_batch_size, "TBA - DynamicBufferUpdate")
        self._set_budget_on__init__(learner_config_dict=learner_config_dict)
        # Set on algorithm init
        self._initial_minibatch_size: int = None
        self._batch_size_current: int = None
        self._env_steps_current: int = None

    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        # TODO: What when using checkpoint?
        assert algorithm.config
        assert algorithm.config.minibatch_size is not None
        self._set_step_counter_on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger)
        self._set_budget_on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
        logger.debug("Initial rollout size for DynamicBuffer %s", self._budget["step_sizes"][0])
        # legacy only; might be wrong if adjusted later because minibatch_size > batch_size
        self._initial_minibatch_size = min(
            algorithm.config.minibatch_size, algorithm.config.train_batch_size_per_learner
        )
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
        # Safer way to get correct steps, # TODO: re-evaluate: possible use the one from metrics_logger instead
        # Updater needs to change current_step_planned for NEXT step in case there is an increase in batch size.
        batch_size_old = algorithm.config.train_batch_size_per_learner  # pyright: ignore[reportOptionalMemberAccess]
        self._set_step_counter_on_train_result(algorithm=algorithm, metrics_logger=metrics_logger)
        self._updater(algorithm, metrics_logger, global_step=self._planned_current_step)  # pyright: ignore[reportArgumentType]
        if algorithm.config.train_batch_size_per_learner != batch_size_old:  # pyright: ignore[reportOptionalMemberAccess]
            # report that batch_size has changed
            self.batch_size_changed(
                iteration=self._training_iterations,
                old_batch_size=batch_size_old,
                new_batch_size=algorithm.config.train_batch_size_per_learner,  # pyright: ignore[reportOptionalMemberAccess]
            )

        # if train_batch_size_per_learner changed adjust other callbacks
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


def make_dynamic_buffer_callback(func: UpdateFunction) -> type[DynamicBufferUpdate]:
    """Create a callback that seeds the environment."""

    class CustomDynamicBufferUpdate(DynamicBufferUpdate):
        _calculate_buffer_and_batch_size = func  # type: ignore[assignment]

    return CustomDynamicBufferUpdate
