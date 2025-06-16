# ruff: noqa: ARG002
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional


from ray_utilities.callbacks.algorithm.dynamic_hyperparameter import DynamicHyperparameterCallback, UpdateFunction
from ray_utilities.dynamic_config.dynamic_buffer_update import (
    UpdateNStepsArgs,
    get_dynamic_evaluation_intervals,
    split_timestep_budget,
    update_buffer_and_rollout_size,
)

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


logger = logging.getLogger(__name__)


class DynamicBufferUpdate(DynamicHyperparameterCallback):
    """
    Attributes:
        updater
    """

    @dataclass
    class _UpdateArgs(UpdateNStepsArgs):
        total_steps: int
        n_envs: int
        dynamic_buffer: bool
        static_batch: bool

    def _calculate_buffer_and_batch_size(
        self,
        algorithm: "Algorithm",
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
        # TEST:
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
            # HACK algorithm.config is FROZEN
            # algorihm.config.train_batch_size_per_learner = n_steps
            self._update_algorithm(algorithm, key="_train_batch_size_per_learner", value=env_steps)
            # decrease minibatch size if necessary to minibatch == batch_size
            if env_steps < self._initial_minibatch_size:
                logger.debug("Minibatch size changed from %s to %s", self._initial_minibatch_size, env_steps)
                self._update_algorithm(algorithm, key="minibatch_size", value=env_steps)
            elif algorithm.config.minibatch_size != self._initial_minibatch_size:
                logger.debug("Resetting minibatch_size to %s", self._initial_minibatch_size)
                self._update_algorithm(algorithm, key="minibatch_size", value=self._initial_minibatch_size)

            # Change evaluation interval
            new_eval_interval = self._evaluation_intervals.get(env_steps, None)
            if new_eval_interval is None:
                logger.error(
                    "No evaluation interval for %s steps in %s. Expected a value in the dictionary.",
                    env_steps,
                    self._evaluation_intervals,
                )
                # do not change it
            elif algorithm.config.evaluation_interval != new_eval_interval:
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
        # Set on algorithm init
        self._accumulate_gradients_every_initial: int = None
        self._initial_minibatch_size: int = None
        self._batch_size_current: int = None
        self._env_steps_current: int = None
        self._accumulate_gradients_every_current: int = None
        self._planned_current_step: int = None
        if learner_config_dict:
            if "total_steps" not in learner_config_dict:
                logger.warning(
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
                logger.warning(
                    "Missing key in learner_config_dict: %s during creation of %s. "
                    "Potentially this callback is created before setting the learner_config_dict. "
                    "If the key is not present during the algorithm initialization, "
                    "this will raise an error later.",
                    e,
                    self.__class__.__name__,
                )
        else:
            self._budget = None

    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        # TODO: What when using checkpoint?
        assert algorithm.config
        assert algorithm.config.minibatch_size is not None
        learner_config_dict = algorithm.config.learner_config_dict
        try:
            budget = split_timestep_budget(
                total_steps=learner_config_dict["total_steps"],
                min_size=learner_config_dict["min_dynamic_buffer_size"],
                max_size=learner_config_dict["max_dynamic_buffer_size"],
                assure_even=True,
            )
        except KeyError as e:
            logger.error("Missing key in learner_config_dict: %s", e)
            raise
        # TEST #XXX
        if self._budget is not None:
            assert budget == self._budget
        self._budget = budget
        logger.debug("Initial rollout size for DynamicBuffer %s", self._budget["step_sizes"][0])
        self._evaluation_intervals: dict[int, int] = dict(
            zip(
                self._budget["step_sizes"],
                (
                    get_dynamic_evaluation_intervals(
                        self._budget["step_sizes"],
                        eval_freq=algorithm.config.evaluation_interval,
                        batch_size=algorithm.config.train_batch_size_per_learner,
                        take_root=True,
                    )
                    # 0 for no evaluation
                    if algorithm.config.evaluation_interval
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
        # legacy only
        self._accumulate_gradients_every_initial = 1
        self._initial_minibatch_size = algorithm.config.minibatch_size
        # training_iterations is most likely not yet logged - only after the learner, therefore increase attr manually
        self._training_iterations = metrics_logger.peek("training_iteration", default=0) if metrics_logger else 0
        self._planned_current_step = (
            self._get_global_step(metrics_logger) if metrics_logger and metrics_logger.stats else 0
        )

        # stats is empty initially
        if metrics_logger and metrics_logger.stats:
            logger.debug("Algorithm initialized with stats already present")
            self._updater(algorithm, metrics_logger, global_step=self._get_global_step(metrics_logger))
        else:
            self._updater(algorithm, metrics_logger, global_step=0)

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        result: dict,
        **kwargs,
    ) -> None:
        self._training_iterations += 1
        self._planned_current_step += algorithm.config.total_train_batch_size  # pyright: ignore[reportOptionalMemberAccess]
        assert metrics_logger
        # logger.debug("Expected step: %d, reported step: %d", self._planned_current_step, self._get_global_step(metrics_logger))
        if self._planned_current_step != self._get_global_step(metrics_logger):
            logger.error(
                "Expected step %d (%d + %d) but got %d instead. "
                "Expected step should at least be smaller but not larger.",
                self._planned_current_step,
                self._planned_current_step - algorithm.config.total_train_batch_size,  # pyright: ignore[reportOptionalMemberAccess]
                algorithm.config.total_train_batch_size,  # pyright: ignore[reportOptionalMemberAccess]
                self._get_global_step(metrics_logger),
            )
        # Safer way to get correct steps:
        self._updater(algorithm, metrics_logger, global_step=self._planned_current_step)  # pyright: ignore[reportArgumentType]
        # Exact way to update steps:
        # self._updater(
        #    algorithm, metrics_logger, global_step=self._get_global_step(metrics_logger)
        # )

    # num_module_steps_trained', 'num_module_steps_trained_lifetime'

    def on_checkpoint_loaded(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        self._updater(algorithm, None, global_step=None)
        # TODO: self._training_iterations = 0


def make_dynamic_buffer_callback(func: UpdateFunction) -> type[DynamicBufferUpdate]:
    """Create a callback that seeds the environment."""

    class CustomDynamicBufferUpdate(DynamicBufferUpdate):
        _calculate_buffer_and_batch_size = func  # type: ignore[assignment]

    return CustomDynamicBufferUpdate


if __name__ == "__main__":
    # simulate Increase
    from ray_utilities import nice_logger
    from ray_utilities.dynamic_config.dynamic_buffer_update import split_timestep_budget

    nice_logger(logger, "DEBUG")
    global_step = 0
    dynamic_buffer = True
    dynamic_batch = True
    iterations = 340
    total_steps = 1_000_000
    # Test
    budget = split_timestep_budget(
        total_steps=total_steps,
        min_size=32,
        max_size=2**13,
        assure_even=True,
    )
    total_steps = budget["total_steps"]
    n_envs = 1
    n_steps = 2048 // 8
    initial_steps = budget["step_sizes"][0]  # 128
    # batch_size = n_envs * n_steps
    n_steps_old = None

    while global_step < total_steps:
        global_step += n_steps
        batch_size, _, n_steps = update_buffer_and_rollout_size(
            total_steps=total_steps,
            dynamic_buffer=dynamic_buffer,
            dynamic_batch=dynamic_batch,
            global_step=global_step,
            accumulate_gradients_every_initial=1,
            initial_steps=initial_steps,
            num_increase_factors=len(budget["step_sizes"]),
            n_envs=1,
        )
        if n_steps_old != n_steps:
            n_steps_old = n_steps
            logger.debug(
                "Updating at step %d / %s (%f%%) to '%s x %s=%s' from initially (%s), batch size=%s (initial: %s)",
                global_step,
                total_steps,
                round((global_step / total_steps) * 100, 0),
                n_steps,
                1,
                n_steps * 1,
                initial_steps,
                batch_size,
                int(n_envs * initial_steps),
            )
    logger.debug(
        "Finished at step %d / %s (%f%%) to '%s x %s=%s' from initially (%s), batch size=%s (initial: %s)",
        global_step,
        total_steps,
        round((global_step / total_steps) * 100, 0),
        n_steps,
        1,
        n_steps * 1,
        initial_steps,
        batch_size,  # pyright: ignore[reportPossiblyUnboundVariable]
        int(n_envs * initial_steps),
    )
