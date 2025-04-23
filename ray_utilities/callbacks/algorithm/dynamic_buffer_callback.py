from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol

from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray_utilities.dynamic_buffer_update import UpdateNStepsArgs, update_buffer_and_rollout_size

if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


logger = logging.getLogger(__name__)

class UpdateFunction(Protocol):
    def __call__(
        self,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        *,
        global_step: int,
    ) -> None:
        ...

class DynamicBufferUpdate(DefaultCallbacks):
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

    @staticmethod
    def _update_worker(env_runner: EnvRunner | Learner, *args, key: str, value: Any):  # noqa: ARG004
        object.__setattr__(env_runner.config, key, value)

    @classmethod
    def _update_algorithm(cls, algorithm: "Algorithm", *, key, value, update_env_runners=True, update_learner=True):
        object.__setattr__(algorithm.config, key, value)
        if update_env_runners or update_learner:
            update = partial(cls._update_worker, key=key, value=value)
        if update_env_runners and algorithm.env_runner_group:
            algorithm.env_runner_group.foreach_env_runner(update)  # pyright: ignore[reportPossiblyUnboundVariable]
        if update_learner and algorithm.learner_group:
            algorithm.learner_group.foreach_learner(update)  # pyright: ignore[reportPossiblyUnboundVariable]

    def _calculate_buffer_and_batch_size(
        self, algorithm: "Algorithm", metrics_logger: Optional[MetricsLogger], *, global_step: int
    ) -> None:
        assert algorithm.config
        # Need to modify algorithm.config.train_batch_size_per_learner
        args = self._UpdateArgs(
            n_envs=1,
            total_steps=1_000_000,  # episodes/training_steps (-e) x steps_per_episode
            # batch_size (4096) * epochs (20) * episodes (100) =
            static_batch=not algorithm.config.learner_config_dict["dynamic_batch"],
            dynamic_buffer=algorithm.config.learner_config_dict["dynamic_buffer"],
        )
        # these are Stats objects
        batch_size, _accumulate_gradients_every, n_steps = update_buffer_and_rollout_size(
            args,
            global_step=global_step,
            accumulate_gradients_every_initial=1,
            initial_steps=self._initial_steps,  # train_batch_size_per_learner // 8
        )
        if batch_size != self._batch_size_old:
            logger.debug("Batch size changed from %s to %s", self._batch_size_old, batch_size)
            self._batch_size_old = batch_size
            assert n_steps == batch_size
            # Both are currently the same!
            # object.__setattr__(algorihm.config, "_train_batch_size_per_learner", n_steps)
        if n_steps != self._n_steps_old:
            logger.debug("Rollout size changed from %s to %s", self._n_steps_old, n_steps)
            self._n_steps_old = n_steps
            assert algorithm.config
            # HACK algorithm.config is FROZEN
            # algorihm.config.train_batch_size_per_learner = n_steps
            self._update_algorithm(algorithm, key="_train_batch_size_per_learner", value=n_steps)
            # decrease minibatch size if necessary to minibatch == batch_size
            if n_steps < self._initial_minibatch_size:
                logger.debug("Minibatch size changed from %s to %s", self._initial_minibatch_size, n_steps)
                self._update_algorithm(algorithm, key="minibatch_size", value=n_steps)
            elif algorithm.config.minibatch_size != self._initial_minibatch_size:
                logger.debug("Resetting minibatch_size to %s", self._initial_minibatch_size)
                self._update_algorithm(algorithm, key="minibatch_size", value=self._initial_minibatch_size)

        # maybe need to also adjust epochs -> cycled over, same amount of minibatches until rollout > batch_size;
        # then whole rollout is consumed
        # theoretically also increase minibatch size.

        # In legacy batch_size = int(args.n_envs * n_steps)

    def __init__(self, update_func: UpdateFunction | None = None):
        if update_func is not None:
            self._updater = update_func
        else:
            self._updater = self._calculate_buffer_and_batch_size
        self._initial_steps: int = None  # train_batch_size_per_learner // 8
        self._accumulate_gradients_every_initial: int = None
        self._initial_minibatch_size: int = None

        self._batch_size_old: int = None
        self._n_steps_old: int = None
        self._accumulate_gradients_every_old: int = None

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
        n_steps = algorithm.config.train_batch_size_per_learner
        if algorithm.config.learner_config_dict["dynamic_buffer"]:
            n_steps = max(16, n_steps // 8)  # TODO: 8 is hardcoded
        self._initial_steps = n_steps
        logger.debug("Initial rollout size for DynamicBuffer %s", self._initial_steps)
        # legacy only
        self._accumulate_gradients_every_initial = 1
        self._initial_minibatch_size = algorithm.config.minibatch_size
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
        self._updater(algorithm, metrics_logger, global_step=self._get_global_step(metrics_logger))  # pyright: ignore[reportArgumentType]
    # num_module_steps_trained', 'num_module_steps_trained_lifetime'

    def on_checkpoint_loaded(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        self._initial_steps = None  # TODO: Get the current step
        self._updater(algorithm, None, global_step=None)

    @staticmethod
    def _get_global_step(metrics_logger: MetricsLogger) -> int:
        # other possible keys are num_module_steps_sampled_lifetime/default_policy
        # or num_agent_steps_sampled_lifetime/default_agent
        gs = metrics_logger.stats["env_runners"]["num_env_steps_sampled_lifetime"].peek()
        logger.debug("Global step %s", gs)
        return gs


def make_dynamic_buffer_callback(func: UpdateFunction) -> type[DynamicBufferUpdate]:
    """Create a callback that seeds the environment."""
    class CustomDynamicBufferUpdate(DynamicBufferUpdate):
        _calculate_buffer_and_batch_size = func

    return CustomDynamicBufferUpdate


if __name__ == "__main__":
    # simulate Increase
    from ray_utilities import nice_logger
    nice_logger(logger, "DEBUG")
    global_step = 0
    dynamic_buffer = True
    dynamic_batch = True
    total_steps = 1_000_000
    n_envs = 1
    n_steps = 512
    initial_steps = n_steps
    batch_size = n_envs * n_steps
    n_steps_old = None

    args = DynamicBufferUpdate._UpdateArgs(
        n_envs=1,
        total_steps=total_steps,
        static_batch=not dynamic_batch,
        dynamic_buffer=dynamic_buffer,
    )

    while global_step < total_steps:
        global_step += n_steps
        batch_size, _, n_steps = update_buffer_and_rollout_size(
            args,
            global_step=global_step,
            accumulate_gradients_every_initial=1,
            initial_steps=initial_steps,
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
