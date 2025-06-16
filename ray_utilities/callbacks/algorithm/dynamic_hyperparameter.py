from __future__ import annotations

import abc
import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics import LEARNER_RESULTS, ALL_MODULES  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import Self

from ray_utilities.constants import NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.core.learner.learner import Learner
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


logger = logging.getLogger(__name__)


class UpdateFunction(Protocol):
    def __call__(
        self: DynamicHyperparameterCallback | Any,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        *,
        global_step: int,
        **kwargs: Any,
    ) -> None: ...


class DynamicHyperparameterCallback(DefaultCallbacks, abc.ABC):
    @staticmethod
    def _update_worker(env_runner: EnvRunner | Learner, *args, key: str, value: Any):  # noqa: ARG004
        """
        Update function to update an environment runner or learner's configuration.

        Attention:
            As these objects have their own copy of the algorithm's configuration, they
            need to be updated separately from the algorithm if necessary.
        """
        object.__setattr__(env_runner.config, key, value)

    @classmethod
    def _update_algorithm(
        cls, algorithm: "Algorithm", *, key: str, value: Any, update_env_runners=True, update_learner=True
    ) -> None:
        """
        Update the algorithm's configuration and optionally the environment runners and learner as well.
        Env Runners and Learners have their own copy of the algorithm's configuration
        that need to be updated separately.
        """
        # Warn if config does not have this attr:
        if not hasattr(algorithm.config, key):
            logger.warning(
                "Algorithm config does not have attribute '%s' that is about to be set. Is it perhaps misspelled", key
            )
        object.__setattr__(algorithm.config, key, value)  # necessary hack for frozen objects.
        if update_env_runners or update_learner:
            update = partial(cls._update_worker, key=key, value=value)
        if update_env_runners and algorithm.env_runner_group:
            algorithm.env_runner_group.foreach_env_runner(update)  # pyright: ignore[reportPossiblyUnboundVariable]
        if update_learner and algorithm.learner_group:
            algorithm.learner_group.foreach_learner(update)  # pyright: ignore[reportPossiblyUnboundVariable]
        # TODO: Also change evaluation interval to be slower/faster at start/end when using dynamic buffer/batch

    @staticmethod
    def _get_global_step(metrics_logger: MetricsLogger) -> int:
        # other possible keys are num_module_steps_sampled_lifetime/default_policy
        # or num_agent_steps_sampled_lifetime/default_agent
        gs = metrics_logger.stats[LEARNER_RESULTS][ALL_MODULES][
            NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
        ].peek()  # NOTE: Custom key
        # otherwise:
        # gs = metrics_logger.stats[ENV_RUNNERS][NUM_ENV_STEPS_SAMPLED_LIFETIME"].peek()
        # logger.debug("Global step %s", gs)
        return gs

    def __init__(self, update_function: UpdateFunction, hyperparameter_name: str):
        self._updater = update_function
        self.hyperparameter_name: Final[str] = hyperparameter_name

    @classmethod
    def create_callback_class(cls, func: UpdateFunction, hyperparameter_name: str, **kwargs) -> partial[Self]:
        return partial(cls, update_function=func, hyperparameter_name=hyperparameter_name, **kwargs)

    def change_update_function(self, update_function: UpdateFunction) -> None:
        """Change the updater function."""
        self._updater = update_function

    @abc.abstractmethod
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        # TODO: Check if also called on checkpoint load
        if metrics_logger is None:
            logger.warning(
                "Metrics logger is None in on_algorithm_init. "
                "This may lead to incorrect global step handling when loading checkpoints."
            )
        self._updater(
            algorithm,
            metrics_logger,
            global_step=self._get_global_step(metrics_logger) if metrics_logger else 0,
        )

    def on_checkpoint_loaded(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        # NOTE: Likely no metrics_logger here.
        self._updater(algorithm, None, global_step="???")
