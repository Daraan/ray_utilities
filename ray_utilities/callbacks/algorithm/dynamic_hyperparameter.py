from __future__ import annotations

import abc
import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing_extensions import Self

from ray_utilities.callbacks.algorithm.callback_mixins import GetGlobalStepMixin

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.core.learner.learner import Learner
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


logger = logging.getLogger(__name__)


class UpdateFunction(Protocol):
    def __call__(
        self: DynamicHyperparameterCallback | Any,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        *,
        global_step: int,
        **kwargs: Any,
    ) -> None: ...


class DynamicHyperparameterCallback(GetGlobalStepMixin, DefaultCallbacks, abc.ABC):
    @staticmethod
    def _update_worker(env_runner: EnvRunner | Learner, *args, key: str, value: Any):  # noqa: ARG004
        """
        Update function to update an environment runner or learner's configuration.

        Attention:
            As these objects have their own copy of the algorithm's configuration, they
            need to be updated separately from the algorithm if necessary.
        """
        object.__setattr__(env_runner.config, key, value)

    @staticmethod
    def _update_worker_learner_config(env_runner: EnvRunner | Learner, *args, update: dict[str, Any]):  # noqa: ARG004
        env_runner.config.learner_config_dict.update(update)

    @classmethod
    def _update_algorithm(
        cls, algorithm: Algorithm, *, key: str, value: Any, update_env_runners=True, update_learner=True
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
        # HACK algorithm.config is FROZEN
        object.__setattr__(algorithm.config, key, value)  # necessary hack for frozen objects.
        if update_env_runners or update_learner:
            update = partial(cls._update_worker, key=key, value=value)
        if update_env_runners and algorithm.env_runner_group:
            algorithm.env_runner_group.foreach_env_runner(update)  # pyright: ignore[reportPossiblyUnboundVariable]
        if update_learner and algorithm.learner_group:
            algorithm.learner_group.foreach_learner(update)  # pyright: ignore[reportPossiblyUnboundVariable]

    @classmethod
    def _update_learner_config(
        cls, algorithm: Algorithm, *, key: str | None = None, value: Any = None, update_env_runners=True, **kwargs
    ) -> None:
        """
        Update the algorithm's configuration and optionally the environment runners and learner as well.
        Env Runners and Learners have their own copy of the algorithm's configuration
        that need to be updated separately.
        """
        # Warn if config does not have this attr:
        assert algorithm.config
        if (key is None and (not kwargs or value is not None)) or (key is not None and (kwargs or value is None)):
            raise ValueError(
                f"Either kwargs or key and value must be provided, but not both. key: {key}, value: {value}, kwargs: {kwargs}"
            )
        if key:
            kwargs = {key: value}
        algorithm.config.learner_config_dict.update(kwargs)
        update = partial(cls._update_worker_learner_config, update=kwargs)
        if update_env_runners and algorithm.env_runner_group:
            algorithm.env_runner_group.foreach_env_runner(update)  # pyright: ignore[reportPossiblyUnboundVariable]
        if algorithm.learner_group:  # NOTE: object implements __len__ 0 if local!
            algorithm.learner_group.foreach_learner(update)  # pyright: ignore[reportPossiblyUnboundVariable]
        elif algorithm.learner_group is not None:
            if algorithm.learner_group._learner is None:
                logger.error("Algorithm learner_group is local but learner is None, cannot update")
            else:
                update(algorithm.learner_group._learner)

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
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        # TODO: Check if also called on checkpoint load
        if metrics_logger is None:
            logger.warning(
                "Metrics logger is None in on_algorithm_init. "
                "This may lead to incorrect global step handling when loading checkpoints."
            )
            gs = 0
        elif metrics_logger.stats:
            # Likely this is not the case and we call updater with 0
            logger.debug("Algorithm initialized with stats already present")
            # NOTE: # TODO: In initial version this used gs = 0 and called get_global_step below
            gs = self._get_global_step(metrics_logger)
        else:
            # stats is empty cannot retire global step
            # TODO: # Test on checkpoint load this should not be 0.
            gs = 0
        self._updater(
            algorithm,
            metrics_logger,
            global_step=gs,
        )

    def on_checkpoint_loaded(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        """
        Attention:
            When using the StepCounterMixin - which this class does not -
            also call _set_step_counter_on_checkpoint_loaded in an overridden method.
        """
        # NOTE: Likely no metrics_logger here.
        assert metrics_logger
        gs = self._get_global_step(metrics_logger=metrics_logger)
        self._updater(algorithm, metrics_logger, global_step=gs)
        super().on_checkpoint_loaded(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)
