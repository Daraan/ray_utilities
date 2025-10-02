"""Ray Tune integration for hyperparameter optimization of RLlib experiments.

This module provides the :class:`TunerSetup` class that integrates Ray Tune
hyperparameter optimization with Ray RLlib experiment setups. It handles the
configuration of search algorithms, schedulers, stoppers, and run management
for scalable hyperparameter tuning workflows.

The module bridges the gap between experiment setup classes and Ray Tune,
providing a clean interface for running hyperparameter optimization with
minimal configuration while supporting advanced features like Optuna integration,
pruning, and custom trial naming.

Key Components:
    - :class:`TunerSetup`: Main class for configuring and running hyperparameter tuning
    - Protocol definitions for type safety and flexibility
    - Integration with Optuna search algorithms and pruning
    - Custom stoppers and schedulers for efficient optimization

The setup integrates seamlessly with :class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase`
subclasses to provide a complete solution for hyperparameter optimization of
reinforcement learning experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Optional, Protocol, cast, overload

from ray import train, tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import CombinedStopper, FunctionStopper
from typing_extensions import TypeVar

from ray_utilities.callbacks.tuner.metric_checkpointer import TUNE_RESULT_IS_A_COPY, StepCheckpointer
from ray_utilities.config._tuner_callbacks_setup import TunerCallbackSetup
from ray_utilities.constants import (
    CLI_REPORTER_PARAMETER_COLUMNS,
    DEFAULT_EVAL_METRIC,
    EVAL_METRIC_RETURN_MEAN,
    FORK_FROM,
    NEW_LOG_EVAL_METRIC,
)
from ray_utilities.misc import new_log_format_used, trial_name_creator as default_trial_name_creator
from ray_utilities.setup.optuna_setup import OptunaSearchWithPruner, create_search_algo
from ray_utilities.training.helpers import get_current_step
from ray_utilities.tune.stoppers.maximum_iteration_stopper import MaximumResultIterationStopper

if TYPE_CHECKING:
    from ray.air.config import RunConfig as RunConfigV1
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.tune.execution.placement_groups import PlacementGroupFactory
    from ray.tune.experiment import Trial
    from ray.tune.stopper import Stopper

    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase
    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData


__all__ = [
    "TunerSetup",
]

SetupType_co = TypeVar(
    "SetupType_co", bound="ExperimentSetupBase[DefaultArgumentParser, AlgorithmConfig, Algorithm]", covariant=True
)

logger = logging.getLogger(__name__)


class _TunerSetupBase(Protocol):
    eval_metric: str
    eval_metric_order: Literal["max", "min"]
    trial_name_creator: Callable[[Trial], str]

    def create_tune_config(self) -> tune.TuneConfig: ...

    def create_run_config(
        self, callbacks: list[tune.Callback] | list[train.UserCallback]
    ) -> tune.RunConfig | RunConfigV1 | train.RunConfig: ...

    def create_tuner(self, *args, **kwargs) -> tune.Tuner:
        """Create and return a configured Ray Tune Tuner instance."""
        ...


class TunerSetup(TunerCallbackSetup, _TunerSetupBase, Generic[SetupType_co]):
    """Configuration and management class for Ray Tune hyperparameter optimization.

    This class provides a comprehensive interface for setting up and running
    hyperparameter optimization experiments with Ray Tune, integrating seamlessly
    with Ray RLlib experiment setups. It handles the configuration of search
    algorithms, schedulers, stoppers, and callbacks to provide efficient and
    scalable hyperparameter tuning.

    The class supports various optimization features including Optuna integration
    for advanced search algorithms, automatic pruning of poor-performing trials,
    custom trial naming, and flexible stopping criteria based on training progress.

    Features:
        - Integration with :class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase`
        - Optuna search algorithm support with pruning capabilities
        - Flexible stopping criteria (total steps, iterations, custom functions)
        - Automatic trial naming and experiment organization
        - Callback system for monitoring and custom behaviors
        - Support for both maximization and minimization objectives

    Args:
        eval_metric: Name of the metric to optimize (default: ``"eval/episode_return_mean"``)
        eval_metric_order: Whether to maximize (``"max"``) or minimize (``"min"``) the metric
        setup: The experiment setup instance to optimize
        extra_tags: Additional tags for experiment organization
        add_iteration_stopper: Whether to add a maximum iteration stopper based on
            :attr:`setup.args.iterations<DefaultArgumentParser.iterations>`.
            - True for always (if `setup.args.iterations` is an integer),
            - False for never (Trainable or other stoppers need to determine end.).
            - None

    Attributes:
        eval_metric: The metric being optimized
        eval_metric_order: Optimization direction ("max" or "min")
        trial_name_creator: Function for generating trial names

    Example:
        >>> from ray_utilities.setup import PPOSetup
        >>> setup = PPOSetup()
        >>> tuner_setup = TunerSetup(eval_metric="eval/episode_return_mean", eval_metric_order="max", setup=setup)
        >>> tuner = tuner_setup.create_tuner()
        >>> results = tuner.fit()

    See Also:
        :class:`~ray_utilities.setup.experiment_base.ExperimentSetupBase`: Base experiment setup
        :class:`~ray_utilities.config._tuner_callbacks_setup.TunerCallbackSetup`: Callback management
        :func:`~ray_utilities.misc.trial_name_creator`: Trial naming function
    """

    trial_name_creator = staticmethod(default_trial_name_creator)

    def __init__(
        self,
        eval_metric: str | DEFAULT_EVAL_METRIC = EVAL_METRIC_RETURN_MEAN,
        eval_metric_order: Literal["max", "min"] = "max",
        *,
        setup: SetupType_co,  # ExperimentSetupBase[ParserTypeT, ConfigTypeT, _AlgorithmType_co],
        extra_tags: Optional[list[str]] = None,
        add_iteration_stopper: bool | None = None,
        trial_name_creator: Optional[Callable[[Trial], str]] = None,
    ):
        if eval_metric is DEFAULT_EVAL_METRIC:
            eval_metric = NEW_LOG_EVAL_METRIC if new_log_format_used() else EVAL_METRIC_RETURN_MEAN
        self.eval_metric: str = eval_metric
        self.eval_metric_order: Literal["max", "min"] = eval_metric_order
        self._setup: SetupType_co = setup
        self.add_iteration_stopper = add_iteration_stopper
        if trial_name_creator is not None:
            self.trial_name_creator = trial_name_creator
        super().__init__(setup=setup, extra_tags=extra_tags)
        self._stopper: Optional[OptunaSearchWithPruner | Stopper | Literal["not_set"]] = "not_set"

    def get_experiment_name(self) -> str:
        return self._setup.project_name

    def create_stoppers(self) -> list[Stopper]:
        """Create stopping criteria for hyperparameter optimization trials.

        This method configures various stopping conditions based on the experiment
        setup configuration. It supports stopping based on total training steps,
        maximum iterations, and custom stopping functions.

        The method automatically detects the setup configuration and adds appropriate
        stoppers to ensure trials terminate when desired conditions are met, helping
        to manage computational resources efficiently.

        Returns:
            List of :class:`ray.tune.stopper.Stopper` instances that will be used
            to determine when trials should be stopped.

        Stoppers Added:
            - **Total Steps Stopper**: When ``args.total_steps`` is specified, stops
              trials that reach the target number of environment steps
            - **Maximum Iteration Stopper**: When ``args.iterations`` is specified,
              stops trials that exceed the maximum number of training iterations
            - **Custom Function Stoppers**: Additional stopping logic based on trial results

        Note:
            The stoppers are combined into a :class:`ray.tune.stopper.CombinedStopper`
            when multiple conditions are present, ensuring trials stop when any
            condition is met.

        See Also:
            :class:`~ray_utilities.tune.stoppers.maximum_iteration_stopper.MaximumResultIterationStopper`: Custom iteration stopper
            :func:`~ray_utilities.training.helpers.get_current_step`: Helper for extracting step counts
        """
        stoppers = []

        # Total Steps Stopper
        added_total_steps_stopper = False
        try:
            # can be int, float, numpy, ...
            total_steps = int(self._setup.args.total_steps)
        except ValueError:
            pass  # e.g, "auto"
        else:
            if total_steps != self._setup.args.total_steps and not isinstance(self._setup.args.total_steps, str):
                logger.warning(
                    "args.total_steps is castable to int but did not match after cast %d != %s (type: %s)",
                    total_steps,
                    self._setup.args.total_steps,
                    type(self._setup.args.total_steps),
                )
            logger.info(
                "Adding FunctionStopper for total steps (tied to setup.args.total_steps) %s",
                total_steps,
            )

            def total_steps_stopper(trial_id: str, results: dict[str, Any] | StrictAlgorithmReturnData) -> bool:  # noqa: ARG001
                current_step = get_current_step(results)  # pyright: ignore[reportArgumentType]
                stop = current_step >= total_steps
                # however will self._setup and trainable._setup still be aligned after a restore?
                if stop:
                    logger.info(
                        "Stopping trial %s at step %s >= total_steps %s",
                        trial_id,
                        current_step,
                        total_steps,
                    )
                return stop

            stoppers.append(FunctionStopper(total_steps_stopper))
            added_total_steps_stopper = True
        if not added_total_steps_stopper:
            logger.debug("Not adding FunctionStopper for total steps %s", self._setup.args.total_steps)

        # Maximum Iteration Stopper
        added_iteration_stopper = False
        if self.add_iteration_stopper is not False:
            try:
                # can be int, float, numpy, ...
                iterations = int(self._setup.args.iterations)
            except ValueError:
                pass  # a string
            else:
                tune_keys = set(self._setup.args.tune or [])
                if (
                    self.add_iteration_stopper
                    or not self._setup.args.tune
                    or not {"iterations", "train_batch_size_per_learner", "batch_size"} & tune_keys
                ):
                    logger.info("Adding MaximumResultIterationStopper with %s iterations", iterations)
                    # Do NOT add this stopper if iterations is adjusted, e.g. by scheduler or the trainable itself
                    stoppers.append(MaximumResultIterationStopper(iterations))
                    added_iteration_stopper = True
        if not added_iteration_stopper:
            logger.debug(
                "Not adding MaximumResultIterationStopper for %s, Tuner.add_iteration_stopper=%s",
                self._setup.args.iterations,
                self.add_iteration_stopper,
            )
        return stoppers

    def create_tune_config(self) -> tune.TuneConfig:
        if getattr(self._setup.args, "resume", False):
            tune.ResumeConfig  # TODO
        stoppers = self.create_stoppers()
        if self._setup.args.optimize_config:
            searcher, optuna_stopper = create_search_algo(
                hparams=self._setup.param_space,
                study_name=self.get_experiment_name(),
                seed=self._setup.args.seed,
                metric=self.eval_metric,
                mode=self.eval_metric_order,
                pruner=self._setup.args.optimize_config,
            )  # TODO: metric
            stoppers.append(optuna_stopper)
        else:
            searcher = None
        if len(stoppers) == 0:
            self._stopper = None
        elif len(stoppers) == 1:
            self._stopper = stoppers[0]
        else:
            self._stopper = CombinedStopper(*stoppers)
        return tune.TuneConfig(
            num_samples=1 if self._setup.args.not_parallel else self._setup.args.num_samples,
            metric=self.eval_metric,
            search_alg=searcher,
            mode=self.eval_metric_order,
            trial_name_creator=self.trial_name_creator,
            max_concurrent_trials=None if self._setup.args.not_parallel else self._setup.args.num_jobs,
            # scheduler=Fifo,
        )

    @overload
    def create_run_config(self, callbacks: list[tune.Callback]) -> tune.RunConfig: ...

    @overload
    def create_run_config(self, callbacks: list[train.UserCallback]) -> train.RunConfig: ...

    def create_run_config(
        self, callbacks: list[tune.Callback] | list[train.UserCallback]
    ) -> tune.RunConfig | train.RunConfig:
        # NOTE: RunConfig V2 is coming up in the future, which will disallow some callbacks
        if TYPE_CHECKING:  # Currently type-checker treats RunConfig as the new version, which is wrong
            callbacks = cast("list[tune.Callback]", callbacks)
        logger.debug("Creating run config with %s callbacks %s", len(callbacks), callbacks or "")
        try:
            RunConfig = tune.RunConfig
            FailureConfig = tune.FailureConfig
            CheckpointConfig = tune.CheckpointConfig
        except AttributeError:  # Use Old API instead
            RunConfig = train.RunConfig
            FailureConfig = train.FailureConfig
            CheckpointConfig = train.CheckpointConfig
        if TYPE_CHECKING:
            from ray.air.config import RunConfig  # noqa: PLC0415, TC004

            FailureConfig = tune.FailureConfig
            CheckpointConfig = tune.CheckpointConfig
        if self._stopper == "not_set":
            if self._setup.args.optimize_config:
                logger.warning(
                    "When using --optimize-config, `create_tune_config` should be called first to set up the stopper."
                )
            stopper = None
        else:
            stopper: OptunaSearchWithPruner | Stopper | None = self._stopper
        if self._setup.args.checkpoint_frequency_unit == "steps" and self._setup.args.checkpoint_frequency is not None:
            checkpoint_frequency = 0  # handle it with custom callback
            if TUNE_RESULT_IS_A_COPY:
                logger.info(
                    "Disabling iteration based checkpointing, using StepCheckpointer instead. "
                    "NOTE: However this will not work with ray <= (all versions) as it passes a copy"
                    " of the result dict."
                )

            else:
                logger.info("Disabling iteration based checkpointing, using StepCheckpointer instead")
            callbacks.append(StepCheckpointer(self._setup.args.checkpoint_frequency))
        else:
            checkpoint_frequency = self._setup.args.checkpoint_frequency
        return RunConfig(
            # Trial artifacts are uploaded periodically to this directory
            storage_path=Path("../outputs/experiments").resolve(),  # pyright: ignore[reportArgumentType]
            name=self.get_experiment_name(),
            log_to_file=False,  # True for hydra like logging to files; or (stoud, stderr.log) files
            # JSON, CSV, and Tensorboard loggers are created automatically by Tune
            # to disable set TUNE_DISABLE_AUTO_CALLBACK_LOGGERS environment variable to "1"
            callbacks=callbacks,  # type: ignore[reportArgumentType] # Ray New Train Interface!
            # Use fail_fast for during debugging/testing to stop all experiments
            failure_config=FailureConfig(
                fail_fast=self._setup.args.test, max_failures=0 if self._setup.args.test else 2
            ),
            checkpoint_config=CheckpointConfig(
                num_to_keep=self._setup.args.num_to_keep,
                checkpoint_score_order="max",
                checkpoint_score_attribute=self.eval_metric,
                checkpoint_frequency=(
                    checkpoint_frequency if self._setup.args.checkpoint_frequency_unit != "steps" else 0
                ),
                # checkpoint_at_end=True,  # will raise error if used with function
            ),
            stop=stopper,
            sync_config=self.create_sync_config(),
        )

    def create_sync_config(self):
        return tune.SyncConfig(sync_artifacts=True)

    @staticmethod
    def _grid_search_to_normal_search_space(
        param_space: dict[str, Any | dict[Literal["grid_search"], Any]] | None = None,
    ) -> dict[str, Any]:
        if param_space is None:
            return {}
        return {
            k: tune.choice(v["grid_search"]) if isinstance(v, dict) and "grid_search" in v else v
            for k, v in param_space.items()
        }

    def create_tuner(self, *, adv_loggers: Optional[bool] = None) -> tune.Tuner:
        """
        Create and return a configured Ray Tune Tuner instance.

        Args:
            adv_loggers: Whether to include advanced variants of the standard CSV, TBX, JSON loggers.
                If ``None``, will be set to ``True`` if :attr:`~DefaultArgumentParser.render_mode` is set in ``args`` of
                the setup.
                Its recommended to use ``True`` when using schedulers working with ``FORK_FROM``.
        """
        resource_requirements = PPO.default_resource_request(self._setup.config)
        resource_requirements = cast(
            "PlacementGroupFactory", resource_requirements
        )  # Resources return value is deprecated
        logger.info("Default resource per trial: %s", resource_requirements.bundles)
        assert self._setup.trainable is not None, "Trainable must be set before creating the tuner"
        trainable = tune.with_resources(self._setup.trainable, resource_requirements)
        # functools.update_wrapper(trainable, self._setup.trainable)
        trainable.__name__ = self._setup.trainable.__name__
        tune_config = self.create_tune_config()
        if isinstance(tune_config.search_alg, OptunaSearch) and any(
            isinstance(v, dict) and "grid_search" in v for v in self._setup.param_space.values()
        ):
            # Cannot use grid_search with OptunaSearch, need to provide a search space without grid_search
            # Grid search must be added as a GridSampler in the search_alg
            param_space = self._grid_search_to_normal_search_space(self._setup.param_space)
        else:
            param_space = self._setup.param_space

        assert FORK_FROM not in param_space, (
            f"{FORK_FROM} is not expected to be in the param_space that is passed to the Tuner by default."
        )
        return tune.Tuner(
            trainable=trainable,  # Updated to use the modified trainable with resource requirements
            param_space=param_space,  # TODO: Likely Remove when using space of OptunaSearch
            tune_config=tune_config,
            run_config=self.create_run_config(self.create_callbacks(adv_loggers=adv_loggers)),
        )
