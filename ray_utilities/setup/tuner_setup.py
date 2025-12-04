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
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Optional, Protocol, cast, overload

from ray import train, tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.sample import Categorical, Domain
from ray.tune.stopper import CombinedStopper, FunctionStopper
from typing_extensions import TypeVar

from ray_utilities.callbacks.tuner.add_experiment_key import AddExperimentKeyCallback
from ray_utilities.callbacks.tuner.metric_checkpointer import StepCheckpointer
from ray_utilities.config._tuner_callbacks_setup import TunerCallbackSetup
from ray_utilities.constants import (
    DEFAULT_EVAL_METRIC,
    FORK_FROM,
    TUNE_RESULT_IS_A_COPY,
    get_run_id,
)
from ray_utilities.misc import calc_env_size, get_current_step, resolve_default_eval_metric
from ray_utilities.misc import trial_name_creator as default_trial_name_creator
from ray_utilities.nice_logger import ImportantLogger
from ray_utilities.tune.experiments import set_experiment_key_on_trial
from ray_utilities.tune.scheduler.add_experiment_keys_mixin import AddExperimentKeysMixin
from ray_utilities.tune.searcher.constrained_minibatch_search import constrained_minibatch_search
from ray_utilities.tune.searcher.optuna_searcher import OptunaSearchWithPruner, create_optuna_searcher
from ray_utilities.tune.stoppers.maximum_iteration_stopper import MaximumResultIterationStopper

if TYPE_CHECKING:
    from ray.air.config import RunConfig as RunConfigV1
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.tune.experiment import Trial
    from ray.tune.stopper import Stopper

    from ray_utilities.config import DefaultArgumentParser
    from ray_utilities.config.parser.pbt_scheduler_parser import PopulationBasedTrainingParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase
    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData


__all__ = [
    "TunerSetup",
]

SetupType_co = TypeVar(
    "SetupType_co",
    bound="ExperimentSetupBase[DefaultArgumentParser[PopulationBasedTrainingParser] | DefaultArgumentParser[None], AlgorithmConfig, Algorithm]",  # noqa: E501
    covariant=True,
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
        eval_metric: str | DEFAULT_EVAL_METRIC = "DEFAULT_EVAL_METRIC",
        eval_metric_order: Literal["max", "min"] = "max",
        *,
        setup: SetupType_co,  # ExperimentSetupBase[ParserTypeT, ConfigTypeT, _AlgorithmType_co],
        extra_tags: Optional[list[str]] = None,
        add_iteration_stopper: bool | None = None,
        trial_name_creator: Optional[Callable[[Trial], str]] = None,
    ):
        if eval_metric is DEFAULT_EVAL_METRIC or eval_metric == "DEFAULT_EVAL_METRIC":
            eval_metric = resolve_default_eval_metric(for_logger=False, use_ema=not setup.args.no_eval_ema)
        self.eval_metric: str = eval_metric
        self.eval_metric_order: Literal["max", "min"] = eval_metric_order
        self._setup: SetupType_co = setup
        self.add_iteration_stopper = add_iteration_stopper
        if trial_name_creator is not None:
            self.trial_name_creator = trial_name_creator
        super().__init__(setup=setup, extra_tags=extra_tags)
        self._stopper: Optional[OptunaSearchWithPruner | Stopper | Literal["not_set"]] = "not_set"

    def get_experiment_name(self) -> str:
        """
        Get the experiment name for organizing tuning results.
        This will be the subdir or the storage_path the tuner uses.
        """
        algorithm = self._setup.args.algorithm
        if algorithm == "default":
            try:
                _, algo_class = self._setup.get_algorithm_classes(self._setup.args)
            except Exception:
                logger.exception("Error getting algorithm class for default naming")
            else:
                algorithm = algo_class.__name__ if algo_class is not None else "default"
        return f"{self._setup.project}-{algorithm.upper()}-{get_run_id()}"

    def get_storage_path(self) -> str:
        """
        Get the storage path for organizing tuning results.
        This will be the base directory where the tuner saves outputs.
        Can also point to an S3 bucket.
        """
        if str(self._setup.storage_path).startswith("s3://"):
            return str(self._setup.storage_path)
        return str(Path(self._setup.storage_path).absolute())

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

            def total_steps_stopper(
                trial_id: str, results: dict[str, Any] | StrictAlgorithmReturnData, total_steps=total_steps
            ) -> bool:  # noqa: ARG001
                current_step = get_current_step(results)  # pyright: ignore[reportArgumentType]
                stop = current_step >= total_steps
                # however will self._setup and trainable._setup still be aligned after a restore?
                if stop:
                    ImportantLogger.important_info(
                        logger,
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
                    ImportantLogger.important_info(
                        logger, "Adding MaximumResultIterationStopper with %s iterations", iterations
                    )
                    # Do NOT add this stopper if iterations is adjusted, e.g. by scheduler or the trainable itself
                    stoppers.append(MaximumResultIterationStopper(iterations))
                    added_iteration_stopper = True
        if not added_iteration_stopper:
            logger.info(
                "Not adding MaximumResultIterationStopper for %s, Tuner.add_iteration_stopper=%s",
                self._setup.args.iterations,
                self.add_iteration_stopper,
            )
        return stoppers

    def create_searcher(self, stoppers: list[Stopper]):
        """
        Create a ``Searcher`` or ``SearchAlgorithm`` for hyperparameter optimization. Return None to use default.

        When the minibatch_size is in the parameter space
        """
        # If we only use grid_search and make no use of optuna's TPE sampler and we want pruning,
        # just use rays MedianStoppingRule which supports custom time metrics.
        use_ray_median_scheduler_instead = all(
            not isinstance(v, Domain) or (isinstance(v, Categorical) and (len(v.categories) == 1 or v.is_grid()))
            for v in self._setup.param_space.values()
        )
        if not use_ray_median_scheduler_instead and (
            self._setup.args.tune or self._setup.args.optimize_config
        ):  # legacy argument
            # TODO: Using Bayesian Hyperband BOHB is a better alternative (can be used without scheduler)
            # TODO: Can still use OptunaSearcher without pruner
            # NOTE: Optuna's Pruner prune by training_iteration and not current step.
            ImportantLogger.important_info(
                logger,
                "Using OptunaSearch as search space is more complex. %s",
                (
                    "Using Optunas Pruner."
                    if not use_ray_median_scheduler_instead
                    else "Not using Optunas Pruner, using Ray's MedianStoppingRule instead."
                ),
            )
            searcher, optuna_stopper = create_optuna_searcher(
                hparams=self._setup.param_space,
                study_name=self.get_experiment_name(),
                seed=self._setup.args.seed,
                metric=self.eval_metric,
                mode=self.eval_metric_order,
                # NOTE: Use pruner only search spaces with random sampling.
                # NOTE: Optunas Pruner does only support training_iteration based pruning.
                pruner=not use_ray_median_scheduler_instead,
                pruner_warmup_steps=self._setup.args.pruner_warmup_steps,
                pruner_min_trials=self._setup.args.pruner_min_trials,
            )  # TODO: metric
            if optuna_stopper is not None:
                stoppers.append(optuna_stopper)
        elif "minibatch_size" in self._setup.param_space and not isinstance(
            self._setup.param_space["minibatch_size"], (int, float)
        ):
            searcher = constrained_minibatch_search(self._setup)
        else:
            searcher = None
        return searcher

    def create_tune_config(self) -> tune.TuneConfig:
        if getattr(self._setup.args, "resume", False):
            tune.ResumeConfig  # TODO
        stoppers = self.create_stoppers()
        searcher = self.create_searcher(stoppers)
        use_ray_median_scheduler_instead = all(
            not isinstance(v, Domain) or (isinstance(v, Categorical) and (len(v.categories) == 1 or v.is_grid()))
            for v in self._setup.param_space.values()
        )
        scheduler = None
        # Check if PBT scheduler is configured
        if self._setup.args.command is not None:
            # Use the scheduler from the PBT parser (TopPBT or GroupedTopPBT)
            try:
                scheduler = self._setup.args.command.to_scheduler()
            except AttributeError as e:
                if "to_scheduler" in str(e):
                    scheduler = None
                else:
                    raise
            else:
                logger.info("Using PBT scheduler from parser: %s", type(scheduler).__name__)
        if scheduler is None and use_ray_median_scheduler_instead:
            ImportantLogger.important_info(
                logger, "Using MedianStoppingRule as scheduler space does not require sampling"
            )
            scheduler = MedianStoppingRule(
                time_attr="current_step",
                grace_period=8192 * 16,
                # Trials are stopped to reach the same point in time - slow trials can come up
                # This metrics controls for how much longer a trial is allowed to train before starting
                # pausing it and starting a pending one.
                min_time_slice=8192 * 8,
                min_samples_required=int(min(8, max(3, 0.15 * self._setup.args.num_samples))),
                # Instead of stopping bad trials, pause them and resume them at the very end
                hard_stop=True,
            )
            # patch to add experiment key
            orig_on_trial_add = scheduler.on_trial_add

            def on_trial_add_with_experiment_key(tune_controller, trial: Trial, *args, **kwargs) -> None:
                set_experiment_key_on_trial(trial)
                return orig_on_trial_add(*args, tune_controller=tune_controller, trial=trial, **kwargs)

            scheduler.on_trial_add = on_trial_add_with_experiment_key
        # if scheduler is none use FIFO as default
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
            scheduler=scheduler,
            reuse_actors=False,  # self._setup.args.test,  # EXPERIMENTAL!
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
        # We do not checkpoint when we do PBT
        if self._setup.args.command_str and "pbt" not in self._setup.args.command_str.lower():
            if (
                self._setup.args.checkpoint_frequency_unit == "steps"
                and self._setup.args.checkpoint_frequency is not None
            ):
                checkpoint_frequency = 0  # handle it with custom callback
                if TUNE_RESULT_IS_A_COPY:
                    logger.info(
                        "Disabling iteration based checkpointing, using StepCheckpointer instead. "
                        "NOTE: However this will not work with ray <= (all versions) as it passes a copy"
                        " of the result dict."
                    )

                else:
                    logger.info("Disabling iteration based checkpointing, using StepCheckpointer instead")
                callbacks.append(StepCheckpointer(self._setup.args.checkpoint_frequency, min_iterations=24))
            else:
                checkpoint_frequency = self._setup.args.checkpoint_frequency
        else:
            checkpoint_frequency = 0
            logger.info("Disabling checkpointing by Tuner, handled by PBT scheduler.")
        return RunConfig(
            # Trial artifacts are uploaded periodically to this directory
            name=self.get_experiment_name(),
            storage_path=str(self.get_storage_path()),
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
        return tune.SyncConfig(
            sync_artifacts=True,
            sync_period=600,
        )

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

    def create_tuner(self, *, adv_loggers: Optional[bool] = True) -> tune.Tuner:
        """
        Create and return a configured Ray Tune Tuner instance.

        By default the tuner will reserve 3 GB + 0.5 GB per environment runner.
        To account for memory usage during training, e.g. 1-2 GB should be reserved for the WandbLogger

        Args:
            adv_loggers: Whether to include advanced variants of the standard CSV, TBX, JSON loggers.
                If ``None``, will be set to ``True`` if :attr:`~DefaultArgumentParser.render_mode` is set in ``args`` of
                the setup.
                Its recommended to use ``True`` when using schedulers working with ``FORK_FROM``.

        Returns:
            A configured :class:`tune.Tuner` instance, either restored from a previous experiment
            if :attr:`args.restore_path<DefaultArgumentParser.restore_path>` is set,
            or a new tuner instance.

        Raises:
            ValueError: If ``restore_path`` is set but the path holds not tuner.pkl file.
        """
        # Prepare resource requirements (used for both new and restored tuners)
        # NOTE: "memory" is not allowed in custom_resources_per_env_runner when building a config
        GB = 1024 * 1024 * 1024
        custom_resources_per_env_runner = {
            "memory": int(
                (
                    (
                        max(0.5, (self._setup.config.num_envs_per_env_runner or 8) * 0.075)
                        # NOTE: config.num_envs_per_env_runner does work when argument is added later by tune.
                        # Use a mean scale instead
                        if self._setup.args.tune and "num_envs_per_env_runner" not in self._setup.args.tune
                        else 0.8
                    )
                    * (1.0 + calc_env_size(self._setup.config.env) / 1000)
                )
                * GB
            )
        }
        config_class = type(self._setup.config)
        resource_requirements = (config_class().algo_class or PPO).default_resource_request(
            self._setup.config.copy(copy_frozen=False).update_from_dict(
                config_class.overrides(custom_resources_per_env_runner=custom_resources_per_env_runner)
            )
        )
        resource_requirements = cast(
            "PlacementGroupFactory", resource_requirements
        )  # Resources return value is deprecated
        bundles = resource_requirements.bundles
        # When tracking memory calculate RES - SHR
        # https://docs.ray.io/en/latest/ray-core/scheduling/memory-management.html
        bundles[0]["memory"] = int(
            (bundles[0].get("memory", 1.75 * GB))  # base memory
            # Scale also with batch_size, however when tuned argument is not available here
            * (1.0 + self._setup.config.train_batch_size_per_learner / 2024 // 10)
            # So when we tune it increase by a flat amount, mean need is about * 1.2
            * (1.25 if self._setup.args.tune and "batch_size" in self._setup.args.tune else 1.0)
        )
        # If we are restoring operations could be more expensive
        # add more memory to be safe as checkpoint / state can be large
        if self._setup.__restored__:
            bundles[0]["memory"] += int(2 * GB)
        if len(bundles) == 1:
            # No env runners, reserve some extra memory as default need. But has less overhead
            bundles[0]["memory"] = int(
                bundles[0].get("memory", 0) + custom_resources_per_env_runner.get("memory", 0.5 * GB) * 0.75
            )
        if self._setup.args.node_id_selector or self._setup.args.hostname_selector:
            hostname_label = (
                {"hostname": self._setup.args.hostname_selector} if self._setup.args.hostname_selector else {}
            )
            node_id_label = (
                {"ray.io/node-id": self._setup.args.node_id_selector} if self._setup.args.node_id_selector else {}
            )
            bundle_label_selector = [hostname_label | node_id_label for _ in bundles]
        else:
            bundle_label_selector = None
        resource_requirements = PlacementGroupFactory(
            bundles, resource_requirements.strategy, bundle_label_selector=bundle_label_selector
        )
        logger.info("Default resource per trial: %s", resource_requirements.bundles)

        assert self._setup.trainable is not None, "Trainable must be set before creating the tuner"
        trainable = tune.with_resources(self._setup.trainable, resource_requirements)
        # functools.update_wrapper(trainable, self._setup.trainable)
        trainable.__name__ = self._setup.trainable.__name__

        # Check if we should restore from a previous experiment
        if self._setup.args.restore_path:
            restore_path = self._setup.args.restore_path
            logger.info("Restoring Tuner from path: %s", restore_path)

            # Verify the path can be restored before attempting
            if not tune.Tuner.can_restore(restore_path):
                raise ValueError(
                    f"Cannot restore Tuner from path: {restore_path}. "
                    "The path does not contain a valid Ray Tune experiment. "
                    "Please check that the path exists and contains experiment metadata."
                )

            # Use Tuner.restore() with the trainable
            return tune.Tuner.restore(
                path=restore_path,
                trainable=trainable,
                resume_errored=self._setup.args.restore_errored == "resume",
                restart_errored=self._setup.args.restore_errored == "restart",
                # Possibly add param_space if it contained object references, todo if we add the cli_args not as a dict
            )

        # Create a new tuner
        tune_config = self.create_tune_config()
        if isinstance(tune_config.search_alg, OptunaSearch) and any(
            isinstance(v, dict) and "grid_search" in v for v in self._setup.param_space.values()
        ):
            # Cannot use grid_search with OptunaSearch, need to provide a search space without grid_search
            # Grid search must be added as a GridSampler in the search_alg
            param_space = self._grid_search_to_normal_search_space(self._setup.param_space)
        else:
            param_space = self._setup.param_space
        # NOTE: This allows for late setup of group / project names
        if param_space.get("group", None) is None:
            param_space["group"] = self._setup.group_name
        if param_space.get("project") in ("Unnamed Project", None):
            param_space["project"] = self._setup.project
        if "minibatch_size" in param_space:
            if "train_batch_size_per_learner" not in param_space:
                # Let searcher know which train_batch_size is used. Alternatively could set this info on the searcher
                # Problem / side-effect. This puts train_batch_size_per_learner into the hparameter space
                param_space["train_batch_size_per_learner"] = tune.choice(
                    [self._setup.args.train_batch_size_per_learner]
                )

        assert FORK_FROM not in param_space, (
            f"{FORK_FROM} is not expected to be in the param_space that is passed to the Tuner by default."
        )
        run_config = self.create_run_config(self.create_callbacks(adv_loggers=adv_loggers))
        if "_ray_pkg_" in str(run_config.storage_path):
            raise ValueError(
                "Storage path points to a _ray_pkg_ directory, which is an UNWRITABLE and temporary working dir. "
                "Check storage_path / RAY_UTILITIES_STORAGE_PATH setup.\n"
                f"Storage path: {run_config.storage_path} "
                f"RAY_UTILITIES_STORAGE_PATH: {os.environ.get('RAY_UTILITIES_STORAGE_PATH')}"
            )
        logger.info(
            "Creating new Tuner with experiment name: %s at storage path: %s",
            run_config.name,
            run_config.storage_path,
        )
        if isinstance(tune_config.scheduler, AddExperimentKeysMixin) and run_config.callbacks:
            # remove AddExperimentKeyCallback
            run_config.callbacks = [cb for cb in run_config.callbacks if not isinstance(cb, AddExperimentKeyCallback)]
        return tune.Tuner(
            trainable=trainable,  # Updated to use the modified trainable with resource requirements
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
