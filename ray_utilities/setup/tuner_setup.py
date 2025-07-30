from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol, cast, overload

from ray import train, tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import CombinedStopper, FunctionStopper
from typing_extensions import TypeVar

from ray_utilities.config._tuner_callbacks_setup import TunerCallbackSetup
from ray_utilities.constants import (
    CLI_REPORTER_PARAMETER_COLUMNS,
    EVAL_METRIC_RETURN_MEAN,
)
from ray_utilities.misc import trial_name_creator
from ray_utilities.setup.optuna_setup import OptunaSearchWithPruner, create_search_algo
from ray_utilities.training.helpers import get_current_step
from ray_utilities.tune.stoppers.maximum_iteration_stopper import MaximumResultIterationStopper

if TYPE_CHECKING:
    from ray.air.config import RunConfig as RunConfigV1
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.tune.execution.placement_groups import PlacementGroupFactory
    from ray.tune.experiment import Trial
    from ray.tune.stopper import Stopper

    from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase
    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData


__all__ = [
    "TunerSetup",
]


ConfigTypeT = TypeVar("ConfigTypeT", bound="AlgorithmConfig")
ParserTypeT = TypeVar("ParserTypeT", bound="DefaultArgumentParser")
_AlgorithmType_co = TypeVar("_AlgorithmType_co", bound="Algorithm", covariant=True)

logger = logging.getLogger(__name__)


class _TunerSetupBase(Protocol):
    eval_metric: str
    eval_metric_order: Literal["max", "min"]
    trial_name_creator: Callable[[Trial], str]

    def create_tune_config(self) -> tune.TuneConfig: ...

    def create_run_config(
        self, callbacks: list[tune.Callback] | list[train.UserCallback]
    ) -> tune.RunConfig | RunConfigV1 | train.RunConfig: ...

    def create_tuner(self) -> tune.Tuner: ...


class TunerSetup(TunerCallbackSetup, _TunerSetupBase):
    trial_name_creator = trial_name_creator

    def __init__(
        self,
        eval_metric: str = EVAL_METRIC_RETURN_MEAN,
        eval_metric_order: Literal["max", "min"] = "max",
        *,
        setup: ExperimentSetupBase[ParserTypeT, ConfigTypeT, _AlgorithmType_co],
        extra_tags: Optional[list[str]] = None,
    ):
        self.eval_metric: str = eval_metric
        self.eval_metric_order: Literal["max", "min"] = eval_metric_order
        self._setup = setup
        super().__init__(setup=setup, extra_tags=extra_tags)
        self._stopper: Optional[OptunaSearchWithPruner | Stopper | Literal["not_set"]] = "not_set"

    def get_experiment_name(self) -> str:
        return self._setup.project_name

    def create_stoppers(self) -> list[Stopper]:
        """
        Create a stopper for the tuner based on the setup configuration.
        If `optimize_config` is enabled, it uses OptunaSearchWithPruner.
        Otherwise, it returns None or an empty list.
        """
        stoppers = []
        if isinstance(self._setup.args.total_steps, (int, float)):
            logger.debug(
                "Adding FunctionStopper for total steps (tied to setup.args.total_steps) %s",
                self._setup.args.total_steps,
            )

            def total_steps_stopper(trial_id: str, results: dict[str, Any] | StrictAlgorithmReturnData) -> bool:  # noqa: ARG001
                current_step = get_current_step(results)  # pyright: ignore[reportArgumentType]
                stop = current_step >= self._setup.args.total_steps  # <-- this allows late modification
                # however will self._setup and trainable._setup still be aligned after a restore?
                if stop:
                    logger.info(
                        "Stopping trial %s at step %s >= total_steps %s",
                        trial_id,
                        current_step,
                        self._setup.args.total_steps,
                    )
                return stop

            stoppers.append(FunctionStopper(total_steps_stopper))
        if isinstance(self._setup.args.iterations, (int, float)):
            logger.debug("Adding MaximumResultIterationStopper with %s iterations", self._setup.args.iterations)
            stoppers.append(MaximumResultIterationStopper(int(self._setup.args.iterations)))
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
            trial_name_creator=trial_name_creator,
            max_concurrent_trials=None if self._setup.args.not_parallel else self._setup.args.num_jobs,
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
        logger.info("Creating run config with %s callbacks", len(callbacks))
        try:
            RunConfig = tune.RunConfig
            FailureConfig = tune.FailureConfig
            CheckpointConfig = tune.CheckpointConfig
        except AttributeError:
            RunConfig = train.RunConfig
            FailureConfig = train.FailureConfig
            CheckpointConfig = train.CheckpointConfig
        if TYPE_CHECKING:
            from ray.air.config import RunConfig  # noqa: TC004

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
        return RunConfig(
            # Trial artifacts are uploaded periodically to this directory
            storage_path=Path("../outputs/experiments").resolve(),  # pyright: ignore[reportArgumentType]
            name=self.get_experiment_name(),
            log_to_file=False,  # True for hydra like logging to files; or (stoud, stderr.log) files
            # JSON, CSV, and Tensorboard loggers are created automatically by Tune
            # to disable set TUNE_DISABLE_AUTO_CALLBACK_LOGGERS environment variable to "1"
            callbacks=callbacks,  # type: ignore[reportArgumentType] # Ray New Train Interface!
            # Use fail_fast for during debugging/testing to stop all experiments
            failure_config=FailureConfig(fail_fast=True),
            checkpoint_config=CheckpointConfig(
                num_to_keep=4,
                checkpoint_score_order="max",
                checkpoint_score_attribute=self.eval_metric,
                checkpoint_frequency=0,  # No automatic checkpointing
                # checkpoint_at_end=True,  # will raise error if used with function
            ),
            stop=stopper,
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

    def create_tuner(self) -> tune.Tuner:
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

        return tune.Tuner(
            trainable=trainable,  # Updated to use the modified trainable with resource requirements
            param_space=param_space,  # TODO: Likely Remove when using space of OptunaSearch
            tune_config=tune_config,
            run_config=self.create_run_config(self.create_callbacks()),
        )
