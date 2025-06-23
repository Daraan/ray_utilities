from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Literal, Optional, Protocol, cast, overload

from ray import train, tune
from ray.rllib.algorithms.ppo import PPO
from typing_extensions import TypeVar

from ray_utilities.config._tuner_callbacks_setup import TunerCallbackSetup
from ray_utilities.constants import (
    CLI_REPORTER_PARAMETER_COLUMNS,
    DISC_EVAL_METRIC_RETURN_MEAN,
    EVAL_METRIC_RETURN_MEAN,
)
from ray_utilities.misc import trial_name_creator
from ray_utilities.setup.optuna_setup import create_search_algo

if TYPE_CHECKING:
    from ray.air.config import RunConfig as RunConfigV1
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
    from ray.tune.experiment import Trial

    from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
    from ray_utilities.setup.experiment_base import ExperimentSetupBase

__all__ = [
    "TunerSetup",
]


ConfigTypeT = TypeVar("ConfigTypeT", bound="AlgorithmConfig")
ParserTypeT = TypeVar("ParserTypeT", bound="DefaultArgumentParser")
_AlgorithmType_co = TypeVar("_AlgorithmType_co", bound="Algorithm", covariant=True)

logger = logging.getLogger(__name__)


class _TunerSetupBase(Protocol):
    eval_metric: ClassVar[str]
    eval_metric_order: ClassVar[Literal["max", "min"]]
    trial_name_creator: Callable[[Trial], str]

    def create_tune_config(self) -> tune.TuneConfig: ...

    def create_run_config(
        self, callbacks: list[tune.Callback] | list[train.UserCallback]
    ) -> tune.RunConfig | RunConfigV1 | train.RunConfig: ...

    def create_tuner(self) -> tune.Tuner: ...


class TunerSetup(TunerCallbackSetup, _TunerSetupBase):
    eval_metric = DISC_EVAL_METRIC_RETURN_MEAN
    eval_metric_order = "max"
    trial_name_creator = trial_name_creator

    def __init__(
        self,
        *,
        setup: ExperimentSetupBase[ParserTypeT, ConfigTypeT, _AlgorithmType_co],
        extra_tags: Optional[list[str]] = None,
    ):
        self._setup = setup
        super().__init__(setup=setup, extra_tags=extra_tags)

    def get_experiment_name(self) -> str:
        return self._setup.project_name

    def create_tune_config(self) -> tune.TuneConfig:
        if getattr(self._setup.args, "resume", False):
            tune.ResumeConfig  # TODO
        return tune.TuneConfig(
            num_samples=1 if self._setup.args.not_parallel else self._setup.args.num_samples,
            metric=EVAL_METRIC_RETURN_MEAN,
            search_alg=(
                create_search_algo(
                    hparams=self._setup.param_space,
                    study_name=self.get_experiment_name(),
                    seed=self._setup.args.seed,
                    metric=EVAL_METRIC_RETURN_MEAN,
                    mode="max",
                )  # TODO: metric
                if self._setup.args.optimize_config
                else None
            ),
            mode="max",
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
            ),
        )

    def create_tuner(self) -> tune.Tuner:
        resource_requirements = PPO.default_resource_request(self._setup.config)
        logger.info("Default resource per trial: %s", resource_requirements.bundles)
        trainable = tune.with_resources(self._setup.trainable, resource_requirements)
        # functools.update_wrapper(trainable, self._setup.trainable)
        trainable.__name__ = self._setup.trainable.__name__
        return tune.Tuner(
            trainable=trainable,  # Updated to use the modified trainable with resource requirements
            param_space=self._setup.param_space,
            tune_config=self.create_tune_config(),
            run_config=self.create_run_config(self.create_callbacks()),
        )
