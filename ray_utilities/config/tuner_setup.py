from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Literal, Optional, Protocol

from ray import train, tune
from typing_extensions import TypeVar

from ray_utilities import trial_name_creator
from ray_utilities.config._tuner_callbacks_setup import TunerCallbackSetup
from ray_utilities.constants import DISC_EVAL_METRIC_RETURN_MEAN

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig
    from ray.tune.experiment import Trial

    from ray_utilities.config._typed_argument_parser import DefaultArgumentParser
    from ray_utilities.config.experiment_base import ExperimentSetupBase

__all__ = [
    "TunerSetup",
]


ConfigType = TypeVar("ConfigType", bound="AlgorithmConfig")
ParserType = TypeVar("ParserType", bound="DefaultArgumentParser", default="DefaultArgumentParser")


class _TunerSetupBase(Protocol):
    eval_metric: ClassVar[str]
    eval_metric_order: ClassVar[Literal["max", "min"]]
    trial_name_creator: Callable[[Trial], str]

    def create_tune_config(self) -> tune.TuneConfig: ...

    def create_run_config(self, callbacks: list[tune.Callback]) -> train.RunConfig: ...

    def create_tuner(self) -> tune.Tuner: ...


class TunerSetup(TunerCallbackSetup, _TunerSetupBase):
    eval_metric = DISC_EVAL_METRIC_RETURN_MEAN
    eval_metric_order = "max"
    trial_name_creator = trial_name_creator

    def __init__(
        self,
        *,
        setup: ExperimentSetupBase[ConfigType, ParserType],
        extra_tags: Optional[list[str]] = None,
    ):
        self._setup = setup
        super().__init__(setup=setup, extra_tags=extra_tags)

    def get_experiment_name(self) -> str:
        return self._setup.project_name

    def create_tune_config(self) -> tune.TuneConfig:
        return tune.TuneConfig(
            num_samples=1 if self._setup.args.not_parallel else self._setup.args.num_jobs,
            # metric=
            #    (EVALUATION_RESULTS + "/" + ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN
            #     if config.evaluation_interval else ENV_RUNNER_RESULTS + "/" + EPISODE_RETURN_MEAN),
            mode="max",
            trial_name_creator=trial_name_creator,
        )

    def create_run_config(self, callbacks: list[tune.Callback]) -> train.RunConfig:
        return train.RunConfig(
            # Trial artifacts are uploaded periodically to this directory
            storage_path=Path("../outputs/experiments").resolve(),  # pyright: ignore[reportArgumentType]
            name=self.get_experiment_name(),
            log_to_file=False,  # True for hydra like logging to files; or (stoud, stderr.log) files
            progress_reporter=tune.CLIReporter(mode="max", max_report_frequency=45),
            # JSON, CSV, and Tensorboard loggers are created automatically by Tune
            # to disable set TUNE_DISABLE_AUTO_CALLBACK_LOGGERS environment variable to "1"
            callbacks=callbacks,
            # Use fail_fast for during debugging/testing to stop all experiments
            failure_config=train.FailureConfig(fail_fast=True),
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=4,
                checkpoint_score_order="max",
                checkpoint_score_attribute=self.eval_metric,
            ),
        )

    def create_tuner(self) -> tune.Tuner:
        return tune.Tuner(
            trainable=self._setup.trainable,  # Note: possibly can also be a list
            param_space=self._setup.param_space,
            tune_config=self.create_tune_config(),
            run_config=self.create_run_config(self.create_callbacks()),
        )
