from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup
from ray_utilities.setup.tuner_setup import SetupType_co, TunerSetup
from ray_utilities.tune.scheduler.re_tune_scheduler import ReTuneScheduler

if TYPE_CHECKING:
    from ray import tune
    from ray.tune import schedulers

    from ray_utilities.config.parser.mlp_argument_parser import MLPArgumentParser

logger = logging.getLogger(__name__)

__all__ = [
    "PBTTunerSetup",
    "PPOMLPWithPBTSetup",
    "PPOMLPWithReTuneSetup",
    "ReTunerSetup",
    "ScheduledTunerSetup",
]


class ScheduledTunerSetup(TunerSetup[SetupType_co]):
    @abstractmethod
    def create_scheduler(self) -> schedulers.TrialScheduler: ...

    def create_tune_config(self) -> tune.TuneConfig:
        tune_config = super().create_tune_config()
        tune_config.scheduler = self.create_scheduler()
        if tune_config.search_alg is not None:
            logger.info(
                "Unsetting search algorithm %s to not conflict with the scheduler %s",
                tune_config.search_alg,
                tune_config.scheduler,
            )
            tune_config.search_alg = None
        return tune_config


# region ReTuner


class ReTunerSetup(ScheduledTunerSetup[SetupType_co]):
    def create_scheduler(self) -> schedulers.TrialScheduler:
        return ReTuneScheduler(
            perturbation_interval=self._setup.args.perturbation_interval,
            resample_probability=self._setup.args.resample_probability,
            hyperparam_mutations={
                "train_batch_size_per_learner": {"grid_search": [256, 512, 1024, 2048]}
            },  # TODO: experimental
            mode=None,  # filled in by Tuner
            metric=None,  # filled in by Tuner
            synch=True,
        )


class PPOMLPWithReTuneSetup(PPOMLPSetup):
    def create_tuner(self) -> tune.Tuner:
        return ReTunerSetup(setup=self, eval_metric=self.args.metric, eval_metric_order=self.args.mode).create_tuner()


# endregion

# region PBT


class PBTTunerSetup(ScheduledTunerSetup["PPOMLPWithPBTSetup"]):
    def create_scheduler(self) -> schedulers.TrialScheduler:
        return self._setup.args.to_scheduler()


class PPOMLPWithPBTSetup(PPOMLPSetup["MLPArgumentParser"]):
    def create_tuner(self) -> tune.Tuner:
        return PBTTunerSetup(setup=self, eval_metric=self.args.metric, eval_metric_order=self.args.mode).create_tuner()


# endregion
