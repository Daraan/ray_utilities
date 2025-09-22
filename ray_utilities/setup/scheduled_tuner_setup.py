from abc import abstractmethod

from ray import tune
from ray.tune import schedulers

from ray_utilities.constants import EVAL_METRIC_RETURN_MEAN
from ray_utilities.setup.ppo_mlp_setup import PPOMLPSetup
from ray_utilities.setup.tuner_setup import SetupType_co, TunerSetup
from ray_utilities.tune.scheduler.re_tune_scheduler import ReTuneScheduler


class ScheduledTunerSetup(TunerSetup[SetupType_co]):
    @abstractmethod
    def create_scheduler(self) -> schedulers.TrialScheduler: ...

    def create_tune_config(self) -> tune.TuneConfig:
        tune_config = super().create_tune_config()
        tune_config.scheduler = self.create_scheduler()
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
        return ReTunerSetup(
            setup=self, eval_metric=EVAL_METRIC_RETURN_MEAN, eval_metric_order=self.args.mode
        ).create_tuner()


# endregion

# region PBT


class PBTTunerSetup(ScheduledTunerSetup):
    def create_scheduler(self) -> schedulers.TrialScheduler:
        return schedulers.PopulationBasedTraining(
            perturbation_interval=2048 * 3,
            hyperparam_mutations={
                "train_batch_size_per_learner": [256, 512, 1024, 2048],
                "sgd_minibatch_size": [64, 128, 256, 512],
                "lr": [1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
                "clip_param": [0.1, 0.2, 0.3, 0.4],
                "vf_clip_param": [10.0, 20.0, 30.0, 40.0],
            },
            mode=None,  # filled in by Tuner
            metric=None,  # filled in by Tuner
        )
