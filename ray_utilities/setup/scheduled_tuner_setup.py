from __future__ import annotations

import logging
import os
from abc import abstractmethod
from typing import TYPE_CHECKING

from ray.tune.search.searcher import Searcher
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, Stopper

from ray_utilities.callbacks.tuner.sync_config_files_callback import SyncConfigFilesCallback
from ray_utilities.nice_logger import ImportantLogger
from ray_utilities.setup.ppo_mlp_setup import MLPSetup, PPOMLPSetup
from ray_utilities.setup.tuner_setup import SetupType_co, TunerSetup
from ray_utilities.tune.scheduler.re_tune_scheduler import ReTuneScheduler
from ray_utilities.tune.searcher.constrained_minibatch_search import constrained_minibatch_search
from ray_utilities.tune.searcher.optuna_searcher import OptunaSearchWithPruner

if TYPE_CHECKING:
    from ray import tune
    from ray.tune import Callback, schedulers

    from ray_utilities.config.parser.mlp_argument_parser import MLPArgumentParser
    from ray_utilities.config.parser.pbt_scheduler_parser import PopulationBasedTrainingParser


logger = logging.getLogger(__name__)

__all__ = [
    "MLPPBTSetup",
    "PBTTunerSetup",
    "PPOMLPWithReTuneSetup",
    "ReTunerSetup",
    "ScheduledTunerSetup",
]


class ScheduledTunerSetup(TunerSetup[SetupType_co]):
    @abstractmethod
    def create_scheduler(self) -> schedulers.TrialScheduler: ...

    def create_tune_config(self) -> tune.TuneConfig:
        tune_config = super().create_tune_config()
        if tune_config.scheduler is not None:
            ImportantLogger.important_info(
                logger,
                "Overwriting existing scheduler %s for PBT with %s",
                tune_config.scheduler,
                self.create_scheduler(),
            )
        tune_config.scheduler = self.create_scheduler()
        # Searcher are turned into SearchGenerators which are not OK, SearchAlgorithms are OK
        if isinstance(tune_config.search_alg, Searcher):
            logger.info(
                "Unsetting search algorithm %s to not conflict with the scheduler %s",
                tune_config.search_alg,
                tune_config.scheduler,
            )
            tune_config.search_alg = None
        return tune_config

    # NOTE: MaximumIterationStopper is not necessary here depending on the TunerSetup.
    # Our create_stoppers is smart to not add it if the batch_size is perturbed.
    stoppers_to_remove: tuple[type[tune.Stopper], ...] = (OptunaSearchWithPruner, MaximumIterationStopper)
    """Stoppers classes to remove when using this scheduler."""

    def create_run_config(self, callbacks: list[tune.Callback]) -> tune.RunConfig:  # pyright: ignore[reportIncompatibleMethodOverride]
        run_config = super().create_run_config(callbacks=callbacks)
        # Should also remove OptunaPruner as stopper which is placed in run_config
        if run_config.stop is not None:
            if isinstance(run_config.stop, self.stoppers_to_remove):
                logger.info(
                    "Removing stopper %s to not conflict with the scheduler %s", run_config.stop, run_config.stop
                )
                run_config.stop = None
            elif isinstance(run_config.stop, CombinedStopper):
                new_stoppers = [s for s in run_config.stop._stoppers if not isinstance(s, self.stoppers_to_remove)]
                if len(new_stoppers) != len(run_config.stop._stoppers):
                    logger.info("Removing OptunaPruner stopper to not conflict with the scheduler %s", run_config.stop)
                if len(new_stoppers) == 0:
                    run_config.stop = None
                elif len(new_stoppers) == 1:
                    run_config.stop = new_stoppers[0]
                else:
                    run_config.stop = CombinedStopper(*new_stoppers)
        return run_config


# region ReTuner


class ReTunerSetup(ScheduledTunerSetup[SetupType_co]):
    def create_scheduler(self) -> schedulers.TrialScheduler:
        assert self._setup.args.command is not None
        logger.warning("Creating ReTuneScheduler with hardcoded hyperparam_mutations.")
        return ReTuneScheduler(
            perturbation_interval=self._setup.args.command.perturbation_interval,
            resample_probability=self._setup.args.command.resample_probability,
            hyperparam_mutations={
                "train_batch_size_per_learner": {"grid_search": [256, 512, 1024, 2048]}
            },  # TODO: experimental
            mode=None,  # filled in by Tuner
            metric=None,  # filled in by Tuner
            synch=True,
        )


class PPOMLPWithReTuneSetup(PPOMLPSetup["MLPArgumentParser[PopulationBasedTrainingParser]"]):
    _tuner_setup_cls = ReTunerSetup


# endregion

# region PBT


class PBTTunerSetup(ScheduledTunerSetup["MLPPBTSetup"]):
    def create_searcher(self, stoppers: list[Stopper]):  # noqa: ARG002
        return constrained_minibatch_search(self._setup)

    def create_scheduler(self) -> schedulers.TrialScheduler:
        assert self._setup.args.command is not None
        return self._setup.args.command.to_scheduler()

    def create_callbacks(self, *, adv_loggers: bool | None = True) -> list[Callback]:
        callbacks = super().create_callbacks(adv_loggers=adv_loggers)
        callbacks.append(SyncConfigFilesCallback())
        return callbacks


class MLPPBTSetup(MLPSetup["MLPArgumentParser[PopulationBasedTrainingParser]"]):
    _tuner_setup_cls = PBTTunerSetup

    def _tuner_add_iteration_stopper(self):
        """PBT handles stopping"""
        return False

    def create_tuner(self, *, adv_loggers: bool | None = None) -> tune.Tuner:
        if self.args.command_str != "pbt":
            raise RuntimeError(f"{type(self)} requires 'pbt' command, got '{self.args.command_str}'")
        # Save trial state every 15 minutes as PBT can be long running, can take ~1 min to save
        if os.environ.get("RAY_UTILITIES_NO_PBT_CHECKPOINT_CHANGE") != "1":
            os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = str(60 * 15)
        assert self.args.command is not None
        # NOTE: Uses args.metrics/mode not the args.command.metric/mode
        return super().create_tuner(adv_loggers=True if adv_loggers is None else adv_loggers)


# endregion
