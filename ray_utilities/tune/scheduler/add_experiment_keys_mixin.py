from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ray.tune.execution.tune_controller import TuneController
from ray.tune.schedulers.pbt import PopulationBasedTraining

from ray_utilities.tune.experiments import set_experiment_key_on_trial

if TYPE_CHECKING:
    from ray.tune.execution.tune_controller import TuneController
    from ray.tune.experiment import Trial


logger = logging.getLogger(__name__)


class AddExperimentKeysMixin(PopulationBasedTraining):
    """Executes :func:`set_experiment_key_on_trial` for the started trials"""

    def on_trial_add(self, tune_controller: TuneController, trial: Trial, **kwargs):
        set_experiment_key_on_trial(trial, pbt_epoch=getattr(self, "_current_epoch", None))
        super().on_trial_add(tune_controller, trial, **kwargs)
