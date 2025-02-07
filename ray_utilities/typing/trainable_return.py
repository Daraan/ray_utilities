from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Required

from .algorithm_return import _NotRequiredEnvRunners

if TYPE_CHECKING:
    from .metrics import _LogMetricsEvaluationResultsDict

# pyright: enableExperimentalFeatures=true


class TrainableReturnData(_NotRequiredEnvRunners, total=False, closed=False):
    evaluation: Required[_LogMetricsEvaluationResultsDict]
    training_iteration: int
    done: Required[bool]
    comment: str
    trial_id: int | str
