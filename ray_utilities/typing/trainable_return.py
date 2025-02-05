from __future__ import annotations
from typing import TYPE_CHECKING

# pyright: enableExperimentalFeatures=true

from typing_extensions import Required

from ray_utilities.typing.algorithm_return import EvaluationResultsDict, _RequiredEnvRunners

from . import ExtraItems, _PEP_728_AVAILABLE

if _PEP_728_AVAILABLE or TYPE_CHECKING:

    class TrainableReturnData(_RequiredEnvRunners, total=False, extra_items=ExtraItems):
        evaluation: EvaluationResultsDict
        training_iteration: int
        done: Required[bool]
        comment: str
        trial_id: int | str
else:
    TrainableReturnData = dict
