from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

if TYPE_CHECKING:
    from ray.rllib.core.rl_module.rl_module import RLModule


class LimitedToNumpyConverter(ConnectorV2):
    """
    Converts Jax arrays to numpy to pass trough the SampleBatch converter

    Warning:
        Experimental might slow down the training;
        jax -> numpy -> jax

    Possibly usable for GAE results
    """

    def __call__(
        self,
        *,
        rl_module: RLModule | MultiRLModule,  # noqa: ARG002
        batch: dict[str, Any],
        **kwargs,  # noqa: ARG002
    ) -> Any:
        # Code from NumpyToTensor

        is_single_agent = False
        is_multi_rl_module = isinstance(rl_module, MultiRLModule)
        # `data` already a ModuleID to batch mapping format.
        if not (is_multi_rl_module and all(c in rl_module._rl_modules for c in batch)):  # pyright: ignore[reportAttributeAccessIssue]
            is_single_agent = True
            batch = {DEFAULT_MODULE_ID: batch}

        for module_id, module_data in batch.copy().items():
            infos = module_data.pop(Columns.INFOS, None)
            for k in ("advantages",):
                module_data[k] = np.asarray(module_data[k])
            if infos is not None:
                module_data[Columns.INFOS] = infos
            # Early out with data under(!) `DEFAULT_MODULE_ID`, b/c we are in plain
            # single-agent mode.
            if is_single_agent:
                return module_data
            batch[module_id] = module_data

        return batch
