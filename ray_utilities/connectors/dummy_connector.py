# ruff: noqa: ARG002

from __future__ import annotations

from typing import Any

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import NumpyToTensor


class DummyConnector(ConnectorV2):
    """A dummy connector to be used instead of other connectors when a connector is needed"""

    def __call__(
        self,
        *,
        batch: dict[str, Any],
        **kwargs,
    ) -> Any:
        """Return unmodified batch."""
        return batch


class DummyNumpyToTensor(DummyConnector, NumpyToTensor):
    """A dummy connector to be used instead of a NumpyToTensor or other connectors when a connector is needed"""
