# ruff: noqa: ARG002

from __future__ import annotations

from typing import Any

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import NumpyToTensor


class DummyConnector(ConnectorV2):
    """Minimal pass-through connector for Ray RLlib data processing pipelines.

    This connector provides a no-op implementation that returns batch data unchanged.
    It's useful as a placeholder when a connector is required by the framework but
    no actual data processing is needed.

    The connector maintains full compatibility with the ConnectorV2 interface while
    performing zero transformations on the data, making it ideal for testing or
    scenarios where data should flow through unmodified.

    Example:
        >>> connector = DummyConnector()
        >>> result = connector(batch={"obs": observation_data})
        >>> # result == {"obs": observation_data}

    See Also:
        :class:`ray.rllib.connectors.connector_v2.ConnectorV2`: Base connector class
        :class:`DummyNumpyToTensor`: Dummy version of numpy-to-tensor conversion
    """

    def __call__(
        self,
        *,
        batch: dict[str, Any],
        **kwargs,
    ) -> Any:
        """Return the input batch unchanged.

        Args:
            batch: Dictionary containing batch data to process.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            The input batch without any modifications.
        """
        return batch


class DummyNumpyToTensor(DummyConnector, NumpyToTensor):
    """Dummy connector combining pass-through behavior with NumpyToTensor interface.

    This connector inherits from both :class:`DummyConnector` and
    :class:`ray.rllib.connectors.env_to_module.NumpyToTensor`, providing a no-op
    implementation when tensor conversion is expected but not needed.

    It's particularly useful when the framework expects a NumpyToTensor connector
    but the data is already in the appropriate format or conversion should be bypassed.

    See Also:
        :class:`DummyConnector`: Base dummy connector implementation
        :class:`ray.rllib.connectors.env_to_module.NumpyToTensor`: Tensor conversion connector
    """
