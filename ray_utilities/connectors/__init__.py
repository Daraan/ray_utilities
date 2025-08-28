"""Custom Ray RLlib connectors for data processing and debugging.

This module provides custom connector implementations that extend Ray RLlib's
connector framework for processing data between different components of the
training pipeline. Connectors handle data transformation, filtering, and
debugging in the data flow between environments, agents, and learners.

The connectors in this module focus on debugging capabilities, data validation,
and specialized data processing needs that aren't covered by the standard
RLlib connector library.

Key Components:
    - :class:`DebugConnector`: Debugging connector with logging and breakpoint support
    - Custom data processing connectors for specialized use cases
    - Integration with RLlib's ConnectorV2 framework

These connectors can be inserted into the RLlib data processing pipeline to
add custom functionality while maintaining compatibility with the standard
training workflow.
"""
