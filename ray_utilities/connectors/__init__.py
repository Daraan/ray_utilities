"""Custom Ray RLlib connectors for data processing and debugging.

Provides custom connector implementations that extend Ray RLlib's connector framework
for processing data between environments, agents, and learners. Includes debugging
capabilities and specialized data processing connectors.

Key Components:
    - :class:`DebugConnector`: Debugging connector with logging capabilities
    - Data filtering and processing connectors
    - Integration with RLlib's ConnectorV2 framework
"""
