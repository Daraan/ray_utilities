"""Helper utilities for Ray initialization with automatic cluster detection.

This module provides utilities to automatically detect and connect to existing Ray
clusters or start a new local cluster based on the environment.

Key Features:
    - Automatic cluster detection via SLURM environment or Ray status
    - Seamless switching between local and SLURM execution
    - Proper resource configuration for both scenarios
    - Integration with ray_utilities runtime environment

Example:
    >>> from ray_utilities.setup import PPOSetup
    >>> from experiments.ray_init_helper import init_ray_with_setup
    >>>
    >>> setup = PPOSetup(config_files=["experiments/default.cfg"])
    >>> with init_ray_with_setup(setup) as ray_context:
    >>>     results = run_tune(setup)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import exceptiongroup  # noqa: F401
import ray

from ray_utilities import get_runtime_env

if TYPE_CHECKING:
    from collections.abc import Generator

    from ray._private.worker import RayContext
    from ray.runtime_env import RuntimeEnv

    from ray_utilities.setup.experiment_base import ExperimentSetupBase

logger = logging.getLogger(__name__)

__all__ = [
    "init_ray_with_setup",
    "is_ray_cluster_available",
]


def is_ray_cluster_available(address: Optional[str] = None) -> bool:
    """Check if a Ray cluster is already running and available.

    This function detects whether a Ray cluster is available by checking:
    1. Explicit address parameter
    2. SLURM_JOB_ID environment variable (indicates SLURM cluster execution)
    3. RAY_ADDRESS environment variable (explicit Ray cluster address)
    4. Existing Ray initialization status

    Args:
        address: Optional explicit Ray cluster address. If provided and not None,
            returns True immediately.

    Returns:
        True if a Ray cluster is available to connect to, False otherwise.

    Note:
        This function does not attempt to connect to the cluster, it only
        checks for indicators that a cluster should be available.
    """
    # Check if explicit address provided
    if address is not None:
        logger.debug("Explicit address provided: %s", address)
        return True

    # Check if running in SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        logger.debug("Detected SLURM environment (SLURM_JOB_ID=%s)", os.environ["SLURM_JOB_ID"])
        return True

    # Check if RAY_ADDRESS is set
    if "RAY_ADDRESS" in os.environ:
        logger.debug("Detected RAY_ADDRESS=%s", os.environ["RAY_ADDRESS"])
        return True

    # Check if Ray is already initialized
    if ray.is_initialized():
        logger.debug("Ray is already initialized")
        return True

    logger.debug("No Ray cluster detected")
    return False


@contextmanager
def init_ray_with_setup(
    setup: ExperimentSetupBase,
    runtime_env: Optional[RuntimeEnv | dict] = None,
    **ray_init_kwargs: Any,
) -> Generator[RayContext, None, None]:
    """Initialize Ray with automatic cluster detection based on setup configuration.

    This context manager automatically chooses between connecting to an existing
    Ray cluster (e.g., on SLURM) or starting a new local cluster based on the
    environment and setup configuration. Resources are read from ``setup.args``
    (num_cpus, ray_address, object_store_memory).

    **Cluster Detection Logic:**
        - If ``setup.args.ray_address`` is provided and not "auto": Connects to that address
        - If SLURM_JOB_ID is set: Uses ``address='auto'`` to connect to SLURM cluster
        - If RAY_ADDRESS is set: Uses ``address='auto'`` to connect to specified cluster
        - If Ray is already initialized: Returns existing context
        - Otherwise: Starts a new local cluster with resources from setup.args

    **Resource Configuration:**
        - For existing clusters: Resource arguments are ignored (cluster manages resources)
        - For new local clusters: Uses ``setup.args.num_cpus`` and ``setup.args.object_store_memory``
        - If not specified in setup.args, uses sensible defaults

    Args:
        setup: An ExperimentSetupBase instance containing experiment configuration.
            Resources are read from ``setup.args`` (from :class:`DefaultResourceArgParser`):
            - ``num_cpus``: Number of CPUs for local cluster
            - ``ray_address``: Ray cluster address ("auto" or explicit address)
            - ``object_store_memory``: Object store memory in bytes
        runtime_env: Optional Ray runtime environment. If None, uses the default
            from ``ray_utilities.runtime_env``.
        **ray_init_kwargs: Additional keyword arguments passed to ``ray.init()``.

    Yields:
        RayContext: The Ray context manager that can be used to interact with Ray.

    Example:
        Basic usage with automatic detection:

        >>> from ray_utilities.setup import PPOSetup
        >>> from ray_utilities import runtime_env
        >>> setup = PPOSetup(config_files=["experiments/default.cfg"])
        >>> with init_ray_with_setup(setup, runtime_env=runtime_env):
        >>>     results = run_tune(setup)

        Custom runtime environment:

        >>> from ray.runtime_env import RuntimeEnv
        >>> custom_env = RuntimeEnv(env_vars={"MY_VAR": "value"})
        >>> with init_ray_with_setup(setup, runtime_env=custom_env):
        >>>     results = run_tune(setup)

        Additional Ray configuration:

        >>> with init_ray_with_setup(setup, include_dashboard=False, log_to_driver=False):
        >>>     results = run_tune(setup)

    Note:
        - Resources (num_cpus, object_store_memory) are read from ``setup.args``
        - Address detection uses ``setup.args.ray_address`` if available
        - On SLURM: Connects to existing cluster; resource args are ignored
        - Locally: Starts new cluster with resources from setup.args
        - The context manager handles both connection and disconnection/shutdown appropriately
    """
    # Use default runtime_env if not provided
    if runtime_env is None:
        runtime_env_to_use = get_runtime_env()
    else:
        runtime_env_to_use = runtime_env

    # Get address from setup.args if available
    address = None
    if hasattr(setup, "args") and hasattr(setup.args, "ray_address"):
        address = setup.args.ray_address
        # Don't use "auto" as explicit address for detection
        if address == "auto":
            address = None

    cluster_available = is_ray_cluster_available(address)

    if cluster_available:
        # Use provided address or default to 'auto'
        connect_address = address if address is not None else "auto"
        logger.info("Connecting to existing Ray cluster (address=%s)", connect_address)
        # Connect to existing cluster (SLURM or pre-started)
        # Resource arguments are ignored when connecting to existing cluster
        context = ray.init(
            address=connect_address,
            runtime_env=runtime_env_to_use,
            **ray_init_kwargs,
        )
    else:
        # Start new local cluster - read resources from setup.args
        num_cpus = None
        if hasattr(setup, "args") and hasattr(setup.args, "num_cpus"):
            num_cpus = setup.args.num_cpus
            logger.debug("Using num_cpus=%s from setup.args", num_cpus)

        object_store_memory = None
        if hasattr(setup, "args") and hasattr(setup.args, "object_store_memory"):
            object_store_memory = setup.args.object_store_memory
            if object_store_memory is not None:
                logger.debug(
                    "Using object_store_memory=%s MB from setup.args",
                    object_store_memory // (1024**2),
                )

        # Apply defaults if not set
        if object_store_memory is None:
            object_store_memory = 2 * 1024**3  # 2 GB default
            logger.debug("Using default object_store_memory=2GB")

        logger.info(
            "Starting new Ray cluster (num_cpus=%s, object_store_memory=%s MB)",
            num_cpus,
            object_store_memory // (1024**2) if object_store_memory else "auto",
        )

        try:
            context = ray.init(
                num_cpus=num_cpus,
                object_store_memory=object_store_memory,
                runtime_env=runtime_env_to_use,
                **ray_init_kwargs,
            )
        except ValueError:
            context = ray.init(
                runtime_env=runtime_env_to_use,
                **ray_init_kwargs,
            )

    try:
        yield context
    finally:
        # Only shutdown if we started the cluster (not connected to existing)
        if not cluster_available and ray.is_initialized():
            logger.debug("Shutting down local Ray cluster")
            ray.shutdown()
        elif cluster_available and ray.is_initialized():
            logger.debug("Disconnecting from Ray cluster")
            ray.shutdown()
