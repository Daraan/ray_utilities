"""
Import that enables argcomplete for this file using the :class:`DefaultArgumentParser`.

Using this import is much faster than using :meth:`DefaultArgumentParser.enable_completion`
as imports are mocked during argument gathering.
"""  # noqa: N999

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray_utilities.config.parser.default_argument_parser import (
        DefaultArgumentParser,
    )

parser: str = "ray_utilities.config.parser.default_argument_parser.DefaultArgumentParser"
"""
Import path to the parser class to be used for shell completion.
Must either point to a class implementing :meth:`~DefaultArgumentParser.enable_completion`
or be a compatible parser for :meth:`argcomplete.autocomplete`.
"""

# Only run this code in completion mode
if "COMP_LINE" in os.environ:
    import argcomplete

    # Now we can safely import what we need for completion

    # Setup more advanced mocking system
    from sphinx.ext.autodoc.mock import _MockModule
    from sphinx.ext.autodoc.mock import mock as autodoc_mock

    # Create a logger specifically for the mocking system
    mock_logger = logging.getLogger("mock_imports")
    mock_logger.setLevel(logging.DEBUG)

    import ray.tune.schedulers  # real import for to_tap_class  # noqa: F401

    # Create a more comprehensive list of modules to mock
    modules_to_mock = [
        # Ray and its components with deeper paths
        "ray",
        "ray.tune",
        "ray.rllib",
        "ray.train",
        "ray.air",
        "ray.experimental",
        # Important Ray submodules that are commonly imported
        "ray.rllib.core",
        "ray.rllib.utils",
        "ray.rllib.utils.metrics",
        "ray.rllib.utils.typing",
        "ray.rllib.policy",
        "ray.rllib.algorithms",
        "ray.rllib.algorithms.algorithm",
        "ray.rllib.algorithms.callbacks",
        "ray.rllib.models",
        "ray.rllib.evaluation",
        "ray.rllib.execution",
        # Tune modules
        "ray.tune.integration",
        "ray.tune.schedulers",
        "ray.tune.search",
        "ray.tune.result",
        "ray.tune.error",
        "ray.tune.logger",
        # Deep learning frameworks
        "tensorflow",
        "tensorflow_probability",
        "torch",
        "jax",
        "flax",
        "haiku",
        "optax",
        # Experiment tracking
        "wandb",
        "comet_ml",
        "mlflow",
        # RL environments
        "gym",
        "gymnasium",
        "pybullet_envs",
        "mujoco",
        "dm_control",
        # Other heavy dependencies
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "scipy",
        "opencv-python",
        "PIL",
        "imageio",
        "h5py",
    ]
    import atexit

    ad_mock = autodoc_mock(modules_to_mock)
    ad_mock.__enter__()
    atexit.register(ad_mock.__exit__, None, None, None)

    # Use two dummy classes to not have multiple base classes
    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

    class Dummy2:
        def __init__(self, *args, **kwargs):
            pass

    # mock has a bug that one cannot subclass MockObjects together with typing.Generic, add non mock classes for that:
    class TuneMock(_MockModule):
        Trainable = Dummy

    tune_mock = TuneMock("ray.tune")

    class RayMock(_MockModule):
        tune = tune_mock

    sys.modules["ray"] = RayMock("ray")
    sys.modules["gym"] = _MockModule("gym")
    sys.modules["gymnasium"] = _MockModule("gymnasium")
    from default_arguments import _ray_metrics

    sys.modules["ray.rllib.utils.metrics"] = _ray_metrics
    sys.modules["ray.rllib.utils.metrics.metrics_logger"] = _MockModule("ray.rllib.utils.metrics.metrics_logger")
    sys.modules["ray"].__version__ = "2.48.0+mocked"  # type: ignore[attr-defined]
    sys.modules["gym"].__version__ = "0.26.0+mocked"  # type: ignore[attr-defined]
    sys.modules["gymnasium"].__version__ = "1.0.0+mocked"  # type: ignore[attr-defined]
    sys.modules["ray.rllib.utils.checkpoints"] = _MockModule("ray.rllib.utils.checkpoints")
    sys.modules["ray.tune"] = _MockModule("ray.tune")
    sys.modules["ray.tune.trainable"] = _MockModule("ray.tune.trainable")

    sys.modules["ray.tune"] = tune_mock
    sys.modules["ray.tune.trainable"].Trainable = Dummy  # type: ignore[attr-defined]
    sys.modules["ray.rllib.utils.checkpoints"].Checkpointable = Dummy2  # type: ignore[attr-defined]
    sys.modules["ray.rllib.core.rl_module"] = _MockModule("ray.rllib.core.rl_module")
    sys.modules["ray.rllib.core.rl_module"].RLModule = Dummy  # type: ignore[attr-defined]
    sys.modules["ray.rllib.core.models.catalog"] = _MockModule("ray.rllib.core.models.catalog")
    sys.modules["ray.rllib.core.models.catalog"].Catalog = Dummy  # type: ignore[attr-defined]
    sys.modules["ray.rllib.utils.metrics.stats"] = _MockModule("ray.rllib.utils.metrics.stats")
    sys.modules["ray.rllib.utils.annotations"] = _MockModule("ray.rllib.utils.annotations")
    sys.modules["ray.rllib.utils.annotations"].override = lambda _: lambda x: x  # type: ignore[attr-defined]

    def import_object_from_string(path: str):
        """Import an object given its full dotted path."""
        module_path, _, attr = path.rpartition(".")
        if not module_path:
            raise ValueError(f"Invalid import path: {path}")
        module = importlib.import_module(module_path)
        return getattr(module, attr)

    _parser_for_complete: DefaultArgumentParser | Any = import_object_from_string(
        "ray_utilities.config.parser.default_argument_parser.DefaultArgumentParser"
    )

    if hasattr(_parser_for_complete, "enable_completion"):
        _parser_for_complete.enable_completion()
    else:
        argcomplete.autocomplete(_parser_for_complete)
