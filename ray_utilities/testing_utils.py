"""Testing utilities and mock classes for Ray RLlib experiments.

This module provides comprehensive testing infrastructure for Ray RLlib and related
workflows, including:

- Custom test case base classes with specialized setup/teardown
- Mock implementations of Ray Tune components for testing
- Debugging utilities including remote debugging support
- JAX testing utilities and numerical comparison helpers
- Progress bar and checkpoint testing utilities

The module is designed to support both unit tests and integration tests for
machine learning experiments, with particular focus on reinforcement learning
workflows using Ray RLlib.
"""
# pyright: reportOptionalMemberAccess=information
# pyright: enableExperimentalFeatures=true

from __future__ import annotations

import atexit
import difflib
import logging
import math
import os
import pathlib
import pprint
import random
import shutil
import sys
import unittest
import unittest.util
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import ContextDecorator, nullcontext
from copy import deepcopy
from functools import partial, wraps
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Iterable,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    final,
    overload,
)
from unittest import mock

import debugpy  # noqa: T100
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy.testing as npt
import ray
import ray.tune
import ray.tune.logger
import ray.tune.logger.unified
import tree
from ray.experimental import tqdm_ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms import algorithm as algorithm_module
from ray.rllib.core import ALL_MODULES
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    LEARNER_RESULTS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES,
    NUM_MODULE_STEPS_SAMPLED,
    NUM_MODULE_STEPS_SAMPLED_LIFETIME,
    TIMERS,
)
from ray.rllib.utils.metrics.stats import Stats
from ray.train import Checkpoint
from ray.train._internal.checkpoint_manager import _CheckpointManager
from ray.train._internal.session import _FutureTrainingResult, _TrainingResult
from ray.tune import CheckpointConfig
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment.trial import Trial, _TemporaryTrialState
from ray.tune.result import TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.schedulers import TrialScheduler
from ray.tune.search.sample import Categorical, Domain, Float, Integer
from ray.tune.stopper import CombinedStopper
from ray.tune.trainable.metadata import _TrainingRunMetadata
from testfixtures import LogCapture
from typing_extensions import Final, NotRequired, Required, Sentinel, TypeAliasType, get_origin, get_type_hints

import ray_utilities.callbacks.algorithm.model_config_saver_callback
import ray_utilities.config.create_algorithm
from ray_utilities import runtime_env
from ray_utilities.config import DefaultArgumentParser
from ray_utilities.config.parser.mlp_argument_parser import MLPArgumentParser
from ray_utilities.constants import ENVIRONMENT_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME
from ray_utilities.dynamic_config.dynamic_buffer_update import logger as dynamic_buffer_logger
from ray_utilities.misc import is_pbar, raise_tune_errors
from ray_utilities.nice_logger import change_log_level
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.setup.experiment_base import logger as experiment_base_logger
from ray_utilities.setup.ppo_mlp_setup import MLPSetup, PPOMLPSetup
from ray_utilities.setup.tuner_setup import TunerSetup
from ray_utilities.setup.tuner_setup import logger as tuner_setup_logger
from ray_utilities.training.default_class import DefaultTrainable, TrainableBase, TrainableStateDict
from ray_utilities.training.functional import training_step
from ray_utilities.training.helpers import make_divisible, nan_to_zero_hist_leaves

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


if TYPE_CHECKING:
    from collections.abc import Callable

    import chex
    from flax.training.train_state import TrainState
    from jaxlib.xla_extension import pytree  # pyright: ignore[reportMissingModuleSource,reportMissingImports] pyi file
    from ray import tune
    from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
    from ray.rllib.callbacks.callbacks import RLlibCallback
    from ray.tune import Result

    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co
    from ray_utilities.typing import StopperType
    from ray_utilities.typing.algorithm_return import StrictAlgorithmReturnData
    from ray_utilities.typing.metrics import LogMetricsDict

    LeafType = TypeAliasType("LeafType", "pytree.SequenceKey | pytree.DictKey | pytree.GetAttrKey")

if "--fast" in sys.argv:
    TWO_ENV_RUNNER_CASES: list[tuple[int, int]] = [(0, 1)]
elif "--mp-only" in sys.argv:
    TWO_ENV_RUNNER_CASES = [(1, 2)]
else:
    TWO_ENV_RUNNER_CASES = [(0, 1), (0, 2), (1, 2)]

if "--fast" in sys.argv:
    ENV_RUNNER_CASES: list[int] = [0]
elif "--mp-only" in sys.argv:
    ENV_RUNNER_CASES = [1, 2]
else:
    ENV_RUNNER_CASES = [0, 1, 2]


args_train_no_tuner = mock.patch.object(
    sys, "argv", ["file.py", "--a", "NA", "--no-render_env", "-J", "1", "-it", "2", "-np"]
)
clean_args = mock.patch.object(sys, "argv", ["file.py", "-a", "NA"])
"""Use when comparing to CLIArgs"""

_CMock = TypeVar("_CMock", bound="Callable[[Any, mock.MagicMock], Any]")
_C = TypeVar("_C", bound="Callable[..., Any]")

_NOT_PROVIDED = Sentinel(
    "_NOT_PROVIDED",
)

logger = logging.getLogger(__name__)


@final
class Cases:
    def __init__(self, cases: Iterable[Any] | Callable[[], Any] | BaseException, *args, **kwargs):  # noqa: ARG002
        self._cases = cases
        self._args = args
        self._kwargs = kwargs

    def __call__(self, func: _CMock) -> _CMock:
        """Allows to use TestCases as a decorator."""
        return self.cases(self._cases, *self._args, **self._kwargs)(func)  # pyright: ignore[reportReturnType]

    @classmethod
    def next(cls) -> Any:
        raise NotImplementedError("Mock this function with the test cases to return")

    @classmethod
    def cases(cls, cases: Iterable[Any] | Callable[[], Any] | BaseException, *args, **kwargs):
        return mock.patch.object(cls, "next", *args, side_effect=cases, **kwargs)


def iter_cases(cases: type[Cases] | mock.MagicMock | Iterator[Any] | Iterable[Any]):
    try:
        while True:
            if isinstance(cases, mock.MagicMock):
                next_case = cases()
            elif isinstance(cases, Iterator):
                next_case = next(cases)
            elif isinstance(cases, Iterable):
                yield from iter_cases(iter(cases))
                return
            else:
                next_case = cases.next()
            logger.info("======= NEXT CASE: %s =======", next_case)
            yield next_case
    except StopIteration:
        return
    except BaseException:
        raise


@overload
def check_args(func: None = None, *, exceptions: Optional[list[str]] = None, expected=None) -> Callable[[_C], _C]: ...


@overload
def check_args(func: _C, *, exceptions: Optional[list[str]] = None, expected=None) -> _C: ...


def check_args(
    func: _C | None = None, *, exceptions: Optional[list[str]] = None, expected=None
) -> _C | Callable[[_C], _C]:
    """
    Attention:
        This function must be wrapped by patch_args
    """
    if func is None:
        if exceptions is None and expected is None:
            raise ValueError("Either exceptions or expected must be provided")
        return partial(check_args, exceptions=exceptions, expected=expected)

    @wraps(func)
    def wrapper(*args, **kwargs):
        error = None
        with LogCapture(
            "ray_utilities.setup.experiment_base",
            level=logging.WARNING,
            attributes=["getMessage", "args", "levelname", "name"],
        ) as capture:
            try:
                result = func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                error = e
        record_errors = []
        for record in capture.records:
            if "The following arguments were not recognized by the parser" in record.message:
                assert record.args
                if isinstance(record.args, (list, tuple)):
                    if not isinstance(record.args[0], list):
                        logger.error("Expected a list as the first element of args but got: %s", record.args[0])
                        continue
                    unknown_args: list[str] = record.args[0].copy()  # expect a list here
                else:
                    logger.error("Expected a list or tuple as args passed to the logger but got: %s", record.args)
                    continue
                assert isinstance(unknown_args, list), (
                    f"Expected a list of arguments passed to parser.parse_args, got: {unknown_args}"
                )
                assert len(unknown_args) > 0
                if os.path.exists(unknown_args[0]):
                    unknown_args.pop(0)
                # Compare with patch_args input
                if exceptions:
                    # We might not have access to patched args
                    # Find all matching sub-sequences of arexceptionsgs in unknown_args
                    matches = []
                    args_seq = tuple(unknown_args)
                    argv_seq = tuple(exceptions)
                    for i in range(len(argv_seq) - len(args_seq) + 1):
                        if argv_seq[i : i + len(args_seq)] == args_seq:
                            matches.append(i)
                    # Remove subsequences from unknown_args:
                    for match in matches:
                        unknown_args = unknown_args[:match] + unknown_args[match + len(args_seq) :]
                if unknown_args:
                    args_error = ValueError(f"Unexpected unrecognized args: {unknown_args}")
                    record_errors.append(args_error)
        if record_errors:
            if error:
                record_errors.insert(0, error)
            if len(record_errors) == 1:
                raise record_errors[0]
            raise ExceptionGroup("Unexpected unrecognized args and further errors encountered.", record_errors)
        if error is not None:
            raise error
        return result  # pyright: ignore[reportPossiblyUnboundVariable]

    return wrapper


class PatchArgsDecorator(ContextDecorator):
    def _extract_arg_errors(self, log_capture):
        errors = []
        for record in log_capture.records:
            if "The following arguments were not recognized by the parser" in record.message:
                unknown_args = record.args[0] if record.args and isinstance(record.args[0], list) else []
                if unknown_args and self._exceptions:
                    args_seq = tuple(unknown_args)
                    argv_seq = tuple(self._exceptions)
                    matches = [
                        i
                        for i in range(len(args_seq) - len(argv_seq) + 1)
                        if args_seq[i : i + len(argv_seq)] == argv_seq
                    ]
                    for match in matches:
                        unknown_args = list(args_seq[:match]) + list(args_seq[match + len(argv_seq) :])
                        args_seq = tuple(unknown_args)
                if unknown_args:
                    errors.append(ValueError(f"Unexpected unrecognized args: {unknown_args}"))
        return errors

    def __init__(self, patch_obj: mock._patch, exceptions):
        self._patch_obj = patch_obj
        self._exceptions = exceptions
        self._patch_cm = None
        self._log_capture = None

    def __enter__(self):
        self._patch_cm = self._patch_obj.__enter__()
        self._log_capture = LogCapture(
            "ray_utilities.setup.experiment_base",
            level=logging.WARNING,
            attributes=["getMessage", "args", "levelname", "name"],
        )
        self._log_capture.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._patch_cm = None
        self._log_capture.__exit__(exc_type, exc_value, traceback)  # pyright: ignore[reportOptionalMemberAccess]
        arg_errors = self._extract_arg_errors(self._log_capture)
        if arg_errors:
            if exc_value:
                arg_errors.insert(0, exc_value)
            if len(arg_errors) == 1:
                raise arg_errors[0]
            raise ExceptionGroup("Unexpected unrecognized args and further errors encountered.", arg_errors)
        if exc_value is not None:
            raise exc_value
        return self._patch_obj.__exit__(exc_type, exc_value, traceback)

    def __call__(self, func: _C) -> _C:
        @wraps(func)
        def wrapper(*args, **kwargs):
            error = None
            with self._patch_obj:
                with LogCapture(
                    "ray_utilities.setup.experiment_base",
                    level=logging.WARNING,
                    attributes=["getMessage", "args", "levelname", "name"],
                ) as capture:
                    try:
                        result = func(*args, **kwargs)
                    except Exception as e:  # noqa: BLE001
                        error = e
                arg_errors = self._extract_arg_errors(capture)
                if arg_errors:
                    if error:
                        arg_errors.insert(0, error)
                    if len(arg_errors) == 1:
                        raise arg_errors[0]
                    raise ExceptionGroup("Unexpected unrecognized args and further errors encountered.", arg_errors)
                if error is not None:
                    raise error
                return result  # pyright: ignore[reportPossiblyUnboundVariable]

        return wrapper  # pyright: ignore[reportReturnType]


@overload
def patch_args(
    *args: str | int,
    extend_argv: bool = False,
    log_level: str | None = "DEBUG",
    check_for_errors: Literal[False],
    except_parser_errors: Optional[list[str]] = None,
) -> mock._patch: ...


@overload
def patch_args(
    *args: str | int,
    extend_argv: bool = False,
    log_level: str | None = "DEBUG",
    check_for_errors: Literal[True] = True,
    except_parser_errors: Optional[list[str]] = None,
) -> PatchArgsDecorator: ...


def patch_args(
    *args: str | int,
    extend_argv: bool = False,
    log_level: str | None = "DEBUG",
    check_for_errors: bool = True,
    except_parser_errors: Optional[list[str]] = None,
) -> mock._patch[list[str]] | PatchArgsDecorator:
    """Patch sys.argv. Optionally compose with check_args.

    If neither check_exceptions nor check_expected are provided this function
    behaves like before and returns the unittest.mock._patch object so it can
    be used directly as a decorator or context manager. When either of the
    check_args parameters is provided, a decorator is returned that first
    applies the check_args wrapper to the target function and then applies
    the argv patch (so both effects are combined).
    """
    old_args = sys.argv[1:]
    actor_args = (
        ("-a", "no_actor_by_patch")
        if (
            "-a" not in args
            and "--agent_type" not in args
            and (not extend_argv or ("-a" not in old_args and "--agent_type" not in old_args))
        )
        else ()
    )
    log_args = ()
    if log_level and "--log_level" not in args:
        log_args = ("--log_level", "DEBUG")
    patched_args = map(str, (*old_args, *args) if extend_argv else args)
    patched_args = [
        sys.argv[0] if sys.argv else "_imaginary_file_for_patch.py",
        *actor_args,
        *log_args,
        *patched_args,
    ]
    if "--test" in patched_args and "COMET_API_KEY" not in os.environ:
        logger.warning("Using --test in tests will enable Comet/Wandb on GitHub Actions but API might be missing.")
    patch_obj = mock.patch.object(
        sys,
        "argv",
        patched_args,
    )
    if not check_for_errors:
        return patch_obj

    # Otherwise return a decorator/contextmanager that applies check_args then the patch.

    return PatchArgsDecorator(patch_obj, except_parser_errors)


def get_explicit_required_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is Required}


def get_explicit_unrequired_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is NotRequired}


def get_required_keys(cls):
    return cls.__required_keys__ - get_explicit_unrequired_keys(cls)


def get_optional_keys(cls):
    return cls.__optional__keys - get_explicit_required_keys(cls)


_NOT_FOUND = object()


def get_leafpath_value(leaf: "LeafType"):
    """Returns the path value of a leaf, could be index (list), key (dict), or name (attribute)."""
    return getattr(leaf, "name", getattr(leaf, "key", getattr(leaf, "idx", _NOT_FOUND)))


def _noop_callback_replacement(*a, **k):  # noqa: ARG001
    return None


class DisableLoggers(unittest.TestCase):
    """Disable loggers for tests, so they do not interfere with the output."""

    def enable_loggers(self):
        """Enable loggers after disabling them in setUp."""
        self._mock_env.stop()
        self._disable_tune_loggers.stop()
        self._disable_file_loggers.stop()
        self._disable_file_loggers2.stop()
        self._disable_save_model_architecture_callback_added.stop()
        self._disable_save_model_architecture_module.stop()

    def setUp(self):
        super().setUp()
        self._mock_env = mock.patch.dict("os.environ", {"TUNE_DISABLE_AUTO_CALLBACK_LOGGERS": "1"})
        self._mock_env.start()
        self._disable_tune_loggers = mock.patch("ray_utilities.callbacks.tuner.create_tuner_callbacks", return_value=[])
        self._disable_tune_loggers.start()
        self._disable_file_loggers = mock.patch.object(ray.tune.logger, "DEFAULT_LOGGERS", ())
        self._disable_file_loggers.start()
        self._disable_file_loggers2 = mock.patch.object(ray.tune.logger.unified, "DEFAULT_LOGGERS", ())
        """Disable local copy used by UnifiedLogger"""
        self._disable_file_loggers2.start()
        self._disable_save_model_architecture_module = mock.patch(
            "ray_utilities.callbacks.algorithm.model_config_saver_callback"
        )

        self._disable_save_model_architecture_callback_added = mock.patch(
            "ray_utilities.config.create_algorithm.save_model_config_and_architecture",
            new=_noop_callback_replacement,
        )
        self._disable_save_model_architecture_callback_added.start()
        self._disable_save_model_architecture_module.start()

    def tearDown(self):
        self.enable_loggers()
        super().tearDown()


class InitRay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Ray for the test class."""
        # TODO: Possibly also test without runtime env to check errors, especially comet related
        if not ray.is_initialized():
            # NOTE: might have already been started (in surprising ways) by another test
            # e.g. EnvRunnerGroup creation. In that case runtime_env is NOT applied!
            ray.init(
                include_dashboard=False,
                ignore_reinit_error=True,
                num_cpus=cls._num_cpus,
                object_store_memory=1024**3 // 2,  # 512MB
                runtime_env=runtime_env,
            )
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        """Shutdown Ray after the test class."""
        if ray.is_initialized():
            ray.shutdown()
        super().tearDownClass()

    def __init_subclass__(cls, num_cpus: Optional[int] = None, *args, **kwargs) -> None:
        cls._num_cpus = num_cpus
        super().__init_subclass__(*args, **kwargs)


OVERRIDE_KEYS: Final[set[str]] = {"num_env_runners", "num_epochs", "minibatch_size", "train_batch_size_per_learner"}
"""Keys that are overridden in set_trainable"""


def _remove_values_on_tensor_stats(struct, path: tuple[str, ...] = (), parent: dict[str, Any] | None = None):
    if isinstance(struct, dict):
        return {k: _remove_values_on_tensor_stats(v, (*path, k), struct) for k, v in struct.items()}
    if isinstance(struct, list):
        return [_remove_values_on_tensor_stats(v, path, parent) for v in struct]
    if path and path[-1] == "values" and parent is not None and parent.get("_is_tensor"):
        if isinstance(struct, deque):
            return deque(maxlen=struct.maxlen)
    elif path and path[-1] == "_is_tensor":
        return False
    return struct


def _fix_throughput_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Using > 1 num_env_runners will save > 1 value in throughput_stats, get_state will reduce them however"""
    stats = stats.copy()
    for k, stat in stats.items():
        if "throughput_stats" not in stat:
            continue
        stats[k] = stat = stat.copy()  # noqa: PLW2901
        t_stat = Stats.from_state(stat["throughput_stats"])
        stat["throughput_stats"]["values"] = t_stat.peek()
    return stats


def _remove_throughput_stats(stats: dict[str, Any]):
    stats = stats.copy()
    for k, stat in stats.items():
        if "throughput_stats" not in stat:
            continue
        stats[k] = stat = stat.copy()  # noqa: PLW2901
        del stat["throughput_stats"]
    return stats


class TestHelpers(unittest.TestCase):
    # region setups
    _fast_model_fcnet_hiddens: int = 1

    @classmethod
    def _disable_ray_auto_init(cls):
        cls._pop_auto_connect = False
        cls._auto_init_hook = None
        if "RAY_ENABLE_AUTO_CONNECT" not in os.environ:
            os.environ["RAY_ENABLE_AUTO_CONNECT"] = "0"
            cls._pop_auto_connect = True
            try:
                import ray._private.auto_init_hook  # noqa: PLC0415

                cls._auto_init_hook = ray._private.auto_init_hook
            except ImportError:
                pass
            else:
                ray._private.auto_init_hook.enable_auto_connect = False

    @classmethod
    def _enable_ray_auto_init(cls):
        """NOTE: This is a classmethod, should only be called on classSetup/tearDownClass."""
        if cls._pop_auto_connect:
            os.environ.pop("RAY_ENABLE_AUTO_CONNECT", None)
        if cls._auto_init_hook is not None:
            cls._auto_init_hook.enable_auto_connect = True

    @classmethod
    def setUpClass(cls):
        cls._disable_ray_auto_init()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls._enable_ray_auto_init()
        super().tearDownClass()

    def setUp(self):
        # Do not initialize ray if we do not have to
        super().setUp()
        AlgorithmSetup.PROJECT = "TESTING"
        os.environ["WANDB_API_KEY"] = "test"
        assert TrainableBase.cls_model_config is None
        TrainableBase.cls_model_config = {}
        self.mock_reduced_model = mock.patch.dict(
            TrainableBase.cls_model_config,
            {"fcnet_hiddens": [self._fast_model_fcnet_hiddens], "head_fcnet_hiddens": []},
        )
        self.mock_reduced_model.start()
        self._env_seed_rng = random.Random(111)
        atexit.register(self._clean_output_dir)

    def tearDown(self):
        super().tearDown()
        TrainableBase.cls_model_config = None
        self.mock_reduced_model.stop()
        for trainable in self._created_trainables:
            trainable.stop()

    @staticmethod
    def _clean_output_dir():
        # if on GitHub do not clean
        if "GITHUB_REF" in os.environ:
            logger.info("Skipping cleaning output dir in GitHub Actions")
            return
        if "RAY_UTILITIES_KEEP_TESTING_STORAGE" in os.environ:
            logger.info("Skipping cleaning output dir, RAY_UTILITIES_KEEP_TESTING_STORAGE is set")
            return
        # Remove TESTING storage path
        try:
            AlgorithmSetup.PROJECT = "TESTING"
            # Create run config to have access to output dir
            with (
                change_log_level(experiment_base_logger, logging.ERROR),
                change_log_level(tuner_setup_logger, logging.ERROR),
                mock.patch("logging.getLogger") as mock_get_logger,  # not log or adjust
            ):
                mock_get_logger.return_value.name.split.return_value = ["Nothing"]
                run_config = TunerSetup(
                    setup=AlgorithmSetup(
                        init_config=False, init_trainable=False, init_param_space=False, change_log_level=False
                    )
                ).create_run_config([])
            if run_config.storage_path is None:
                return
            storage_path = pathlib.Path(run_config.storage_path) / run_config.name  # pyright: ignore[reportOperatorIssue]
            if storage_path.exists():
                assert "TESTING" in storage_path.name, f"{storage_path} is not a TESTING storage path"
                logger.info("Removing testing storage path: %s", storage_path)

                shutil.rmtree(storage_path.as_posix(), ignore_errors=True)
        except OSError:
            logger.exception("Failed to remove testing storage path")
        except Exception:
            logger.exception("Failed to remove testing storage path, unknown error")

    _created_trainables: ClassVar[list[TrainableBase]] = []

    @patch_args(
        "--iterations", "5",
        "--total_steps", "320",
        "--batch_size", "64",  # overwritten with 32!
        "--comment", "created by TestHelpers.get_trainable",
        "--seed", "42",
        "--min_step_size", "64",  # try not to adjust total_steps
        "--max_step_size", "64",  # try not to adjust total_steps
        "--num_envs_per_env_runner", "1",
        "--no_dynamic_eval_interval",
    )  # fmt: skip
    def get_trainable(
        self,
        *,
        num_env_runners: int = 0,
        env_seed: int | None | _NOT_PROVIDED = _NOT_PROVIDED,
        train: bool = True,
        fast_model=True,
        eval_interval: Optional[int] = 1,
    ):
        # NOTE: In this test attributes are shared BY identity, this is just a weak test.
        if fast_model:
            self._model_config = {"fcnet_hiddens": [self._fast_model_fcnet_hiddens], "head_fcnet_hiddens": []}
        else:
            self._model_config = None
        self.TrainableClass: type[DefaultTrainable[DefaultArgumentParser, PPOConfig, PPO]] = DefaultTrainable.define(
            PPOMLPSetup.typed(), model_config=self._model_config
        )
        if self._model_config is not None:
            self.TrainableClass.cls_model_config = self._model_config
        # this initializes the algorithm; overwrite batch_size of 64 again.
        # This does not modify the state["setup"]["config"]
        overrides = AlgorithmConfig.overrides(
            num_env_runners=num_env_runners,
            num_epochs=2,
            minibatch_size=32,
            train_batch_size_per_learner=32,
        )
        if eval_interval is not None:
            overrides["evaluation_interval"] = eval_interval
        if env_seed is _NOT_PROVIDED:
            # use a random but reproducible seed
            if not hasattr(self, "_env_seed_rng"):
                self.setUp()
            env_seed = self._env_seed_rng.randint(0, 2**15 - 1)
        with (
            MLPArgumentParser.patch_args(
                "--fcnet_hiddens", self._model_config["fcnet_hiddens"][0], # pyright: ignore[reportOptionalSubscript]
                "--head_fcnet_hiddens", "[]",
            )
            if fast_model
            else nullcontext()
        ):  # fmt: skip
            trainable = self.TrainableClass(
                {"env_seed": env_seed}, algorithm_overrides=overrides, model_config=self._model_config
            )
        self._created_trainables.append(trainable)
        self.assertEqual(trainable._algorithm_overrides, overrides)
        if eval_interval is None:
            self.assertSetEqual(set(overrides.keys()), OVERRIDE_KEYS)
        else:
            self.assertSetEqual(set(overrides.keys()), OVERRIDE_KEYS | {"evaluation_interval"})
            self.assertEqual(trainable.algorithm_config.evaluation_interval, eval_interval)
        self.assertEqual(trainable.algorithm_config.num_env_runners, num_env_runners)
        self.assertEqual(trainable.algorithm_config.minibatch_size, 32)
        self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 32)  # overwritten
        self.assertEqual(trainable.algorithm_config.num_epochs, 2)
        self.assertEqual(trainable._setup.args.iterations, 5)
        self.assertEqual(trainable._setup.args.total_steps, 320)
        self.assertEqual(trainable._setup.args.train_batch_size_per_learner, 64)  # not overwritten

        if not train:
            return trainable, None
        result1 = trainable.train()
        self.assertEqual(result1[TRAINING_ITERATION], 1)
        self.assertEqual(result1["current_step"], 32)
        self.assertFalse(trainable._setup.args.no_exact_sampling)
        self.assertEqual(
            trainable.algorithm.metrics.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME)), 32
        )
        return trainable, result1

    @staticmethod
    def on_checkpoint_loaded_callbacks(trainable: TrainableBase[Any, Any, Algorithm | Any]):
        """Executed the on_checkpoint_loaded callbacks of the algorithm if any."""
        if trainable.algorithm.callbacks is not None:
            if isinstance(trainable.algorithm.callbacks, Iterable):
                cb: RLlibCallback
                for cb in trainable.algorithm.callbacks:
                    cb.on_checkpoint_loaded(algorithm=trainable.algorithm, metrics_logger=trainable.algorithm.metrics)
            else:
                trainable.algorithm.callbacks.on_checkpoint_loaded(
                    algorithm=trainable.algorithm, metrics_logger=trainable.algorithm.metrics
                )

    # endregion

    def check_tune_result(self, result: tune.ResultGrid):
        raise_tune_errors(result)
        self.assertEqual(result.num_errors, 0, format_result_errors(result.errors))  # pyright: ignore[reportAttributeAccessIssue,reportOptionalSubscript]
        return True

    def set_max_diff(self, max_diff: int | None = None):
        """Changes the maxDiff only when environment variable KEEP_MAX_DIFF is not set."""
        if int(os.environ.get("KEEP_MAX_DIFF", "0")):
            return
        self.maxDiff = max_diff

    def no_pbar_updates(self):
        # points were they are
        import ray_utilities.callbacks.progress_bar
        import ray_utilities.training.default_class
        import ray_utilities.training.functional

        pbar_updates = [
            mock.patch.object(ray_utilities.callbacks.progress_bar, "update_pbar"),
            mock.patch.object(ray_utilities.training.default_class, "update_pbar"),
            mock.patch.object(ray_utilities.training.functional, "update_pbar"),
        ]
        for pbar_update in pbar_updates:
            pbar_update.start()

    # region tests

    def util_test_tree_equivalence(
        self,
        tree1: TrainState | Any,
        tree2: TrainState | Any,
        ignore_leaves: Collection[str] = (),
        msg: str = "",
        attr_checked: str = "",
        *,
        use_subtests: bool = False,
    ):
        leaves1 = jax.tree.leaves_with_path(tree1)
        leaves2 = jax.tree.leaves_with_path(tree2)
        # flat1 = tree.flatten(val1)
        # flat_params2 = tree.flatten(val2)
        tree.assert_same_structure(leaves1, leaves2)
        path1: jax.tree_util.KeyPath
        leaf: LeafType
        for (path1, val1), (path2, val2) in zip(leaves1, leaves2):
            with self.subTest(msg=msg, attr=attr_checked, path=path1) if use_subtests else nullcontext():
                self.assertEqual(path1, path2, msg)
                if path1:  # empty tuple for top-level attributes
                    leaf = path1[-1]
                    leaf_name = get_leafpath_value(leaf)
                    if leaf_name in ignore_leaves:
                        continue
                npt.assert_array_equal(
                    val1, val2, err_msg=f"Attribute '{attr_checked}.{path1}' not equal in both states {msg}"
                )

    def compare_env_runner_results(
        self,
        metrics_0: dict[str, Any],
        metrics_1: dict[str, Any],
        msg: str | None = None,
        *,
        strict: bool = False,
        compare_results: bool | None = None,
        compare_steps_sampled: bool | None = None,
        ignore: Collection[str] = (),
        seed_subset_ok=False,
    ):
        """
        Args:
            metrics_0: Metrics from the first env_runner.
            metrics_1: Metrics from the second env_runner.
            msg: Optional message to display on failure.
            strict: If True, all keys must match exactly, otherwise only common keys are compared.
                For restored metrics the min/max/mean keys might be different
            compare_results: If False, some keys are ignored in the comparison.
            compare_steps_sampled: If num_episodes and steps sampled should be compared.
        """
        key_difference = set(metrics_0.keys()).symmetric_difference(metrics_1.keys())
        # print("Key differences for metrics:", sorted(key_difference))
        same_keys = set(metrics_0.keys()).intersection(metrics_1.keys())
        all_keys = same_keys | key_difference
        if not strict:
            all_keys.discard("env_to_module_sum_episodes_length_in")  # might be wrong due to restore
            all_keys.discard("env_to_module_sum_episodes_length_out")
            all_keys.difference_update(key_difference)
        # remove timer stats
        all_keys = {k for k in all_keys if not k.endswith("_throughput")}
        if compare_results is None:
            compare_results = strict
        if compare_steps_sampled is None:
            compare_steps_sampled = strict
        if not compare_steps_sampled:
            all_keys.discard(NUM_EPISODES)
            all_keys.discard(NUM_ENV_STEPS_SAMPLED)
            all_keys.discard(NUM_ENV_STEPS_SAMPLED_LIFETIME)
            all_keys.discard(NUM_MODULE_STEPS_SAMPLED)
            all_keys.discard(NUM_MODULE_STEPS_SAMPLED_LIFETIME)
            all_keys.discard(NUM_AGENT_STEPS_SAMPLED)
            all_keys.discard(NUM_AGENT_STEPS_SAMPLED_LIFETIME)
        if not compare_results:
            all_keys.discard("agent_episode_returns_mean")  # <2.48
            all_keys.discard("agent_episode_return_mean")  # 2.48
            all_keys.discard("module_episode_returns_mean")  # <2.48
            all_keys.discard("module_episode_return_mean")  # 2.48
            all_keys.discard("episode_len_max")
            all_keys.discard("episode_len_min")
            all_keys.discard("episode_len_mean")
            all_keys.discard("episode_return_max")
            all_keys.discard("episode_return_min")
            all_keys.discard("episode_return_mean")
            all_keys.discard("num_episodes_lifetime")  # needs same sampling
        all_keys.discard("num_episodes_lifetime")  # Remove because of metrics restore bug # 54324
        all_keys.difference_update(ignore)
        self.set_max_diff(None)
        # compare nan values, some int values might be (not) cast to float
        self.assertEqual(
            {k: math.isnan(v) for k in all_keys if isinstance(v := metrics_0[k], (float, int))},
            {k: math.isnan(v) for k in all_keys if isinstance(v := metrics_1[k], (float, int))},
            msg=(msg or "") + f" NaN values differ: {metrics_0}\n!=\n{metrics_1} {msg}",
        )
        # not nans
        if ENVIRONMENT_RESULTS in metrics_0:
            metrics_0 = deepcopy(metrics_0)
            metrics_1 = deepcopy(metrics_1)
            seeds_data0: dict[str, Iterable[int]] = metrics_0[ENVIRONMENT_RESULTS]["seeds"]
            seeds_data1: dict[str, Iterable[int]] = metrics_1[ENVIRONMENT_RESULTS]["seeds"]
            seq0 = list(seeds_data0.pop("seed_sequence"))  # A
            seq1 = list(seeds_data1.pop("seed_sequence"))  # A B
            seeds0 = set(seq0)
            seeds1 = set(seq1)
            # when having multiple env runners the logged seeds in the restored one are merged
            # and lack the total length
            self.assertEqual(len(seq0), len(seeds0), f"Seeds are not unique: {seq0}")
            self.assertEqual(len(seq1), len(seeds1), f"Seeds are not unique: {seq1}")
            if seed_subset_ok:
                self.assertTrue(
                    seq0 <= seq1 or seq0 >= seq1,
                    f"One seed sequences should be a subset of the other: {seq0} vs {seq1}",
                )
            else:
                # num_env_runners > 1 order might be different A B vs. B A. Still compare as sets
                self.assertSetEqual(
                    seeds0,
                    seeds1,
                    f"Seed sequences do not match: {seq0} vs {seq1}",
                )
        if len(all_keys) == 0:
            logger.warning("No keys to compare.")

        self.assertDictEqual(
            {k: v for k in all_keys if not (isinstance(v := metrics_0[k], float) and math.isnan(v))},
            {k: v for k in all_keys if not (isinstance(v := metrics_1[k], float) and math.isnan(v))},
            msg=msg,
        )

    def util_test_state_equivalence(
        self,
        state1: TrainState | Any,
        state2: TrainState | Any,
        msg="",
        *,
        ignore: Collection[str] = (),
        ignore_leaves: Collection[str] = (),
    ):
        """Check if two JAX Train States states are equivalent."""
        # Check if the parameters and indices are equal
        if isinstance(ignore, str):
            ignore = {ignore}
        else:
            ignore = set(ignore)
        if isinstance(ignore_leaves, str):
            ignore_leaves = {ignore_leaves}
        else:
            ignore_leaves = set(ignore_leaves)

        for attr in ["params", "indices", "grad_accum", "opt_state"]:
            if attr in ignore:
                continue
            with self.subTest(msg=msg, attr=attr):
                attr1 = getattr(state1, attr, None)
                attr2 = getattr(state2, attr, None)
                self.assertEqual(
                    attr1 is not None, attr2 is not None, f"Attribute {attr} not found in both states {msg}"
                )
                if attr1 is None and attr2 is None:
                    continue
                self.util_test_tree_equivalence(attr1, attr2, ignore_leaves=ignore_leaves, msg=msg, attr_checked=attr)

        # Check if the other attributes are equal
        for attr in set(dir(state1) + dir(state2)) - ignore:
            if not attr.startswith("_") and attr not in [
                "params",
                "indices",
                "grad_accum",
                "opt_state",
                "apply_gradients",
                "tx",
                "replace",
            ]:
                attr1 = getattr(state1, attr, None)
                attr2 = getattr(state2, attr, None)
                self.assertEqual(
                    attr1 is not None, attr2 is not None, f"Attribute '{attr}' not found in both states {msg}"
                )
                comp = attr1 == attr2
                if isinstance(comp, bool):
                    self.assertTrue(comp, f"Attribute '{attr}' not equal in both states: {attr1}\n!=\n{attr2}\n{msg}")
                elif hasattr(comp, "all"):  # numpy, tensors, ...
                    self.assertTrue(
                        comp.all(), f"Attribute '{attr}' not equal in both states: {attr1}\n!=\n{attr2}\n{msg}"
                    )
                else:
                    self.assertTrue(comp, f"Attribute '{attr}' not equal in both states: {attr1}\n!=\n{attr2}\n{msg}")

        # NOTE: Apply gradients modifies state

    def compare_metrics_in_results(
        self,
        result1: Mapping,
        result2: Mapping,
        expected: float | Iterable[Any],
        metrics: Collection[str],
        msg: str | None = None,
    ):
        """Check that the metrics in both results are equal."""
        if not isinstance(expected, Iterable):
            expected = [expected] * len(metrics)  # same result
        for expected_value, metric in zip(expected, metrics):
            self.assertIn(metric, result1)
            self.assertIn(metric, result2)
            with self.subTest(msg.format(metric), metric=metric):
                self.assertEqual(
                    result1[metric],
                    result2[metric],
                )
                self.assertEqual(
                    result1[metric],
                    expected_value,
                    f"Expected {expected_value} for metric '{metric}', but got {result1[metric]}",
                )

    @staticmethod
    def filter_incompatible_remote_config(config: dict[str, Any]) -> dict[str, Any]:
        if "tf_session_args" in config:
            config["tf_session_args"]["inter_op_parallelism_threads"] = "removed_key_for_test"
            config["tf_session_args"]["intra_op_parallelism_threads"] = "removed_key_for_test"
            for key in (k for k, v in config.items() if "callbacks" in k and callable(v)):
                config[key] = (
                    config[key].__name__
                    if hasattr(config[key], "__name__")
                    else type(config[key]).__name__
                    if not isinstance(config[key], type)
                    else config[key].__name__
                )
        return config

    def compare_weights(
        self,
        weights1: dict[str, Any],
        weights2: dict[str, Any],
        msg: str = "",
        ignore: Collection[str] = (),
        *,
        almost: bool = False,
    ):
        keys1 = set(weights1.keys()) - set(ignore)
        keys2 = set(weights2.keys()) - set(ignore)
        self.assertEqual(keys1, keys2, f"Keys in weights do not match: {msg}")
        for key, w1 in weights1.items():
            if key in ignore:
                continue
            self.assertEqual(type(w1), type(weights2[key]), f"Weight '{key}' type does not match: {msg}")
            if isinstance(weights2[key], dict) and isinstance(w1, dict):
                self.compare_weights(w1, weights2[key], f"Weight '{key}' does not match: {msg}", almost=almost)
                continue
            if isinstance(w1, str):  # if other structures are present
                self.assertEqual(w1, weights2[key], f"Key '{key}' not equal in both states {msg}")
                continue
            if isinstance(w1, list) and any(isinstance(x, dict) for x in w1):
                # If list contains dicts, compare dicts
                try:
                    self.assertListEqual(w1, weights2[key], f"Key '{key}' not equal in both states {msg}")
                except (ValueError, AssertionError):  # could be almost equal
                    self.assertEqual(len(w1), len(weights2[key]), f"Key '{key}' not equal in both states {msg}")
                    for i, item in enumerate(w1):
                        self.compare_weights(
                            item, weights2[key][i], f"Key '{key}[{i}]' not equal in both states {msg}", almost=almost
                        )
                    continue
                else:
                    continue
            if w1 is None:
                self.assertIsNone(weights2[key], f"Key '{key}' not equal in both states {msg}")
                continue
            if almost:
                # Use almost equal for floats, arrays, etc.
                npt.assert_array_almost_equal(
                    w1,
                    weights2[key],
                    err_msg=f"Key '{key}' not equal in both states {msg}",
                )
            else:
                npt.assert_array_equal(
                    w1,
                    weights2[key],
                    err_msg=f"Key '{key}' not equal in both states {msg}",
                )

    def compare_env_runner_configs(self, algo: Algorithm, algo_restored: Algorithm, *, ignore_overrides_key=True):
        self.set_max_diff(self.maxDiff and max(self.maxDiff or 0, 20000))

        def assertCleanDictEqual(a, b, *args, **kwargs):  # noqa: N802
            __tracebackhide__ = True
            self.assertDictEqual(
                self.filter_incompatible_remote_config(a), self.filter_incompatible_remote_config(b), *args, **kwargs
            )

        algo_config_dict = algo.config.to_dict()
        algo_restored_config_dict = algo_restored.config.to_dict()
        if ignore_overrides_key:
            assertCleanDictEqual(
                {k: v for k, v in algo_restored_config_dict.items() if k != "_restored_overrides"}, algo_config_dict
            )
        else:
            assertCleanDictEqual(algo_restored_config_dict, algo_config_dict)
        assert algo.config
        if algo.config.num_env_runners == 0:
            self.assertEqual(algo_restored.config.num_env_runners, 0)  # pyright: ignore[reportOptionalMemberAccess]
            assertCleanDictEqual(
                (algo.env_runner.config.to_dict()),
                algo_config_dict,  # pyright: ignore[reportOptionalMemberAccess]
            )
            restored_env_runner_config_dict = algo_restored.env_runner.config.to_dict()
            assertCleanDictEqual(restored_env_runner_config_dict, algo_restored_config_dict)
            if ignore_overrides_key:
                assertCleanDictEqual(
                    {k: v for k, v in restored_env_runner_config_dict.items() if k != "_restored_overrides"},
                    algo_config_dict,
                )
            else:
                assertCleanDictEqual(algo_config_dict, restored_env_runner_config_dict)

        remote_configs = algo.env_runner_group.foreach_env_runner(lambda r: r.config.to_dict(), local_env_runner=False)
        local_config = algo.env_runner_group.local_env_runner.config.to_dict()

        # Possible ignore local env_runner here when using remotes
        for i, config in enumerate((local_config, *remote_configs), start=1):
            assertCleanDictEqual(
                config,
                algo_config_dict,
                ("Local config" if config is local_config else "Remote config")
                + f" {i}/{len(remote_configs)} does not match algo config",
            )
        remote_configs_restored = algo_restored.env_runner_group.foreach_env_runner(
            lambda r: r.config.to_dict(), local_env_runner=False
        )
        local_config_restored = algo_restored.env_runner_group.local_env_runner.config.to_dict()
        for i, config in enumerate((local_config_restored, *remote_configs_restored), start=1):
            if ignore_overrides_key:
                algo_restored_config_dict.pop("_restored_overrides", None)
                config.pop("_restored_overrides", None)
            assertCleanDictEqual(
                config,
                algo_restored_config_dict,
                ("Local config" if config is local_config_restored else "Remote config")
                + f" {i}/{len(remote_configs_restored) + 1} does not match restored config",
            )
            assertCleanDictEqual(
                config,
                algo_config_dict,
                ("Local config" if config is local_config_restored else "Remote config")
                + f" {i}/{len(remote_configs_restored) + 1} does not match algo config",
            )

    def compare_configs(
        self, config1: AlgorithmConfig | dict, config2: AlgorithmConfig | dict, *, ignore: Collection[str] = ()
    ):
        config1_eval = None
        config2_eval = None
        if isinstance(config1, AlgorithmConfig):
            if config1.evaluation_config:
                config1_eval = config1.evaluation_config
            config1 = config1.to_dict()
        else:
            config1 = config1.copy()
        if isinstance(config2, AlgorithmConfig):
            config2_eval = config2.evaluation_config
            config2 = config2.to_dict()
        else:
            config2 = config2.copy()
        # cleanup
        if ignore:
            for key in ignore:
                config1.pop(key, None)
                config2.pop(key, None)
        # remove class
        config1.pop("class", None)
        config2.pop("class", None)
        # Is set to False for one config, OldAPI value
        config1.pop("simple_optimizer", None)
        config2.pop("simple_optimizer", None)
        self.assertDictEqual(config1, config2)  # ConfigType
        if config1_eval or config2_eval:
            if not config1_eval or not config2_eval:
                self.fail("One of the configs has no evaluation_config")
            with self.subTest("Compare evaluation configs"):
                self.compare_configs(config1_eval, config2_eval, ignore=ignore)

    def _compare_metrics_logger_states(
        self,
        state1,
        state2,
        *,
        key: str,
        ignore_timers: bool = False,  # , ignore_episode_stats: bool = False
    ):
        """
        Tensors get their values removed on set_state, furthermore cannot compare nan==nan

        When having a small batch_size, one trainable might have completed an episode
        which results in additional keys in the metrics_logger state, e.g. 'env_runners--episode_return_mean'
        """

        def not_a_timer_key(k: str) -> bool:
            return "timer" not in k and "duration" not in k and not k.endswith(("_throughput", "env_runners--sample"))

        if ignore_timers:
            state1 = _remove_throughput_stats({k: v for k, v in state1.items() if not_a_timer_key(k)})
            state2 = _remove_throughput_stats({k: v for k, v in state2.items() if not_a_timer_key(k)})
        else:
            # when having >= 2 env_runners get_state will store two values, set_state however uses peek
            # and saves only one value in the new state.
            state1 = _fix_throughput_stats(state1)
            state2 = _fix_throughput_stats(state2)

        # With a small batch_size one trainable might have completed an episode an additional keys

        self.assertDictEqual(
            nan_to_zero_hist_leaves(
                _remove_values_on_tensor_stats(
                    state1,
                ),
                remove_all=True,
            ),
            nan_to_zero_hist_leaves(
                _remove_values_on_tensor_stats(state2),
                remove_all=True,
            ),
            f"Algorithm state['{key}']"
            + ("['metrics_logger']" if "metrics_logger" not in key else "")
            + " in checkpoint differs from current algorithm state.",
        )

    def compare_algorithm_states(
        self,
        algorithm_state1: dict,
        algorithm_state2: dict,
        *,
        ignore_timers: bool = False,
        ignore_env_runner_state: bool = False,
        # ignore_multiple_throughput_stats: bool = True,
    ):
        for key in algorithm_state1.keys() | algorithm_state2.keys():
            if isinstance(algorithm_state1[key], dict) and isinstance(
                algorithm_state2[key], dict
            ):  # Check if both are dicts
                if key == "metrics_logger":
                    # Special handling for metrics_logger
                    self._compare_metrics_logger_states(
                        algorithm_state1[key]["stats"],
                        algorithm_state2[key]["stats"],
                        key=key,
                        ignore_timers=ignore_timers,
                    )
                elif key == "learner_group":
                    # currently learner only key, but forward compatible
                    self.assertDictEqual(
                        {k: v for k, v in algorithm_state2[key].items() if k not in ("learner")},
                        {k: v for k, v in algorithm_state1[key].items() if k not in ("learner")},
                    )
                    checkpoint_learner_state = algorithm_state2[key]["learner"]
                    loaded_learner_state = algorithm_state1[key]["learner"]
                    self.assertDictEqual(
                        {
                            k: v
                            for k, v in checkpoint_learner_state.items()
                            if k not in ("metrics_logger", "optimizer", "rl_module")
                        },
                        {
                            k: v
                            for k, v in loaded_learner_state.items()
                            if k not in ("metrics_logger", "optimizer", "rl_module")
                        },
                    )
                    self.compare_weights(
                        checkpoint_learner_state["rl_module"],
                        loaded_learner_state["rl_module"],
                        f"Algorithm state[{key}]['learner']['rl_module'] in checkpoint "
                        "differs from current algorithm state.",
                    )
                    # NOTE about inequality:
                    # As `values` could contain tensors they are not reinstated on set_state,
                    # but saved as such in the checkpoint

                    self._compare_metrics_logger_states(
                        checkpoint_learner_state["metrics_logger"]["stats"],
                        loaded_learner_state["metrics_logger"]["stats"],
                        key=f"{key}]['learner'",
                        ignore_timers=ignore_timers,
                    )
                    self.compare_weights(
                        checkpoint_learner_state["optimizer"],
                        loaded_learner_state["optimizer"],
                        f"Algorithm state[{key}]['learner']['optimizer'] in checkpoint "
                        "differs from current algorithm state.",
                        almost=True,
                    )
                else:
                    try:
                        self.assertDictEqual(
                            algorithm_state2[key],
                            algorithm_state1[key],
                            f"Algorithm state[{key}] in checkpoint differs from current algorithm state.",
                        )
                    except ValueError as e:
                        print("Cannot compare dicts for key %s: %s" % (key, e))
                        raise
                    except AssertionError:
                        if key == "eval_env_runner":
                            print(
                                "eval_env_runner state differs, ignoring for now. "
                                "This is due to Ray syncing the eval worker with the train worker on set_state"
                            )
                            continue
                        if key == "env_runner" and ignore_env_runner_state:
                            print("env_runner states differ, but ignoring as ignore_env_runner_state is True")
                            continue
                        raise
            else:
                self.assertEqual(
                    algorithm_state2[key],
                    algorithm_state1[key],
                    f"Algorithm state[{key}] in checkpoint differs from current algorithm state.",
                )

    def compare_trainable_state(
        self,
        state1: dict[str, Any] | TrainableStateDict,
        state2: dict[str, Any] | TrainableStateDict,
        *,
        ignore_env_runner_state: bool = True,
        ignore_timers: bool = False,
        msg: str = "",
    ):
        """
        Args:
            state1: First state to compare.
            state2: Second state to compare.
            ignore_env_runner_state: If True, do not compare the env_runner state.
                This might need to be chosen when using num_env_runners > 0 as the original
                local env_runner (that is contained in get_state) is not in sync with the remote
                env_runner that we care about. On restore the local env_runner is updated with
                necessary states of the former remote env runners. Hence, they do not align.
            ignore_timers: If True, do not compare the timers in the state.
        """
        try:
            self.assertDictEqual(state1, state2, msg=msg)
        except (ValueError, AssertionError):  # ValueError because of Numpy, AssertionError for Distributions
            pass
        else:
            return
        state1 = state1.copy()
        state2 = state2.copy()
        self.compare_algorithm_states(
            state1.pop("algorithm", {}),
            state2.pop("algorithm", {}),
            ignore_timers=ignore_timers,
            ignore_env_runner_state=ignore_env_runner_state,
        )
        self.compare_configs(state1.pop("algorithm_config"), state2.pop("algorithm_config"))  # pyright: ignore[reportArgumentType]
        self.compare_pbar_state(state1.pop("pbar_state"), state2.pop("pbar_state"))  # pyright: ignore[reportArgumentType]
        setup_state1: dict[str, Any] = state1.pop("setup").copy()  # pyright: ignore[reportAttributeAccessIssue]
        setup_state2: dict[str, Any] = state2.pop("setup").copy()  # pyright: ignore[reportAttributeAccessIssue]

        self.compare_param_space(setup_state1.pop("param_space"), setup_state2.pop("param_space"))
        self.compare_configs(setup_state1.pop("config"), setup_state2.pop("config"))
        trainable_state1 = state1.pop("trainable", {}).copy()  # pyright: ignore[reportAttributeAccessIssue]
        trainable_state2 = state2.pop("trainable", {}).copy()  # pyright: ignore[reportAttributeAccessIssue]
        if ignore_timers:
            trainable_state1.pop("time_total")
            trainable_state2.pop("time_total")
            last_result1 = {
                k: v
                for k, v in trainable_state1.pop("last_result", {}).items()
                if k not in ("date", "time_since_restore", "time_this_iter_s", "time_total_s", "timestamp")
            }
            last_result2 = {
                k: v
                for k, v in trainable_state2.pop("last_result", {}).items()
                if k not in ("date", "time_since_restore", "time_this_iter_s", "time_total_s", "timestamp")
            }
        else:
            last_result1 = trainable_state1.pop("last_result", {})
            last_result2 = trainable_state2.pop("last_result", {})
        last_result1.pop("pid", None)
        last_result2.pop("pid", None)
        self.assertDictEqual(last_result1, last_result2, f"Last result in trainable state differs: {msg}")
        self.assertDictEqual(trainable_state1, trainable_state2, f"Trainable state differs: {msg}")

        self.assertDictEqual(
            nan_to_zero_hist_leaves(state1, key=None, remove_all=True, replace="NaN"),
            nan_to_zero_hist_leaves(state2, key=None, remove_all=True, replace="NaN"),
        )
        self.assertDictEqual(setup_state1, setup_state2)

    def compare_param_space(self, param_space1: dict[str, Any], param_space2: dict[str, Any]):
        if param_space1 == {"__params_not_created__": True} or param_space2 == {"__params_not_created__": True}:
            self.assertTrue(param_space1 == param_space2)
            return
        self.assertCountEqual(param_space1, param_space2)
        self.assertEqual(param_space1.keys(), param_space2.keys())
        self.assertDictEqual(param_space1["cli_args"], param_space2["cli_args"])
        for key in param_space1.keys():  # noqa: PLC0206
            value1 = param_space1[key]
            value2 = param_space2[key]
            if isinstance(value1, Domain) or isinstance(value2, Domain):
                # Domain is not hashable, so we cannot compare them directly
                self.assertIs(type(value1), type(value2))
                if isinstance(value1, Categorical):
                    assert isinstance(value2, Categorical)
                    self.assertListEqual(value1.categories, value2.categories)
                elif isinstance(value1, (Integer, Float)):
                    assert isinstance(value2, type(value1))
                    self.assertEqual(value1.lower, value2.lower)
                    self.assertEqual(value1.upper, value2.upper)
                else:
                    # This will likely fail, need to compare attributes
                    try:
                        self.assertEqual(value1, value2, f"Domain {key} differs: {value1} != {value2}")
                    except AssertionError:
                        self.assertDictEqual(
                            value1.__dict__, value2.__dict__, f"Domain {key} differs: {value1} != {value2}"
                        )
            else:
                self.assertEqual(value1, value2, f"Parameter {key} differs: {value1} != {value2}")

    def compare_pbar_state(self, state1: dict[str, Any] | tuple[Any, ...], state2: dict[str, Any] | tuple[Any, ...]):
        if isinstance(state1, dict):
            self.assertIsInstance(state2, dict, "Both states should be dicts")
            state1 = {k: v for k, v in state1.items() if k not in ("desc", "uuid")}
            state2 = {k: v for k, v in state2.items() if k not in ("desc", "uuid")}  # pyright: ignore[reportAttributeAccessIssue]
        self.assertEqual(state1, state2)

    def compare_trainables(
        self,
        trainable: TrainableBase["DefaultArgumentParser", "ConfigType_co", "AlgorithmType_co"],
        trainable2: TrainableBase["DefaultArgumentParser", "ConfigType_co", "AlgorithmType_co"],
        msg: str = "",
        *,
        ignore_env_runner_state: bool = True,
        ignore_timers: bool = False,
        iteration_after_step=2,
        minibatch_size=32,
        ignore_restored_overrides_key=True,
        **subtest_kwargs,
    ) -> None:
        """
        Test functions for trainables obtained in different ways

        Args:
            trainable: The original trainable or one variant.
            trainable2: The trainable to compare with the original.
            ignore_env_runner_state: If True, do not compare the env_runner key in get_state()
                For num_env_runners > 0 the key of the original trainable does not reflect the remote
                set to False to avoid this False positive in that case. For more see compare_trainable_state.
            iteration_after_step: The expected iteration after the step.
            minibatch_size: The expected minibatch size.
            subtest_kwargs: passed to the subtest context.
            ignore_restored_overrides_key: Ignore the "_restored_overrides" key in the config dict comparison.

        Attention:
            Does perform a step on each trainable
        """
        if trainable.algorithm_config.train_batch_size_per_learner != make_divisible(
            trainable.algorithm_config.train_batch_size_per_learner, minibatch_size
        ):
            logger.warning(
                "compare_trainables: Trainable.train_batch_size_per_learner %d is not divisible "
                "by argument minibatch_size %d. If this test fails pass the divisible minibatch_size as an argument.",
                trainable.algorithm_config.train_batch_size_per_learner,
                minibatch_size,
            )
        self.maxDiff = None
        with self.subTest("Step 1: Compare trainables " + msg, **subtest_kwargs):
            if hasattr(trainable, "_args") or hasattr(trainable2, "_args"):
                self.assertDictEqual(trainable2._args, trainable._args)  # type: ignore[attr-defined]
            self.assertEqual(
                trainable.algorithm_config.minibatch_size, minibatch_size
            )  # <-- passed as divisible or modified and != argument?  # noqa: E501
            self.assertEqual(trainable2.algorithm_config.minibatch_size, trainable.algorithm_config.minibatch_size)
            self.assertEqual(trainable2._iteration, trainable._iteration)

            # get_state stores "class" : type(self) of the config, this allows from_state to work correctly
            # original trainable does not have that key
            config_dict1 = trainable.algorithm_config.to_dict()
            config_dict1.pop("class", None)
            config_dict2 = trainable2.algorithm_config.to_dict()
            config_dict2.pop("class", None)
            if ignore_restored_overrides_key:
                self.assertDictEqual(
                    {k: v for k, v in config_dict2.items() if k != "_restored_overrides"},
                    {k: v for k, v in config_dict1.items() if k != "_restored_overrides"},
                )
            else:
                self.assertDictEqual(config_dict2, config_dict1)
            setup_data1 = trainable._setup.get_state()  # does not compare setup itself
            setup_data2 = trainable2._setup.get_state()
            # check all keys
            self.assertEqual(setup_data1.keys(), setup_data2.keys())
            keys = set(setup_data1.keys())
            keys.remove("__init_config__")
            self.assertDictEqual(vars(setup_data1["args"]), vars(setup_data2["args"]))  # SimpleNamespace
            keys.remove("args")
            self.assertIs(setup_data1["setup_class"], setup_data2["setup_class"])
            keys.remove("setup_class")
            assert setup_data1["config"] and setup_data2["config"]
            self.compare_configs(setup_data1["config"], setup_data2["config"])
            # Check that num_env runners matches config
            self.assertEqual(
                trainable.algorithm_config.num_env_runners,
                trainable.algorithm.env_runner_group.num_remote_env_runners(),
            )
            self.assertEqual(
                trainable2.algorithm_config.num_env_runners,
                trainable2.algorithm.env_runner_group.num_remote_env_runners(),
            )
            if trainable2.algorithm_config.num_env_runners != trainable.algorithm_config.num_env_runners:
                logger.warning("num_env_runners are different for the two trainables. Check if this is intended.")
            trainable2_eval_config = trainable2.algorithm_config.get_evaluation_config_object()
            trainable_eval_config = trainable.algorithm_config.get_evaluation_config_object()
            self.assertIs(
                type(trainable_eval_config),
                type(trainable2_eval_config),
                f"Evaluation config types do not match: {type(trainable_eval_config)} != {type(trainable2_eval_config)}",
            )
            if trainable2_eval_config:
                self.assertEqual(
                    trainable2_eval_config.num_env_runners,
                    trainable2.algorithm.eval_env_runner_group.num_remote_env_runners(),
                )
                assert trainable_eval_config
                if trainable2_eval_config.num_env_runners != trainable_eval_config.num_env_runners:
                    logger.warning("num_env_runners are different for the two trainables. Check if this is intended.")

            keys.remove("config")
            self.assertDictEqual(setup_data1["config_overrides"], setup_data2["config_overrides"])
            keys.remove("config_overrides")
            param_space1 = setup_data1["param_space"]
            param_space2 = setup_data2["param_space"]
            keys.remove("param_space")
            if "trial_name_creator" in setup_data1 and "trial_name_creator" in setup_data2:
                self.assertEqual(setup_data1["trial_name_creator"], setup_data2["trial_name_creator"])
            elif "trial_name_creator" in setup_data1 or "trial_name_creator" in setup_data2:
                self.fail("One of the trainables has a trial_name_creator, the other does not.")
            keys.remove("trial_name_creator")

            self.assertListEqual(setup_data1.get("config_files") or [], setup_data2.get("config_files") or [])
            keys.remove("config_files")

            self.assertEqual(len(keys), 0, f"Unchecked keys: {keys}")  # checked all params
            self.compare_param_space(param_space1, param_space2)  # pyright: ignore[reportArgumentType]

            # Compare attrs
            self.assertIsNot(trainable2._reward_updaters, trainable._reward_updaters)
            for key in trainable2._reward_updaters.keys() | trainable._reward_updaters.keys():
                updater1 = trainable._reward_updaters[key]
                updater2 = trainable2._reward_updaters[key]
                self.assertIsNot(updater1, updater2)
                assert isinstance(updater1, partial) and isinstance(updater2, partial)
                self.assertDictEqual(updater1.keywords, updater2.keywords)
                self.assertIsNot(updater1.keywords["reward_array"], updater2.keywords["reward_array"])

            self.assertIsNot(trainable2._pbar, trainable._pbar)
            self.assertIs(type(trainable2._pbar), type(trainable._pbar))
            if isinstance(trainable2._pbar, tqdm_ray.tqdm):
                self.compare_pbar_state(trainable._pbar._get_state(), trainable2._pbar._get_state())  # pyright: ignore[reportAttributeAccessIssue]

            # Compare states
            state1: dict = nan_to_zero_hist_leaves(trainable.get_state(), key=None, remove_all=True)
            state2: dict = nan_to_zero_hist_leaves(trainable2.get_state(), key=None, remove_all=True)
            if ignore_restored_overrides_key:
                state1.pop("_restored_overrides", None)
                state2.pop("_restored_overrides", None)
                if "algorithm" in state1 and "algorithm" in state2:
                    state1["algorithm"]["config"].pop("_restored_overrides", None)
                    state2["algorithm"]["config"].pop("_restored_overrides", None)
                state1["algorithm_config"].pop("_restored_overrides", None)
                state2["algorithm_config"].pop("_restored_overrides", None)
            self.compare_trainable_state(
                state1,
                state2,
                msg=msg,
                ignore_env_runner_state=ignore_env_runner_state,
                ignore_timers=ignore_timers,
            )

            # Step 2
            result2 = trainable.train()
            result2_restored = trainable2.train()
            self.assertEqual(
                trainable2.algorithm_config.get_rollout_fragment_length(),
                trainable.algorithm_config.get_rollout_fragment_length(),
            )
            self.assertTrue(trainable.algorithm.env_runner is not None)
            self.assertTrue(trainable2.algorithm.env_runner is not None)
            # NOTE: If num_env_runners is not explicitly set for trainable2 it will have 0 env runners the args default!
            # Therefore compare only last env_runner in list (might be the local one)
            self.assertEqual(
                trainable2.algorithm.env_runner_group.foreach_env_runner(
                    lambda r: (
                        r.config.get_rollout_fragment_length(),
                        r.config.total_train_batch_size,
                        r.config.train_batch_size_per_learner,
                        r.config.num_envs_per_env_runner
                        == r.num_envs,  # This is False for the local env runner # pyright: ignore[reportAttributeAccessIssue]
                        r.config.get_rollout_fragment_length() * r.num_envs,  # pyright: ignore[reportAttributeAccessIssue]
                    )
                )[-1],
                trainable.algorithm.env_runner_group.foreach_env_runner(
                    lambda r: (
                        r.config.get_rollout_fragment_length(),
                        r.config.total_train_batch_size,
                        r.config.train_batch_size_per_learner,
                        r.config.num_envs_per_env_runner == r.num_envs,  # pyright: ignore[reportAttributeAccessIssue]
                        r.config.get_rollout_fragment_length() * r.num_envs,  # pyright: ignore[reportAttributeAccessIssue]
                    )
                )[-1],
            )
            self.assertEqual(
                trainable2.algorithm_config.total_train_batch_size,
                trainable.algorithm_config.total_train_batch_size,
            )
            self.assertEqual(
                trainable2.algorithm_config.total_train_batch_size,
                trainable.algorithm_config.total_train_batch_size,
            )
            self.assertEqual(result2[TRAINING_ITERATION], result2_restored[TRAINING_ITERATION], msg)
            self.assertEqual(result2[TRAINING_ITERATION], iteration_after_step, msg)
            self.assertEqual(result2["current_step"], result2_restored["current_step"])
            self.compare_env_runner_results(
                result2_restored[ENV_RUNNER_RESULTS],  # pyright: ignore[reportArgumentType]
                result2[ENV_RUNNER_RESULTS],  # pyright: ignore[reportArgumentType]
                "results after step 2",
                compare_steps_sampled=False,
                compare_results=False,
            )

        # Compare env_runners
        self.maxDiff = None
        with self.subTest("Step 2 Compare env_runner configs " + msg, **subtest_kwargs):
            if trainable.algorithm.env_runner or trainable2.algorithm.env_runner:
                assert trainable.algorithm.env_runner and trainable2.algorithm.env_runner
                self.compare_env_runner_configs(
                    trainable.algorithm,
                    trainable2.algorithm,
                    ignore_overrides_key=ignore_restored_overrides_key,
                )

    # endregion
    # region utilities

    @staticmethod
    def get_checkpoint_dirs(result: Result) -> tuple[pathlib.Path, list[str]]:
        """Returns checkpoint dir of the result and found saved checkpoints"""
        assert result.checkpoint is not None
        checkpoint_dir, file = os.path.split(result.checkpoint.path)
        return pathlib.Path(checkpoint_dir), [
            os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")
        ]

    @classmethod
    def clean_timer_logs(cls, result: dict):
        """
        Cleans the timer logs from the env_runners_dict.
        This is useful to compare results without the timer logs.
        """
        result = deepcopy(result)
        env_runners_dict = result[ENV_RUNNER_RESULTS]
        result.pop(TIMERS, None)
        for key in list(env_runners_dict.keys()):
            if key.startswith("timer") or key.endswith("timer"):
                del env_runners_dict[key]
        del env_runners_dict["module_to_env_connector"]
        del env_runners_dict["env_to_module_connector"]
        env_runners_dict.pop("episode_duration_sec_mean", None)
        del env_runners_dict["sample"]
        del env_runners_dict["time_between_sampling"]
        # possibly also remove 'env_to_module_sum_episodes_length_in' which differ greatly
        env_runners_dict.pop("num_env_steps_sampled_lifetime_throughput", None)
        # result[ENV_RUNNER_RESULTS]["time_between_sampling"]
        if LEARNER_RESULTS in result:  # not for evaluation
            learner_dict = result[LEARNER_RESULTS]
            learner_all_modules = learner_dict[ALL_MODULES]
            # learner_default_policy = learner_all_modules[DEFAULT_POLICY_ID]
            del learner_all_modules["learner_connector"]
        evaluation_dict = result.get("evaluation", {})
        if not evaluation_dict:
            return result
        result["evaluation"] = cls.clean_timer_logs(evaluation_dict)
        return result

    def check_stopper_added(
        self,
        tuner: tune.Tuner,
        stopper_type: Mapping | type[tune.Stopper] | None,
        *,
        check: Callable[[StopperType, Mapping | type[tune.Stopper] | Any | None], bool] = lambda a, b: False,  # noqa: ARG005
        need_match: bool = True,
    ) -> StopperType:
        stoppers: StopperType = tuner._local_tuner.get_run_config().stop
        if stopper_type is None:
            self.assertIsNone(stoppers)
            check(stoppers, stopper_type)
            return None
        if isinstance(stoppers, (dict, Mapping)) or isinstance(stopper_type, (dict, Mapping)):
            self.assertTrue(
                isinstance(stoppers, Mapping) and isinstance(stopper_type, Mapping),
                f"Both types should be a Mapping: {type(stoppers)} !~= {type(stopper_type)}",
            )
            # Fails with TypeError if wrong.
            self.assertEqual(
                stopper_type,
                stopper_type | stoppers,  # pyright: ignore[reportOperatorIssue]
            )
            check(stoppers, stopper_type)
            return stoppers
        if stoppers is None:
            self.assertEqual(stoppers, stopper_type)  # likely fails
            check(stoppers, stopper_type)
            return None
        if not isinstance(stoppers, list):  # here Callable | Stopper
            if isinstance(stoppers, Iterable):
                stopper_list = list(stoppers)
            else:
                stopper_list = [stoppers]
        else:
            stopper_list = stoppers
        while stopper_list:
            stopper = stopper_list.pop()
            if isinstance(stopper, stopper_type):
                check(stopper, stopper_type)
                return stopper
            if isinstance(stopper, CombinedStopper):
                for s in stopper._stoppers:
                    if isinstance(s, stopper_type):
                        check(s, stopper_type)
                        return s
        assert not TYPE_CHECKING or not isinstance(stoppers, Iterable)
        if need_match and check(stoppers, stopper_type) is not True:
            self.fail(f"Did not find stopper of type {stopper_type} in {stoppers}. Check failed")
        return stoppers

    # endregion


class SetupWithEnv(TestHelpers):
    def setUp(self):
        self._env = gym.make("CartPole-v1")
        self._OBSERVATION_SPACE = self._env.observation_space
        self._ACTION_SPACE = self._env.action_space
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self._env.close()


class SetupLowRes(TestHelpers):
    """
    Attributes:
        _DEFAULT_SETUP_LOW_RES: The default low resolution setup.
    Methods:
        _create_low_res_setup: Creates a setup with minimal resources.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # need ray when EnvRunnerGroups are created
        if not ray.is_initialized():
            ray.init(num_cpus=1, log_to_driver=False, include_dashboard=False, object_store_memory=1024**3 // 4)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        ray.shutdown()

    def _create_low_res_setup(self, *args_for_patch, init_trainable=True, **kwargs):
        with (
            change_log_level(experiment_base_logger, logging.ERROR),
            change_log_level(dynamic_buffer_logger, logging.ERROR),
            patch_args(
                "--fcnet_hiddens",
                "[8]",
                "--num_envs_per_env_runner",
                "1",
                *args_for_patch,
            ),
        ):
            setup = MLPSetup(init_trainable=False, **kwargs)
            setup.config.training(train_batch_size_per_learner=64, minibatch_size=32, num_epochs=2).env_runners(
                num_env_runners=0, num_envs_per_env_runner=1, num_cpus_per_env_runner=0, gym_env_vectorize_mode="SYNC"
            ).learners(num_learners=0, num_cpus_per_learner=0)
            if init_trainable:
                setup.create_trainable()
        return setup

    def setUp(self):
        super().setUp()
        self._DEFAULT_SETUP_LOW_RES = self._create_low_res_setup(init_trainable=True)


class SetupDefaults(SetupLowRes, SetupWithEnv, TestHelpers, DisableLoggers):
    @clean_args
    def setUp(self):
        super().setUp()

        self._DEFAULT_NAMESPACE = DefaultArgumentParser()
        self._DEFAULT_NAMESPACE._change_log_level = False
        self._DEFAULT_CONFIG_DICT: MappingProxyType[str, Any] = MappingProxyType(
            self._DEFAULT_NAMESPACE.parse_args().as_dict()
        )
        with (
            change_log_level(experiment_base_logger, logging.ERROR),
            change_log_level(dynamic_buffer_logger, logging.ERROR),
        ):
            self._DEFAULT_SETUP = AlgorithmSetup(init_trainable=False, change_log_level=False)
            self._DEFAULT_SETUP.create_trainable()
        self._INPUT_LENGTH = self._env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
        self._DEFAULT_INPUT = jnp.arange(self._INPUT_LENGTH * 2).reshape((2, self._INPUT_LENGTH))
        self._DEFAULT_BATCH: dict[str, chex.Array] = MappingProxyType({"obs": self._DEFAULT_INPUT})  # pyright: ignore[reportAttributeAccessIssue]
        self._ENV_SAMPLE = jnp.arange(self._INPUT_LENGTH)
        model_key = jax.random.PRNGKey(self._DEFAULT_CONFIG_DICT["seed"] or 2)
        self._RANDOM_KEY, self._ACTOR_KEY, self._CRITIC_KEY = jax.random.split(model_key, 3)
        self._ACTION_DIM: int = self._ACTION_SPACE.n  # pyright: ignore[reportAttributeAccessIssue]
        self._OBS_DIM: int = self._OBSERVATION_SPACE.shape[0]  # pyright: ignore[reportOptionalSubscript]


class DisableGUIBreakpoints(unittest.TestCase):
    _printed_breakpoints: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "GITHUB_REF" in os.environ or (  # no breakpoints on GitHub
            (
                (
                    {"-v", "test*.py"} & set(sys.argv)  # VSCode unittest execution, CLI OK
                    or os.path.split(sys.argv[0])[-1] == "pytest"  # pytest CLI
                )
                and not int(os.environ.get("KEEP_BREAKPOINTS", "0"))  # override
            )
            or int(os.environ.get("DISABLE_BREAKPOINTS", "0"))
        ):
            if not self._printed_breakpoints:
                print("Disabling breakpoints in tests")
                DisableGUIBreakpoints._printed_breakpoints = True
            self._disabled_breakpoints = mock.patch("builtins.breakpoint")
            self._disabled_breakpoints.start()
        else:
            self._disabled_breakpoints = mock.patch("builtins.breakpoint")
            print("enabled breakpoint")


def dict_diff_message(d1: Any, d2: Any) -> str:
    standardMsg = "%s != %s" % unittest.util._common_shorten_repr(d1, d2)
    diff = "\n" + "\n".join(difflib.ndiff(pprint.pformat(d1).splitlines(), pprint.pformat(d2).splitlines()))
    return standardMsg + diff


def format_result_errors(errors):
    return str(errors).replace(r"\n", "\n")


def remote_breakpoint(port=5678):
    """
    A breakpoint implementation that works on remote workers.

    Use VSCode debug Configuration::

        .. code-block:: json

            {
                "name": "Remote Debug Port 5678",
                "type": "debugpy",
                "request": "attach",
                "connect": {
                    "host": "localhost",
                    "port": 5678
                }
            }

    Note:
        Make sure to start the debug server before connecting.
    """
    error = None

    if not debugpy.is_client_connected():
        print("starting debugpy. Listening on port:", port)
        try:
            debugpy.listen(("localhost", port))  # noqa: T100
        except RuntimeError as e:
            error = e
            print(e)
    else:
        print("debugpy already connected, waiting for client")
    if error is None:
        debugpy.wait_for_client()  # noqa: T100
    breakpoint()  # noqa: T100


class _TrainableWithCheckProto(Protocol):
    def setup_check(self, config: dict[str, Any], algorithm_overrides: Optional[dict[str, Any]] = None): ...

    def step_pre_check(self): ...
    def step_post_check(self, result: StrictAlgorithmReturnData, metrics: LogMetricsDict, rewards: Any): ...

    @classmethod
    def define(cls, setup_cls: AlgorithmSetup) -> Any: ...


def SetupWithCheck(check: type["TrainableWithChecks"], base=AlgorithmSetup):  # noqa: N802
    class SetupWithCheck(base):
        _check_class = check

        def _create_trainable(self):
            return self._check_class.define(self)

    return SetupWithCheck


class TrainableWithChecks(DefaultTrainable[Any, "AlgorithmConfig", Any]):
    """
    Debug Variables:
    - debug_setup
    - debug_step
    """

    debug_step: ClassVar[bool] = False
    debug_setup: ClassVar[bool] = False

    def setup_check(self, config: dict[str, Any], algorithm_overrides: Optional[dict[str, Any]] = None):
        pass

    def step_pre_check(self):
        pass

    def step_post_check(self, result: StrictAlgorithmReturnData, metrics: LogMetricsDict, rewards: Any):
        pass

    def setup(self, config, *, algorithm_overrides=None):
        if self.debug_setup:
            print("Start breakpoint")
            remote_breakpoint()
        super().setup(config, algorithm_overrides=algorithm_overrides)
        self.setup_check(config, algorithm_overrides=algorithm_overrides)

    def step(self):
        if self.debug_step:
            print("Start breakpoint")
            remote_breakpoint()
        self.step_pre_check()
        result, metrics, rewards = training_step(
            self.algorithm,
            reward_updaters=self._reward_updaters,
            discrete_eval=self.discrete_eval,
            disable_report=True,
            log_stats=self.log_stats,
        )
        metrics["_checking_class_"] = True  # pyright: ignore[reportGeneralTypeIssues]
        if is_pbar(self._pbar):
            self._pbar.update(1)
        self.step_post_check(result, metrics, rewards)
        return metrics


if TYPE_CHECKING:  # check assignment
    __: type[_TrainableWithCheckProto] = TrainableWithChecks


# region Mock classes


def mock_result(t, rew, *, t_key="current_step"):
    return dict(**{t_key: t}, episode_reward_mean=rew, training_iteration=int(t))


class _FakeFutureResult(_FutureTrainingResult):
    # taken from ray's tests
    def __init__(self, result):
        self.result = result

    def resolve(self, block: bool = True):  # noqa: ARG002, FBT001, FBT002
        return self.result


def mock_trainable_algorithm(
    func: Optional[Callable[..., None]] = None,
    *,
    parallel_envs=1,
    mock_env_runners=True,
    mock_learner=True,
    mock_save_model_callback=True,
):
    """
    Decorator to create cheap Trainable with the expensive algorithm parts mocked out,
    i.e. no learner and env_runners created.
    """
    if func is None:
        return partial(
            mock_trainable_algorithm,
            parallel_envs=parallel_envs,
            mock_env_runners=mock_env_runners,
            mock_learner=mock_learner,
            mock_save_model_callback=mock_save_model_callback,
        )
    if parallel_envs != DefaultArgumentParser.num_envs_per_env_runner:
        env_runner_settings_mock = mock.patch.object(DefaultArgumentParser, "num_envs_per_env_runner", parallel_envs)
    else:
        env_runner_settings_mock = nullcontext()
    learner_mock = mock.patch.object(AlgorithmConfig, "build_learner") if mock_learner else nullcontext()
    learner_group_mock = mock.patch.object(AlgorithmConfig, "build_learner_group") if mock_learner else nullcontext()
    module_spec_mock = (
        mock.patch.object(AlgorithmConfig, "get_multi_rl_module_spec")
        if mock_learner or mock_env_runners
        else nullcontext()
    )
    env_runner_group_mock = mock.patch.object(algorithm_module, "EnvRunnerGroup") if mock_env_runners else nullcontext()
    save_model_mock = (
        # use a lambda to allow comparision
        mock.patch.object(
            ray_utilities.config.create_algorithm,
            "save_model_config_and_architecture",
            new=lambda *args, **kwargs: None,  # noqa: ARG005
        )
        if mock_save_model_callback
        else nullcontext()
    )
    save_model_mock_origin = (
        mock.patch.object(
            ray_utilities.callbacks.algorithm.model_config_saver_callback,
            "_get_module",
        )
        if mock_save_model_callback
        else nullcontext()
    )
    save_model_mock_origin_b = (
        mock.patch.object(
            ray_utilities.callbacks.algorithm.model_config_saver_callback,
            "open",
            new=lambda *args, **kwargs: nullcontext(),  # noqa: ARG005
        )
        if mock_save_model_callback
        else nullcontext()
    )
    save_model_mock_origin_c = (
        mock.patch.object(
            ray_utilities.callbacks.algorithm.model_config_saver_callback,
            "json",
        )
        if mock_save_model_callback
        else nullcontext()
    )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with (
            env_runner_settings_mock,
            learner_group_mock,
            learner_mock,
            module_spec_mock,
            # mock_old_api_multi_agent_setup,
            env_runner_group_mock,
            save_model_mock,
            save_model_mock_origin,
            save_model_mock_origin_b,
            save_model_mock_origin_c,
        ):
            return func(self, *args, **kwargs)

    return wrapper


def no_parallel_envs(func):
    """
    Decorator to set num_envs_per_env_runner to 1 for the duration of the test.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with mock.patch.object(DefaultArgumentParser, "num_envs_per_env_runner", 1):
            return func(self, *args, **kwargs)

    return wrapper


class _MockTrialRunner:
    """From ray testing suite"""

    def __init__(self, scheduler):
        self._scheduler_alg = scheduler
        self.search_alg = None
        self.trials = []

    def process_action(self, trial, action):
        if action == TrialScheduler.CONTINUE:
            pass
        elif action == TrialScheduler.PAUSE:
            self.pause_trial(trial)
        elif action == TrialScheduler.STOP:
            self.stop_trial(trial)

    def pause_trial(self, trial, should_checkpoint: bool = True):  # noqa: FBT001, FBT002
        if should_checkpoint:
            self._schedule_trial_save(trial, None)
        trial.status = Trial.PAUSED

    def stop_trial(self, trial, error=False, error_msg=None):  # noqa: ARG002, FBT002
        if trial.status in [Trial.ERROR, Trial.TERMINATED]:
            return
        if trial.status in [Trial.PENDING, Trial.PAUSED]:
            self._scheduler_alg.on_trial_remove(self, trial)
        else:
            self._scheduler_alg.on_trial_complete(self, trial, mock_result(100, 10))

        trial.status = Trial.ERROR if error else Trial.TERMINATED

    def add_trial(self, trial):
        self.trials.append(trial)
        self._scheduler_alg.on_trial_add(self, trial)

    def get_trials(self):
        return self.trials

    def get_live_trials(self):
        return {t for t in self.trials if t.status != Trial.TERMINATED}

    def _launch_trial(self, trial):
        trial.status = Trial.RUNNING

    def _set_trial_status(self, trial, status):
        trial.status = status

    def start_trial(self, trial, checkpoint_obj=None, train=True):  # noqa: ARG002, FBT002
        trial.logger_running = True
        if checkpoint_obj:
            trial.restored_checkpoint = checkpoint_obj.dir_or_data
        trial.status = Trial.RUNNING
        return True

    def _schedule_trial_restore(self, trial):
        pass

    def _schedule_trial_save(self, trial, result: dict | None = None):
        result = result or {}
        return _FakeFutureResult(
            _TrainingResult(
                checkpoint=Checkpoint.from_directory(trial.trainable_name),
                metrics=result,
            )
        )


class MockTrial(Trial):
    def __init__(self, i, config=None, storage=None):
        self.trainable_name = "trial_{}".format(i)
        self.trial_id = str(i)
        self.config = config or {}
        self.experiment_tag = "{}tag".format(i)
        self.trial_name_creator = None
        self.logger_running = False
        self._restored_checkpoint = None
        self._restore_checkpoint_result = None
        self.placement_group_factory = PlacementGroupFactory([{"CPU": 1}])
        self.custom_trial_name = None
        self.custom_dirname = None
        # ray missing coverage here,  if attr not in trial.config: i.e. config not provided by searcher
        # self.evaluated_params = {}  # XXX: Added by us; why does ray not raise error here
        self._legacy_local_experiment_path = None
        self.relative_logdir = None
        self._default_result_or_future = None
        self.run_metadata = _TrainingRunMetadata()
        self.run_metadata.checkpoint_manager = _CheckpointManager(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="episode_reward_mean",
            ),
        )
        self.temporary_state = _TemporaryTrialState()
        self.storage = storage or mock.MagicMock()

    @property
    def restored_checkpoint(self):
        if hasattr(self.run_metadata.checkpoint_manager, "_latest_checkpoint_result"):
            assert self.run_metadata.checkpoint_manager
            result = self.run_metadata.checkpoint_manager._latest_checkpoint_result
            if result is None:
                return self._restored_checkpoint
            assert result
            assert result.checkpoint
            return result.checkpoint.path
        return self._restored_checkpoint
