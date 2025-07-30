# pyright: reportOptionalMemberAccess=information
from __future__ import annotations

# pyright: reportOptionalMemberAccess=none
import difflib
import math
import os
import pathlib
import pprint
import random
import sys
import unittest
import unittest.util
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Collection, Iterable, TypeAlias, TypeVar, final
from unittest import mock

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
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core import ALL_MODULES
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    LEARNER_RESULTS,
    TIMERS,
)
from ray.tune.result import TRAINING_ITERATION  # pyright: ignore[reportPrivateImportUsage]
from ray.tune.search.sample import Categorical, Domain, Float, Integer
from typing_extensions import Final, NotRequired, Required, Sentinel, get_origin, get_type_hints

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.setup.algorithm_setup import AlgorithmSetup, PPOSetup
from ray_utilities.training.default_class import DefaultTrainable

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import chex
    from flax.training.train_state import TrainState
    from jaxlib.xla_extension import pytree  # pyright: ignore[reportMissingModuleSource,reportMissingImports] pyi file
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
    from ray.tune import Result

    from ray_utilities.setup.experiment_base import AlgorithmType_co, ConfigType_co
    from ray_utilities.training.default_class import TrainableBase

    LeafType: TypeAlias = pytree.SequenceKey | pytree.DictKey | pytree.GetAttrKey

args_train_no_tuner = mock.patch.object(
    sys, "argv", ["file.py", "--a", "NA", "--no-render_env", "-J", "1", "-it", "2", "-np"]
)
clean_args = mock.patch.object(sys, "argv", ["file.py", "-a", "NA"])
"""Use when comparing to CLIArgs"""

_C = TypeVar("_C", bound="Callable[[Any, mock.MagicMock], Any]")


@final
class Cases:
    def __init__(self, cases: Iterable[Any] | Callable[[], Any] | BaseException, *args, **kwargs):  # noqa: ARG002
        self._cases = cases
        self._args = args
        self._kwargs = kwargs

    def __call__(self, func: _C) -> _C:
        """Allows to use TestCases as a decorator."""
        return self.cases(self._cases, *self._args, **self._kwargs)(func)  # pyright: ignore[reportReturnType]

    @classmethod
    def next(cls) -> Any:
        raise NotImplementedError("Mock this function with the test cases to return")

    @classmethod
    def cases(cls, cases: Iterable[Any] | Callable[[], Any] | BaseException, *args, **kwargs):
        return mock.patch.object(cls, "next", *args, side_effect=cases, **kwargs)


def iter_cases(cases: type[Cases] | mock.MagicMock):
    try:
        while True:
            if isinstance(cases, mock.MagicMock):
                yield cases()
            else:
                yield cases.next()
    except StopIteration:
        return
    except BaseException:
        raise


def patch_args(*args: str | int):
    patch = [
        "file.py",
        *(("-a", "no_actor_provided by patch_args") if ("-a" not in args and "--actor_type" not in args) else ()),
        *map(str, args),
    ]
    return mock.patch.object(
        sys,
        "argv",
        patch,
    )


def get_explicit_required_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is Required}


def get_explicit_unrequired_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is NotRequired}


def get_required_keys(cls):
    return cls.__required_keys__ - get_explicit_unrequired_keys(cls)


def get_optional_keys(cls):
    return cls.__optional__keys - get_explicit_required_keys(cls)


NOT_FOUND = object()


def get_leafpath_value(leaf: LeafType):
    """Returns the path value of a leaf, could be index (list), key (dict), or name (attribute)."""
    return getattr(leaf, "name", getattr(leaf, "key", getattr(leaf, "idx", NOT_FOUND)))


class DisableLoggers(unittest.TestCase):
    """Disable loggers for tests, so they do not interfere with the output."""

    def enable_loggers(self):
        """Enable loggers after disabling them in setUp."""
        self._disable_tune_loggers.stop()
        self._disable_file_loggers.stop()
        self._disable_file_loggers2.stop()
        self._mock_env.stop()

    def setUp(self):
        self._mock_env = mock.patch.dict("os.environ", {"TUNE_DISABLE_AUTO_CALLBACK_LOGGERS": "1"})
        self._mock_env.start()
        self._disable_tune_loggers = mock.patch("ray_utilities.callbacks.tuner.create_tuner_callbacks", return_value=[])
        self._disable_tune_loggers.start()
        self._disable_file_loggers = mock.patch.object(ray.tune.logger, "DEFAULT_LOGGERS", ())
        self._disable_file_loggers.start()
        self._disable_file_loggers2 = mock.patch.object(ray.tune.logger.unified, "DEFAULT_LOGGERS", ())
        """Disable local copy used by UnifiedLogger"""
        self._disable_file_loggers2.start()
        super().setUp()

    def tearDown(self):
        self.enable_loggers()
        super().tearDown()


class InitRay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Ray for the test class."""
        if not ray.is_initialized():
            ray.init(
                include_dashboard=False,
                ignore_reinit_error=True,
            )
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        """Shutdown Ray after the test class."""
        if ray.is_initialized():
            ray.shutdown()
        super().tearDownClass()


OVERRIDE_KEYS: Final[set[str]] = {"num_env_runners", "num_epochs", "minibatch_size", "train_batch_size_per_learner"}

# pyright: enableExperimentalFeatures=true
_NOT_PROVIDED = Sentinel(
    "_NOT_PROVIDED",
)


class TestHelpers(unittest.TestCase):
    # region setups

    def setUp(self):
        super().setUp()
        self._env_seed_rng = random.Random(111)

    _created_trainables: ClassVar[list[TrainableBase]] = []

    def tearDown(self):
        for trainable in self._created_trainables:
            trainable.cleanup()
        super().tearDown()

    @patch_args(
        "--iterations", "5", "--total_steps", "320", "--batch_size", "64", "--comment", "running tests", "--seed", "42"
    )
    def get_trainable(self, *, num_env_runners: int = 0, env_seed: int | None | _NOT_PROVIDED = _NOT_PROVIDED):
        # NOTE: In this test attributes are shared BY identity, this is just a weak test.
        self.TrainableClass: type[DefaultTrainable[DefaultArgumentParser, PPOConfig, PPO]] = DefaultTrainable.define(
            PPOSetup.typed()
        )
        # this initializes the algorithm; overwrite batch_size of 64 again.
        # This does not modify the state["setup"]["config"]
        overrides = AlgorithmConfig.overrides(
            num_env_runners=num_env_runners, num_epochs=2, minibatch_size=32, train_batch_size_per_learner=32
        )
        if env_seed is _NOT_PROVIDED:
            # use a random but reproducible seed
            if not hasattr(self, "_env_seed_rng"):
                self.setUp()
            env_seed = self._env_seed_rng.randint(0, 2**15 - 1)
        trainable = self.TrainableClass({"env_seed": env_seed}, algorithm_overrides=overrides)
        self._created_trainables.append(trainable)
        self.assertEqual(trainable._algorithm_overrides, overrides)
        self.assertEqual(overrides.keys(), OVERRIDE_KEYS)
        self.assertEqual(trainable.algorithm_config.num_env_runners, num_env_runners)
        self.assertEqual(trainable.algorithm_config.minibatch_size, 32)
        self.assertEqual(trainable.algorithm_config.train_batch_size_per_learner, 32)
        self.assertEqual(trainable.algorithm_config.num_epochs, 2)
        self.assertEqual(trainable._setup.args.iterations, 5)
        self.assertEqual(trainable._setup.args.total_steps, 320)
        self.assertEqual(trainable._setup.args.train_batch_size_per_learner, 64)  # not overwritten

        result1 = trainable.step()
        return trainable, result1

    # endregion

    def set_max_diff(self, max_diff: int | None = None):
        """Changes the maxDiff only when environment variable KEEP_MAX_DIFF is not set."""
        if int(os.environ.get("KEEP_MAX_DIFF", "0")):
            return
        self.maxDiff = max_diff

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
    ):
        """
        Args:
            metrics_0: Metrics from the first env_runner.
            metrics_1: Metrics from the second env_runner.
            msg: Optional message to display on failure.
            strict: If True, all keys must match exactly, otherwise only common keys are compared.
                For restored metrics the min/max/mean keys might be different
            compare_results: If False, some keys are ignored in the comparison.
        """
        key_difference = set(metrics_0.keys()).symmetric_difference(metrics_1.keys())
        print("Key differences for metrics:", sorted(key_difference))
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
        self.set_max_diff(None)
        # compare nan values
        self.assertEqual(
            {k: math.isnan(v) for k in all_keys if isinstance(v := metrics_0[k], float)},
            {k: math.isnan(v) for k in all_keys if isinstance(v := metrics_1[k], float)},
            msg=f"NaN values differ: {metrics_0} != {metrics_1} {msg}",
        )
        # not nans
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
        """Check if two states are equivalent."""
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

    # region tests

    def compare_env_runner_configs(self, algo: Algorithm, algo_restored: Algorithm):
        self.set_max_diff(max(self.maxDiff or 0, 13000))

        def assertCleanDictEqual(a, b, *args, **kwargs):  # noqa: N802
            self.assertDictEqual(
                self.filter_incompatible_remote_config(a), self.filter_incompatible_remote_config(b), *args, **kwargs
            )

        algo_config_dict = algo.config.to_dict()
        algo_restored_config_dict = algo_restored.config.to_dict()
        assertCleanDictEqual(algo_restored_config_dict, algo_config_dict)
        if algo.config.num_env_runners == 0:  # pyright: ignore[reportOptionalMemberAccess]
            self.assertEqual(algo_restored.config.num_env_runners, 0)  # pyright: ignore[reportOptionalMemberAccess]
            assertCleanDictEqual(
                (algo.env_runner.config.to_dict()),
                algo_config_dict,  # pyright: ignore[reportOptionalMemberAccess]
            )
            restored_env_runner_config_dict = algo_restored.env_runner.config.to_dict()
            assertCleanDictEqual(restored_env_runner_config_dict, algo_restored_config_dict)
            assertCleanDictEqual(algo_config_dict, restored_env_runner_config_dict)

        remote_configs = algo.env_runner_group.foreach_env_runner(lambda r: r.config.to_dict())
        # Possible ignore local env_runner here when using remotes
        for i, config in enumerate(remote_configs):
            assertCleanDictEqual(
                config, algo_config_dict, f"Remote config {i}/{len(remote_configs)} does not match algo config"
            )
        remote_configs_restored = algo_restored.env_runner_group.foreach_env_runner(lambda r: r.config.to_dict())
        for i, config in enumerate(remote_configs_restored):
            assertCleanDictEqual(
                config,
                algo_restored_config_dict,
                f"Remote config {i}/{len(remote_configs_restored)} does not match restored config",
            )
            assertCleanDictEqual(
                config, algo_config_dict, f"Remote config {i}/{len(remote_configs_restored)} does not match algo config"
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

    def compare_trainables(
        self,
        trainable: DefaultTrainable["DefaultArgumentParser", "ConfigType_co", "AlgorithmType_co"],
        trainable2: DefaultTrainable["DefaultArgumentParser", "ConfigType_co", "AlgorithmType_co"],
        msg: str = "",
        *,
        iteration_after_step=2,
        minibatch_size=32,
        **subtest_kwargs,
    ) -> None:
        """
        Test functions for trainables obtained in different ways

        Attention:
            Does perform a step on each trainable
        """
        self.set_max_diff(60_000)
        with self.subTest("Step 1: Compare trainables " + msg, **subtest_kwargs):
            if hasattr(trainable, "_args") or hasattr(trainable2, "_args"):
                self.assertDictEqual(trainable2._args, trainable._args)  # type: ignore[attr-defined]
            self.assertEqual(trainable.algorithm_config.minibatch_size, minibatch_size)
            self.assertEqual(trainable2.algorithm_config.minibatch_size, trainable.algorithm_config.minibatch_size)
            self.assertEqual(trainable2._iteration, trainable._iteration)

            # get_state stores "class" : type(self) of the config, this allows from_state to work correctly
            # original trainable does not have that key
            config_dict1 = trainable.algorithm_config.to_dict()
            config_dict1.pop("class", None)
            config_dict2 = trainable2.algorithm_config.to_dict()
            config_dict2.pop("class", None)
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
            keys.remove("config")
            param_space1 = setup_data1["param_space"]
            param_space2 = setup_data2["param_space"]
            keys.remove("param_space")
            self.assertDictEqual(setup_data1["config_overrides"], setup_data2["config_overrides"])
            keys.remove("config_overrides")
            self.assertEqual(len(keys), 0, f"Unchecked keys: {keys}")  # checked all params
            self.assertCountEqual(param_space1, param_space2)
            self.assertDictEqual(param_space1["cli_args"], param_space2["cli_args"])
            for key in param_space1.keys() | param_space2.keys():
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
                pbar1_state = trainable._pbar._get_state()  # type: ignore
                pbar2_state = trainable2._pbar._get_state()
                pbar1_state = {k: v for k, v in pbar1_state.items() if k not in ("desc", "uuid")}
                pbar2_state = {k: v for k, v in pbar2_state.items() if k not in ("desc", "uuid")}
                self.assertEqual(pbar1_state, pbar2_state)

            # Step 2
            result2 = trainable.step()
            result2_restored = trainable2.step()
            self.assertEqual(result2[TRAINING_ITERATION], result2_restored[TRAINING_ITERATION], msg)
            self.assertEqual(result2[TRAINING_ITERATION], iteration_after_step, msg)

        # Compare env_runners
        with self.subTest("Step 2 Compare env_runner configs " + msg, **subtest_kwargs):
            if trainable.algorithm.env_runner or trainable2.algorithm.env_runner:
                assert trainable.algorithm.env_runner and trainable2.algorithm.env_runner
                self.compare_env_runner_configs(trainable.algorithm, trainable2.algorithm)

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


class SetupDefaults(TestHelpers, DisableLoggers):
    @clean_args
    def setUp(self):
        super().setUp()
        env = gym.make("CartPole-v1")

        self._OBSERVATION_SPACE = env.observation_space
        self._ACTION_SPACE = env.action_space

        self._DEFAULT_CONFIG_DICT: MappingProxyType[str, Any] = MappingProxyType(
            DefaultArgumentParser().parse_args().as_dict()
        )
        self._DEFAULT_NAMESPACE = DefaultArgumentParser()
        self._DEFAULT_SETUP = AlgorithmSetup(init_trainable=False)

        self._DEFAULT_SETUP_LOW_RES = AlgorithmSetup(init_trainable=False)
        self._DEFAULT_SETUP_LOW_RES.config.training(
            train_batch_size_per_learner=128, minibatch_size=64, num_epochs=2
        ).env_runners(num_env_runners=0, num_envs_per_env_runner=1, num_cpus_per_env_runner=0).learners(
            num_learners=0, num_cpus_per_learner=0
        )
        self._DEFAULT_SETUP_LOW_RES.create_trainable()
        self._DEFAULT_SETUP.create_trainable()
        self._INPUT_LENGTH = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
        self._DEFAULT_INPUT = jnp.arange(self._INPUT_LENGTH * 2).reshape((2, self._INPUT_LENGTH))
        self._DEFAULT_BATCH: dict[str, chex.Array] = MappingProxyType({"obs": self._DEFAULT_INPUT})  # pyright: ignore[reportAttributeAccessIssue]
        self._ENV_SAMPLE = jnp.arange(self._INPUT_LENGTH)
        model_key = jax.random.PRNGKey(self._DEFAULT_CONFIG_DICT["seed"] or 2)
        self._RANDOM_KEY, self._ACTOR_KEY, self._CRITIC_KEY = jax.random.split(model_key, 3)
        self._ACTION_DIM: int = self._ACTION_SPACE.n  # type: ignore[attr-defined]
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
