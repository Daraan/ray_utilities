from __future__ import annotations

import sys
import unittest
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Collection
from unittest import mock

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy.testing as npt
import tree
from contextlib import nullcontext
from typing_extensions import NotRequired, Required, get_origin, get_type_hints


from ray_utilities.config import DefaultArgumentParser
from ray_utilities.setup.algorithm_setup import AlgorithmSetup

if TYPE_CHECKING:
    import chex
    from jaxlib.xla_extension import pytree  # pyright: ignore[reportMissingModuleSource] pyi file

    LeafType = pytree.SequenceKey | pytree.DictKey | pytree.GetAttrKey
    from flax.training.train_state import TrainState

args_train_no_tuner = mock.patch.object(
    sys, "argv", ["file.py", "--a", "NA", "--no-render_env", "-J", "1", "-it", "2", "-np"]
)
clean_args = mock.patch.object(sys, "argv", ["file.py", "-a", "NA"])
"""Use when comparing to CLIArgs"""


def patch_args(*args):
    patch = [
        "file.py",
        *(("-a", "no_actor_provided by patch_args") if ("-a" not in args and "--actor_type" not in args) else ()),
        *args,
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
        self._disable_loggers.stop()
        self._mock_env.stop()

    def setUp(self):
        self._mock_env = mock.patch.dict("os.environ", {"TUNE_DISABLE_AUTO_CALLBACK_LOGGERS": "1"})
        self._mock_env.start()
        self._disable_loggers = mock.patch("ray_utilities.callbacks.tuner.create_tuner_callbacks", return_value=[])
        self._disable_loggers.start()
        super().setUp()

    def tearDown(self):
        self.enable_loggers()
        super().tearDown()


class SetupDefaults(DisableLoggers):
    @clean_args
    def setUp(self):
        super().setUp()
        print("Remember to enable/disable justMyCode('\"debugpy.debugJustMyCode\": false,') in the settings")
        env = gym.make("CartPole-v1")

        self._OBSERVATION_SPACE = env.observation_space
        self._ACTION_SPACE = env.action_space

        self._DEFAULT_CONFIG_DICT: MappingProxyType[str, Any] = MappingProxyType(
            DefaultArgumentParser().parse_args().as_dict()
        )
        self._DEFAULT_NAMESPACE = DefaultArgumentParser()
        self._DEFAULT_SETUP = AlgorithmSetup()
        self._DEFAULT_SETUP_LOW_RES = AlgorithmSetup()
        self._DEFAULT_SETUP_LOW_RES.config.training(
            train_batch_size_per_learner=128, minibatch_size=64, num_epochs=2
        ).env_runners(num_env_runners=0, num_envs_per_env_runner=1, num_cpus_per_env_runner=0).learners(
            num_learners=0, num_cpus_per_learner=0
        )
        self._INPUT_LENGTH = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
        self._DEFAULT_INPUT = jnp.arange(self._INPUT_LENGTH * 2).reshape((2, self._INPUT_LENGTH))
        self._DEFAULT_BATCH: dict[str, chex.Array] = MappingProxyType({"obs": self._DEFAULT_INPUT})  # pyright: ignore[reportAttributeAccessIssue]
        self._ENV_SAMPLE = jnp.arange(self._INPUT_LENGTH)
        model_key = jax.random.PRNGKey(self._DEFAULT_CONFIG_DICT["seed"] or 2)
        self._RANDOM_KEY, self._ACTOR_KEY, self._CRITIC_KEY = jax.random.split(model_key, 3)
        self._ACTION_DIM: int = self._ACTION_SPACE.n  # type: ignore[attr-defined]
        self._OBS_DIM: int = self._OBSERVATION_SPACE.shape[0]  # pyright: ignore[reportOptionalSubscript]

    def util_test_tree_equivalence(
        self,
        tree1: TrainState | Any,
        tree2: TrainState | Any,
        ignore_leaves: Collection[str] = (),
        msg: str = "",
        attr_checked: str = "",
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
                else:
                    self.assertTrue(
                        comp.all(), f"Attribute '{attr}' not equal in both states: {attr1}\n!=\n{attr2}\n{msg}"
                    )

        # NOTE: Apply gradients modifies state


class DisableBreakpointsForGUI(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if {"-v", "test*.py"} & set(sys.argv):
            print("disable breakpoint")
            self._disabled_breakpoints = mock.patch("builtins.breakpoint")
            self._disabled_breakpoints.start()
        else:
            self._disabled_breakpoints = mock.patch("builtins.breakpoint")
            print("enable breakpoint")
