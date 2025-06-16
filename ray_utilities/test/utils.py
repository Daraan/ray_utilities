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
from typing_extensions import NotRequired, Required, get_origin, get_type_hints

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.setup.algorithm_setup import AlgorithmSetup

if TYPE_CHECKING:
    import chex
    from flax.training.train_state import TrainState

args_train_no_tuner = mock.patch.object(
    sys, "argv", ["file.py", "--a", "NA", "--no-render_env", "-J", "1", "-it", "2", "-np"]
)
clean_args = mock.patch.object(sys, "argv", ["file.py", "--a", "NA"])
"""Use when comparing to CLIArgs"""


def patch_args(*args):
    return mock.patch.object(sys, "argv", ["file.py", "--a", "NA", *args])


def get_explicit_required_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is Required}


def get_explicit_unrequired_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is NotRequired}


def get_required_keys(cls):
    return cls.__required_keys__ - get_explicit_unrequired_keys(cls)


def get_optional_keys(cls):
    return cls.__optional__keys - get_explicit_required_keys(cls)


class SetupDefaults(unittest.TestCase):
    @clean_args
    def setUp(self):
        print("Remember to enable/disable justMyCode('\"debugpy.debugJustMyCode\": false,') in the settings")
        env = gym.make("CartPole-v1")

        self._OBSERVATION_SPACE = env.observation_space
        self._ACTION_SPACE = env.action_space

        self._DEFAULT_CONFIG_DICT: MappingProxyType[str, Any] = MappingProxyType(
            DefaultArgumentParser().parse_args().as_dict()
        )
        self._DEFAULT_NAMESPACE = DefaultArgumentParser()
        self._DEFAULT_SETUP = AlgorithmSetup()
        self._INPUT_LENGTH = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
        self._DEFAULT_INPUT = jnp.arange(self._INPUT_LENGTH * 2).reshape((2, self._INPUT_LENGTH))
        self._DEFAULT_BATCH: dict[str, chex.Array] = MappingProxyType({"obs": self._DEFAULT_INPUT})  # pyright: ignore[reportAttributeAccessIssue]
        self._ENV_SAMPLE = jnp.arange(self._INPUT_LENGTH)
        model_key = jax.random.PRNGKey(self._DEFAULT_CONFIG_DICT["seed"] or 2)
        self._RANDOM_KEY, self._ACTOR_KEY, self._CRITIC_KEY = jax.random.split(model_key, 3)
        self._ACTION_DIM: int = self._ACTION_SPACE.n  # type: ignore[attr-defined]
        self._OBS_DIM: int = self._OBSERVATION_SPACE.shape[0]  # pyright: ignore[reportOptionalSubscript]

    def util_test_state_equivalence(
        self,
        state1: TrainState | Any,
        state2: TrainState | Any,
        msg="",
        *,
        ignore: Collection[str] = (),
    ):
        """Check if two states are equivalent."""
        # Check if the parameters and indices are equal
        if isinstance(ignore, str):
            ignore = {ignore}
        else:
            ignore = set(ignore)

        for attr in ["params", "indices", "grad_accum", "opt_state"]:
            if attr in ignore:
                continue
            with self.subTest(msg=msg, attr=attr):
                val1 = getattr(state1, attr, None)
                val2 = getattr(state2, attr, None)
                self.assertEqual(val1 is not None, val2 is not None, f"Attribute {attr} not found in both states {msg}")
                if val1 is None and val2 is None:
                    continue
                flat1 = tree.flatten(val1)
                flat_params2 = tree.flatten(val2)
                tree.assert_same_structure(flat1, flat_params2)
                for p1, p2 in zip(flat1, flat_params2):
                    npt.assert_array_equal(p1, p2, err_msg=f"Attribute '{attr}' not equal in both states {msg}")

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
                val1 = getattr(state1, attr, None)
                val2 = getattr(state2, attr, None)
                self.assertEqual(
                    val1 is not None, val2 is not None, f"Attribute '{attr}' not found in both states {msg}"
                )
                comp = val1 == val2
                if isinstance(comp, bool):
                    self.assertTrue(comp, f"Attribute '{attr}' not equal in both states: {val1}\n!=\n{val2}\n{msg}")
                else:
                    self.assertTrue(
                        comp.all(), f"Attribute '{attr}' not equal in both states: {val1}\n!=\n{val2}\n{msg}"
                    )
        # NOTE: Apply gradients modifies state


class DisableBreakpointsForGUI(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if {"-v", "test*.py"} & set(sys.argv):
            print("disable breakpoint")
            mock.patch("builtins.breakpoint").start()
        else:
            print("enable breakpoint")
