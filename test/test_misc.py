from __future__ import annotations

import io
import os
import random
import re
import sys
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest import TestCase

import pytest
import ray.tune.logger
from ray import tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.runtime_env import RuntimeEnv

from ray_utilities.config import DefaultArgumentParser
from ray_utilities.constants import (
    COMET_OFFLINE_DIRECTORY,
    FORK_DATA_KEYS,
    FORK_FROM_CSV_KEY_MAPPING,
    RE_PARSE_FORK_FROM,
)
from ray_utilities.misc import RE_GET_TRIAL_ID, make_experiment_key, parse_fork_from
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import (
    Cases,
    DisableLoggers,
    MockPopen,
    MockPopenClass,
    check_args,
    iter_cases,
    mock_trainable_algorithm,
    no_parallel_envs,
    patch_args,
)
from ray_utilities.training.helpers import make_divisible
from ray_utilities.typing import ForkFromData

if TYPE_CHECKING:
    from ray_utilities.typing import Forktime

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


@pytest.mark.basic
class TestMeta(TestCase):
    @Cases([1, 2, 3])
    def test_test_cases(self, cases):
        tested = []
        for i, r in enumerate(iter_cases(cases), start=1):
            self.assertEqual(r, i)
            tested.append(r)
        self.assertEqual(tested, [1, 2, 3])
        # test list
        tested = []
        for i, r in enumerate(iter_cases([1, 2, 3]), start=1):
            self.assertEqual(r, i)
            tested.append(r)
        self.assertEqual(tested, [1, 2, 3])
        # Test iterator
        tested = []
        for i, r in enumerate(iter_cases(v for v in [1, 2, 3]), start=1):
            self.assertEqual(r, i)
            tested.append(r)
        self.assertEqual(tested, [1, 2, 3])

    def test_env_config_merge(self):
        config = AlgorithmConfig()
        config.environment(env_config={"a": 1, "b": 2})
        config.evaluation(evaluation_config=AlgorithmConfig.overrides(env_config={"a": 5, "d": 5}))
        override = AlgorithmConfig.overrides(env_config={"b": 3, "c": 4})
        config.update_from_dict(override)
        self.assertDictEqual(config.env_config, {"a": 1, "b": 3, "c": 4})
        eval_config = config.get_evaluation_config_object()
        assert eval_config
        self.assertDictEqual(eval_config.env_config, {"a": 5, "b": 3, "c": 4, "d": 5})

    @no_parallel_envs
    @mock_trainable_algorithm
    def test_no_parallel_envs(self):
        self.assertEqual(DefaultArgumentParser.num_envs_per_env_runner, 1)
        self.assertEqual(
            AlgorithmSetup(init_param_space=False).trainable_class().algorithm_config.num_envs_per_env_runner, 1
        )


class TestNoLoggers(DisableLoggers):
    @mock_trainable_algorithm
    def test_no_loggers(self):
        # This test is just to ensure that the DisableLoggers context manager works.
        # It does not need to do anything, as the context manager will disable loggers.
        self.assertEqual(ray.tune.logger.DEFAULT_LOGGERS, ())
        setup = AlgorithmSetup()
        trainable = setup.trainable_class()
        if isinstance(trainable._result_logger, ray.tune.logger.UnifiedLogger):
            self.assertEqual(trainable._result_logger._logger_cls_list, ())
            self.assertEqual(len(trainable._result_logger._loggers), 0)
        if isinstance(trainable.algorithm._result_logger, ray.tune.logger.UnifiedLogger):
            self.assertEqual(trainable.algorithm._result_logger._logger_cls_list, ())
            self.assertEqual(len(trainable.algorithm._result_logger._loggers), 0)


@pytest.mark.basic
class TestMisc(TestCase):
    def test_re_find_id(self):
        match = RE_GET_TRIAL_ID.search("sdf_sdgsg_12:12:id=52e65_00002_sdfgf")
        assert match is not None
        self.assertEqual(match.groups(), ("52e65_00002", "52e65", "00002"))
        self.assertEqual(match.group(), "id=52e65_00002")
        self.assertEqual(match.group(1), "52e65_00002")
        self.assertEqual(match.group("trial_id"), "52e65_00002")
        self.assertEqual(
            match.groupdict(), {"trial_id": "52e65_00002", "trial_id_part1": "52e65", "trial_number": "00002"}
        )

        match = RE_GET_TRIAL_ID.search("sdf_sdgsg_12:12:id=52e65_sdfgf")
        assert match is not None
        self.assertEqual(match.groups(), ("52e65", "52e65", None))
        self.assertEqual(match.group(), "id=52e65")
        self.assertEqual(match.group(1), "52e65")
        self.assertEqual(match.group("trial_id"), "52e65")
        self.assertEqual(match.groupdict(), {"trial_id": "52e65", "trial_id_part1": "52e65", "trial_number": None})

    def test_re_parse_forked_from(self):
        id1 = "52e65_00002"
        step1 = 123
        complete_id1 = f"{id1}?_step={step1}"
        id2 = complete_id1  # .replace("?", "")
        step2 = 456
        complete_id2 = f"{id2}?_step={step2}"
        # in practice we take care that ? is never in a trial id
        id3 = complete_id2.replace("?", "")
        step3 = 789
        complete_id3 = f"{id3}?_step={step3}"
        self.assertIsNone(RE_PARSE_FORK_FROM.search(complete_id2))  # contains ? x2
        for test_id, expected_id, expected_step in [
            (id1, id1, None),
            (complete_id1, id1, step1),
            (complete_id3, id3, step3),
        ]:
            with self.subTest(test_id=test_id, expected_id=expected_id, expected_step=expected_step):
                match = RE_PARSE_FORK_FROM.search(test_id)
                assert match is not None
                self.assertEqual(
                    match.groups(), (expected_id, str(expected_step) if expected_step is not None else None)
                )
                self.assertEqual(match.group(), test_id)
                self.assertEqual(match.group(1), expected_id)
                self.assertEqual(match.group("fork_id"), expected_id)
                self.assertEqual(
                    match.groupdict(),
                    {"fork_id": expected_id, "fork_step": str(expected_step) if expected_step is not None else None},
                )
                # Check with utility function
                self.assertEqual(parse_fork_from(test_id), (expected_id, expected_step))

    def test_make_divisible(self):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        a_div = make_divisible(a, b)
        self.assertEqual(a_div % b, 0)
        self.assertGreaterEqual(a_div, a)

    def test_experiment_key_length(self):
        trial1 = SimpleNamespace(trial_id="abcd0000")
        trial2 = SimpleNamespace(trial_id="abcd0000_00001")

        self.assertTrue(32 <= len(make_experiment_key(trial1)) <= 50)  # pyright: ignore[reportArgumentType]
        self.assertTrue(32 <= len(make_experiment_key(trial2)) <= 50)  # pyright: ignore[reportArgumentType]
        for t1 in [trial1, trial2]:
            for t2_id in [trial1.trial_id, trial2.trial_id]:
                for step in [None, 2_000_000]:
                    with self.subTest(t1=t1, t2_id=t2_id, step=step):
                        fork_data: ForkFromData = {
                            "parent_trial_id": t2_id,
                            "parent_training_iteration": step,
                            "parent_time": cast("Forktime", ("training_iteration", step)),
                        }
                        self.assertTrue(
                            32
                            < len(
                                make_experiment_key(
                                    t1,  # pyright: ignore[reportArgumentType]
                                    fork_data,
                                )
                            )
                            <= 50
                        )

    def test_check_valid_args_decorator(self):
        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):

            @check_args
            @patch_args("--it", 10, check_for_errors=False)
            def f():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            f()

        with self.assertRaisesRegex(ExceptionGroup, "Unexpected unrecognized args"):

            @check_args
            @patch_args("--it", 10, check_for_errors=False)
            def f2():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)
                raise ValueError("Some other error")

            f2()

        # Test exception

        @check_args(exceptions=["--it", "10"])
        @patch_args("--it", 10, check_for_errors=False)
        def h():
            AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

        h()

        # Exception order matters
        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):

            @check_args(exceptions=["10", "--it"])
            @patch_args("--it", 10, check_for_errors=False)
            def g():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            g()

        # Exception order matters
        with self.assertRaisesRegex(
            ValueError, re.escape("Unexpected unrecognized args: ['--foo', '10', '--it', '10', '--bar', '10']")
        ):

            @check_args(exceptions=["--it", "10"])
            @patch_args("--foo", "10", "--it", "10", "--bar", "10", check_for_errors=False)
            def g():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            g()

    def test_parse_args_with_check(self):
        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):

            @patch_args("--it", 10)
            def f():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            f()

        with self.assertRaisesRegex(ExceptionGroup, "Unexpected unrecognized args"):

            @patch_args("--it", 10)
            def f2():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)
                raise ValueError("Some other error")

            f2()

        # Test exception;  OK
        @patch_args("--it", 10, except_parser_errors=["--it", "10"])
        def h():
            AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

        h()

        # Exception order matters

        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):

            @patch_args("--it", 10, except_parser_errors=["10", "--it"])
            def g():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            g()

        # Exception order matters
        with self.assertRaisesRegex(
            ValueError, re.escape("Unexpected unrecognized args: ['--foo', '12', '--bar', '13']")
        ):

            @patch_args("--foo", "10", "--it", "12", "--bar", "13", except_parser_errors=["10", "--it"])
            def g():
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

            g()

    def test_parse_args_as_with(self):
        with self.assertRaisesRegex(ValueError, re.escape("Unexpected unrecognized args: ['--it', '10']")):
            with patch_args("--it", 10):
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

        with self.assertRaisesRegex(ExceptionGroup, "Unexpected unrecognized args"):
            with patch_args("--it", 10):
                AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)
                raise ValueError("Some other error")

        # Test exception;  OK
        with patch_args("--it", 10, except_parser_errors=["--it", "10"]):
            AlgorithmSetup(init_trainable=False, init_config=False, init_param_space=False)

    def test_can_import_default_arguments(self):
        # This test is just to ensure that the default_arguments module can be imported
        # without errors. It does not need to do anything else.
        import default_arguments.PYTHON_ARGCOMPLETE_OK  # noqa: PLC0415

    @unittest.skipIf("GITHUB_REF" in os.environ, "Skip test on GitHub Actions")
    def test_runtime_env(self):
        # Ray might already have been started unknowingly, e.g. by EnvRunnerGroup creation
        # in other tests, however then runtime_env is NOT applied - this test is a bit slow when added
        runtime_env = RuntimeEnv(
            env_vars={
                "RAY_UTILITIES_NEW_LOG_FORMAT": "1",
                "COMET_OFFLINE_DIRECTORY": COMET_OFFLINE_DIRECTORY,
                "RAY_UTILITIES_SET_COMET_DIR": "0",  # do not warn on remote
                "TEST_RANDOM_ENV_VAR": "TEST123",
            }
        )
        ray.shutdown()  # might be started by auto_init side effects
        ray.init(runtime_env=runtime_env, num_cpus=1, include_dashboard=False, object_store_memory=78643200)
        os.environ["LATE_VARIABLE"] = "0000"

        def fake_trainable(config):
            from ray_utilities import COMET_OFFLINE_DIRECTORY as comet_offline_dir_remote  # noqa: N811, PLC0415

            assert os.environ.get("RAY_UTILITIES_NEW_LOG_FORMAT") == "1"
            assert comet_offline_dir_remote == COMET_OFFLINE_DIRECTORY
            assert os.environ.get("COMET_OFFLINE_DIRECTORY") == COMET_OFFLINE_DIRECTORY
            assert os.environ.get("RAY_UTILITIES_SET_COMET_DIR") == "0"
            assert os.environ.get("TEST_RANDOM_ENV_VAR") == "TEST123"
            assert "LATE_VARIABLE" not in os.environ
            return {"return_value": 1}

        result = tune.Tuner(fake_trainable).fit()
        self.assertIn("LATE_VARIABLE", os.environ)
        self.assertEqual(result.get_best_result().metrics["return_value"], 1)  # pyright: ignore[reportOptionalSubscript]
        ray.shutdown()

    def test_mock_popen_class(self):
        cls1 = None
        cls2 = None

        @MockPopenClass.mock
        def foo1(mock_popen_class, mock_popen):
            assert hasattr(MockPopenClass().stdout, "read")
            assert hasattr(MockPopenClass(), "poll")
            assert MockPopenClass() is mock_popen
            nonlocal cls1
            cls1 = mock_popen_class

        @MockPopenClass.mock
        def foo2(mock_popen_class, mock_popen):
            assert hasattr(MockPopenClass().stdout, "read")
            assert hasattr(MockPopenClass(), "poll")
            assert MockPopenClass() is mock_popen
            nonlocal cls2
            cls2 = mock_popen_class

        foo1()  # pyright: ignore[reportCallIssue]
        foo2()  # pyright: ignore[reportCallIssue]
        assert cls1 is not None and cls2 is not None
        self.assertEqual(cls1.call_count, 3)
        self.assertIsNot(cls1, cls2)

        mocked_popen = MockPopen()
        assert isinstance(mocked_popen, MockPopenClass)
        assert isinstance(mocked_popen.stdout, (io.StringIO, io.BytesIO)), type(mocked_popen.stdout)
        assert mocked_popen.args

    def test_fork_from_data_keys(self):
        """Test key membership between FORK_DATA_KEYS, ForkFromData and FORK_FROM_DATA_TO_FORK_DATA_KEYS."""
        fork_dict_keys = ForkFromData.__required_keys__ | ForkFromData.__optional_keys__
        self.assertSetEqual(set(FORK_DATA_KEYS), set(FORK_FROM_CSV_KEY_MAPPING.keys()))
        # The keys in FORK_FROM_DATA_TO_FORK_DATA_KEYS can be a subset of fork_dict_keys
        self.assertTrue({v for v in FORK_FROM_CSV_KEY_MAPPING.values() if v is not None}.issubset(fork_dict_keys))
