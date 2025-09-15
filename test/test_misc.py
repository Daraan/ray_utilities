import random
import re
import sys
from unittest import TestCase

import pytest
import ray.tune.logger
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray_utilities.config.typed_argument_parser import DefaultArgumentParser
from ray_utilities.misc import RE_GET_TRIAL_ID
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import (
    Cases,
    DisableLoggers,
    check_args,
    iter_cases,
    mock_trainable_algorithm,
    no_parallel_envs,
    patch_args,
)
from ray_utilities.training.helpers import make_divisible

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

    def test_make_divisible(self):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        a_div = make_divisible(a, b)
        self.assertEqual(a_div % b, 0)
        self.assertGreaterEqual(a_div, a)

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
