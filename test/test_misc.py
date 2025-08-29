import random
from unittest import TestCase

import pytest
import ray.tune.logger
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray_utilities.misc import RE_GET_TRIAL_ID
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from ray_utilities.testing_utils import Cases, DisableLoggers, iter_cases
from ray_utilities.training.helpers import make_divisible


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
        match = RE_GET_TRIAL_ID.search("sdf_sdgsg_12:12:id=sd353_00002_sdfgf")
        assert match is not None
        self.assertEqual(match.group(), "id=sd353_00002")
        self.assertEqual(match.group(1), "sd353_00002")
        self.assertEqual(match.groups(), ("sd353_00002",))
        self.assertEqual(match.groupdict(), {"trial_id": "sd353_00002"})

    def test_make_divisible(self):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        a_div = make_divisible(a, b)
        self.assertEqual(a_div % b, 0)
        self.assertGreaterEqual(a_div, a)
