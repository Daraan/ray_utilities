from unittest import TestCase

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray_utilities.testing_utils import Cases, iter_cases


class TestMeta(TestCase):
    @Cases([1, 2, 3])
    def test_test_cases(self, cases):
        tested = []
        for i, r in enumerate(iter_cases(cases), start=1):
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
