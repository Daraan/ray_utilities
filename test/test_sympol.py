from ray_utilities.testing_utils import patch_args

import unittest
from unittest import TestCase
import pytest

pytestmark = pytest.mark.xfail(raises=ModuleNotFoundError, reason="sympol module tests currently failing")


class TestSympolSetup(TestCase):
    def setUp(self) -> None:
        global SympolSetup
        from sympol import SympolSetup as SS

        SympolSetup = SS

    def test_sympol_setup_discrete(self):
        # test discrete
        with patch_args("-a", "sympol", "--env_type", "CartPole-v1", "--batch_size", 64):
            setup = SympolSetup()
            assert setup.args.env_type == "CartPole-v1"
        trainable = setup.trainable_class(setup.sample_params())
        trainable.step()

    def test_sympol_setup_continuous_legacy(self):
        # test continuous
        with patch_args(
            "-a", "sympol", "--env_type", "Ant-v5", "--batch_size", 64, "--legacy", "--action_type", "continuous"
        ):
            setup = SympolSetup()
            assert setup.args.env_type == "Ant-v5"
        assert setup.args.action_type == "continuous"
        trainable = setup.trainable_class(setup.sample_params())
        assert "action_type" in trainable.algorithm_config.model_config
        assert trainable.algorithm_config.model_config["action_type"] == "continuous"
        module = trainable.algorithm.get_module()
        assert module.model_config["action_type"] == "continuous"
        assert module.pi.config["action_type"] == "continuous"

        trainable.step()

    def test_sympol_setup_continuous(self):
        # test continuous
        with patch_args("-a", "sympol", "--env_type", "Ant-v5", "--batch_size", 64):
            setup = SympolSetup()
            assert setup.args.env_type == "Ant-v5"
        trainable = setup.trainable_class(setup.sample_params())
        assert "action_type" in trainable.algorithm_config.model_config
        assert trainable.algorithm_config.model_config["action_type"] == "continuous"
        module = trainable.algorithm.get_module()
        assert module.model_config["action_type"] == "continuous"
        assert module.pi.config["action_type"] == "continuous"
        trainable.step()


if __name__ == "__main__":
    unittest.main()
