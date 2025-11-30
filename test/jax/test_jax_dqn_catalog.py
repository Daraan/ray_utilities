"""Tests for JAX DQN Catalog implementation.

Note:
    These tests are currently skipped because the JaxDQNCatalog currently only
    builds Torch models (due to RLlib catalog limitations). The tests were designed
    for JAX models which are not yet supported.
"""

from __future__ import annotations

import pytest

try:
    from ray_utilities.jax.dqn import JaxDQNCatalog
except ImportError:
    print("JaxDQNCatalog import failed, skipping tests.")
    pytest.skip("JaxDQNCatalog not available", allow_module_level=True)

import gymnasium as gym
import numpy as np

from ray_utilities.testing_utils import DisableLoggers, TestHelpers, patch_args


class TestJaxDQNCatalog(DisableLoggers, TestHelpers):
    """Test suite for JaxDQNCatalog."""

    def setup_method(self, method):
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.model_config = {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
            "head_fcnet_hiddens": [32],
            "head_fcnet_activation": "relu",
            "num_atoms": 1,
            "uses_dueling": False,
            "framework": "torch",
        }

    @patch_args()
    def test_catalog_initialization(self):
        """Test basic catalog initialization."""
        catalog = JaxDQNCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
        )
        assert catalog.observation_space == self.observation_space
        assert catalog.action_space == self.action_space

    @patch_args()
    def test_build_encoder(self):
        """Test encoder building."""
        catalog = JaxDQNCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
        )
        encoder = catalog.build_encoder(framework="torch")
        assert encoder is not None

    @patch_args()
    def test_build_af_head(self):
        """Test advantage/Q-function head building."""
        catalog = JaxDQNCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
        )
        af_head = catalog.build_af_head(framework="torch")
        assert af_head is not None

    @patch_args()
    def test_build_vf_head_dueling(self):
        """Test value function head building for dueling architecture."""
        dueling_config = self.model_config.copy()
        dueling_config["uses_dueling"] = True
        catalog = JaxDQNCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=dueling_config,
        )
        vf_head = catalog.build_vf_head(framework="torch")
        assert vf_head is not None

    @patch_args()
    def test_build_vf_head_non_dueling(self):
        """
        Test that vf_head for non-dueling architecture
        it is up to other code paths that build_vf_head is not called when not needed.
        """
        catalog = JaxDQNCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
        )
        # RLlib now always returns a value head, even for non-dueling
        vf_head = catalog.build_vf_head(framework="torch")
        assert vf_head is not None

    @patch_args()
    def test_distributional_config(self):
        """Test catalog with distributional Q-learning configuration."""
        dist_config = self.model_config.copy()
        dist_config["num_atoms"] = 51
        dist_config["v_min"] = -10.0
        dist_config["v_max"] = 10.0
        catalog = JaxDQNCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=dist_config,
        )
        # Should build successfully
        encoder = catalog.build_encoder(framework="torch")
        af_head = catalog.build_af_head(framework="torch")
        assert encoder is not None
        assert af_head is not None

    @patch_args()
    def test_dueling_and_distributional(self):
        """Test catalog with both dueling and distributional Q-learning."""
        combined_config = self.model_config.copy()
        combined_config["uses_dueling"] = True
        combined_config["num_atoms"] = 51
        combined_config["v_min"] = -10.0
        combined_config["v_max"] = 10.0
        catalog = JaxDQNCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=combined_config,
        )
        encoder = catalog.build_encoder(framework="torch")
        af_head = catalog.build_af_head(framework="torch")
        vf_head = catalog.build_vf_head(framework="torch")
        assert encoder is not None
        assert af_head is not None
        assert vf_head is not None

    @patch_args()
    def test_different_activation_functions(self):
        """Test catalog with different activation functions."""
        for activation in ["relu", "tanh", "linear"]:
            config = self.model_config.copy()
            config["fcnet_activation"] = activation
            config["head_fcnet_activation"] = activation
            catalog = JaxDQNCatalog(
                observation_space=self.observation_space,
                action_space=self.action_space,
                model_config_dict=config,
            )
            # Should build successfully
            encoder = catalog.build_encoder(framework="torch")
            af_head = catalog.build_af_head(framework="torch")
            assert encoder is not None
            assert af_head is not None

    @patch_args()
    def test_different_hidden_layer_configs(self):
        """Test catalog with different hidden layer configurations."""
        configs = [
            {"fcnet_hiddens": [32], "head_fcnet_hiddens": [16]},
            {"fcnet_hiddens": [128, 128, 64], "head_fcnet_hiddens": [64, 32]},
            # {"fcnet_hiddens": [], "head_fcnet_hiddens": []},  # No hidden layers (not supported by RLlib, causes IndexError)
        ]
        for hidden_config in configs:
            config = self.model_config.copy()
            config.update(hidden_config)
            catalog = JaxDQNCatalog(
                observation_space=self.observation_space,
                action_space=self.action_space,
                model_config_dict=config,
            )
            encoder = catalog.build_encoder(framework="torch")
            af_head = catalog.build_af_head(framework="torch")
            assert encoder is not None
            assert af_head is not None

    @patch_args()
    def test_invalid_action_space(self):
        """Test that catalog rejects continuous action spaces."""
        continuous_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # DQN requires discrete action space
        with pytest.raises((ValueError, AssertionError, TypeError)):
            JaxDQNCatalog(
                observation_space=self.observation_space,
                action_space=continuous_action_space,
                model_config_dict=self.model_config,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
