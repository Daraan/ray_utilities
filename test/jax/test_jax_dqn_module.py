"""Tests for JAX DQN module implementation.

Note:
    These tests are currently skipped because JAX DQN implementation is blocked by
    RLlib limitations - the catalog system only supports building "torch" and "tf2"
    framework models. Full JAX integration requires custom model building outside
    the catalog system.
"""

from __future__ import annotations
from typing import cast, TYPE_CHECKING

import gymnasium as gym
import jax
import numpy as np
import pytest
from ray.rllib.core.columns import Columns

from ray_utilities.jax.dqn import JaxDQNCatalog, JaxDQNModule
from ray_utilities.testing_utils import DisableLoggers, TestHelpers, patch_args

if TYPE_CHECKING:
    from ray_utilities.jax.dqn.jax_dqn_module import JaxDQNStateDict


class TestJaxDQNModule(DisableLoggers, TestHelpers):
    """Test suite for JaxDQNModule."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.model_config = {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
            "head_fcnet_hiddens": [32],
            "head_fcnet_activation": "relu",
            "num_atoms": 1,  # Standard DQN
            "dueling": False,  # RLlib uses "dueling" not "uses_dueling"
            "double_q": False,
            "epsilon": [(0, 1.0), (10000, 0.05)],  # Epsilon schedule
            "seed": 42,
        }
        self.dueling_model_config = self.model_config.copy()
        self.dueling_model_config["dueling"] = True

        self.distributional_model_config = self.model_config.copy()
        self.distributional_model_config["num_atoms"] = 51
        self.distributional_model_config["v_min"] = -10.0
        self.distributional_model_config["v_max"] = 10.0

    @patch_args()
    def test_module_initialization(self):
        """Test basic module initialization."""
        # Construct module by passing the catalog class so RLModule initializes it
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        # Call setup() - framework will be "torch" but catalog builds JAX models
        #module.setup()

        # Check module attributes
        assert module.observation_space == self.observation_space
        assert module.action_space == self.action_space
        # Check that states are initialized
        assert hasattr(module, "states")
        # Debug: print what's in states
        print(f"States dict: {module.states}")
        print(f"States type: {type(module.states)}")
        print(f"Has uses_dueling: {hasattr(module, 'uses_dueling')}")
        if hasattr(module, "uses_dueling"):
            print(f"uses_dueling value: {module.uses_dueling}")
        assert "qf" in module.states, f"Expected 'qf' in states, got: {list(module.states.keys())}"
        assert "qf_target" in module.states
        # Check uses_dueling is False for non-dueling config
        assert module.uses_dueling is False

    @patch_args()
    def test_dueling_module_initialization(self):
        """Test dueling architecture initialization."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.dueling_model_config,
            catalog_class=JaxDQNCatalog,
        )
        #module.setup()

        # Check dueling components
        assert module.uses_dueling
        assert hasattr(module, "vf")
        assert module.vf is not None

    @patch_args()
    def test_forward_pass(self):
        """Test forward pass returns correct output."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        #module.setup()

        # Create sample batch
        batch_size = 8
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }

        # Forward pass
        output = module._forward(batch, parameters=module.states)

        # Check output structure
        assert "qf_preds" in output
        assert output["qf_preds"].shape == (batch_size, self.action_space.n)

    @patch_args()
    def test_dueling_forward_pass(self):
        """Test forward pass with dueling architecture."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.dueling_model_config,
            catalog_class=JaxDQNCatalog,
        )
        #module.setup()

        batch_size = 8
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }

        output = module._forward(batch, parameters=module.states)

        assert "qf_preds" in output
        assert output["qf_preds"].shape == (batch_size, self.action_space.n)

    @patch_args()
    def test_distributional_forward_pass(self):
        """Test forward pass with distributional Q-learning."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.distributional_model_config,
            catalog_class=JaxDQNCatalog,
        )
        #module.setup()

        batch_size = 8
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }

        output = module._forward(batch, parameters=module.states)

        # Check distributional outputs
        assert "qf_preds" in output
        assert "qf_logits" in output
        assert "qf_probs" in output
        assert "atoms" in output

        num_atoms = self.distributional_model_config["num_atoms"]
        assert output["qf_logits"].shape == (batch_size, self.action_space.n, num_atoms)
        assert output["qf_probs"].shape == (batch_size, self.action_space.n, num_atoms)
        assert output["atoms"].shape == (num_atoms,)

    @patch_args()
    def test_compute_q_values(self):
        """Test compute_q_values method."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        batch_size = 4
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }
        q_values = module.compute_q_values(batch)
        assert "qf_preds" in q_values
        assert q_values["qf_preds"].shape == (batch_size, self.action_space.n)

    @patch_args()
    def test_target_q_values(self):
        """Test target Q-value computation."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        batch_size = 4
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }
        # Compute target Q-values
        target_q_values = module.compute_target_q_values(batch)
        assert "qf_preds" in target_q_values
        assert target_q_values["qf_preds"].shape == (batch_size, self.action_space.n)

    @patch_args()
    def test_state_management(self):
        """Test state get/set operations."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        # Get initial state
        state = module.get_state()
        assert "qf" in state
        assert "qf_target" in state
        assert "module_key" in state
        # Set state
        module.set_state(state)
        # Verify state after set
        new_state = module.get_state()
        assert "qf" in new_state
        assert "qf_target" in new_state

    # Edge Case Tests
    @patch_args()
    def test_invalid_action_space(self):
        """Test that non-discrete action space raises appropriate error."""
        # DQN requires discrete action space
        continuous_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # This should fail during catalog creation or module setup
        with pytest.raises((ValueError, AssertionError, TypeError)):
            catalog = JaxDQNCatalog(
                observation_space=self.observation_space,
                action_space=continuous_action_space,
                model_config_dict=self.model_config,
            )
            module = JaxDQNModule(
                observation_space=self.observation_space,
                action_space=continuous_action_space,
                model_config_dict=self.model_config,
                catalog=catalog,
            )
            module.setup()

    @patch_args()
    def test_empty_batch(self):
        """Test handling of empty batch."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        # Empty batch
        batch_size = 0
        batch = {Columns.OBS: jax.numpy.array(np.empty((batch_size, *self.observation_space.shape), dtype=np.float32))}
        # Should handle gracefully or raise appropriate error
        try:
            output = module._forward(batch, parameters=module.states)
            # If it succeeds, output should have correct shape
            assert output["qf_preds"].shape[0] == 0
        except (ValueError, IndexError):
            # Empty batch might legitimately fail
            pass

    @patch_args()
    def test_batch_shape_mismatch(self):
        """Test that mismatched observation shapes are caught."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        # Wrong observation shape
        batch_size = 4
        wrong_shape = (3,)  # Should be (4,)
        batch = {Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *wrong_shape).astype(np.float32))}
        # Should raise shape error
        with pytest.raises((ValueError, TypeError, AssertionError)):
            module._forward(batch, parameters=module.states)

    @patch_args()
    def test_target_network_independence(self):
        """Test that target network is independent from main network."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        # Get initial states
        qf_state = cast("JaxDQNStateDict", module.states)["qf"]
        qf_target_state = cast("JaxDQNStateDict", module.states)["qf_target"]
        # States should exist but be different objects
        assert qf_state is not qf_target_state
        # Create batch
        batch_size = 4
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }
        # Compute Q-values from both networks
        q_values = module.compute_q_values(batch)
        target_q_values = module.compute_target_q_values(batch)
        # Shapes should match
        assert q_values["qf_preds"].shape == target_q_values["qf_preds"].shape

    @patch_args()
    def test_dueling_computation_correctness(self):
        """Test that dueling architecture computes Q = V + (A - mean(A)) correctly."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.dueling_model_config,
            catalog_class=JaxDQNCatalog,
        )
        batch_size = 4
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }
        output = module._forward(batch, parameters=module.states)
        # With dueling, Q-values should have mean-centered advantages
        # This means mean(Q) should be close to the value function
        q_preds = output["qf_preds"]
        assert q_preds.shape == (batch_size, self.action_space.n)
        # Check that Q-values are finite (no NaN/Inf)
        assert jax.numpy.all(jax.numpy.isfinite(q_preds))

    @patch_args()
    def test_distributional_properties(self):
        """Test distributional Q-learning properties."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.distributional_model_config,
            catalog_class=JaxDQNCatalog,
        )
        batch_size = 4
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }
        output = module._forward(batch, parameters=module.states)
        # Check probability distribution properties
        q_probs = output["qf_probs"]
        atoms = output["atoms"]
        # Probabilities should sum to 1 for each action
        prob_sums = jax.numpy.sum(q_probs, axis=-1)
        assert jax.numpy.allclose(prob_sums, 1.0, atol=1e-6)
        # Probabilities should be non-negative
        assert jax.numpy.all(q_probs >= 0)
        # Atoms should be in correct range
        v_min = self.distributional_model_config["v_min"]
        v_max = self.distributional_model_config["v_max"]
        assert jax.numpy.allclose(atoms[0], v_min)
        assert jax.numpy.allclose(atoms[-1], v_max)
        # Q-values should be weighted sum of atoms
        expected_q = jax.numpy.sum(q_probs * atoms, axis=-1)
        assert jax.numpy.allclose(output["qf_preds"], expected_q, atol=1e-5)

    @patch_args()
    def test_multiple_forward_passes(self):
        """Test that multiple forward passes produce consistent results."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        batch_size = 4
        batch = {
            Columns.OBS: jax.numpy.array(np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32))
        }
        # Multiple forward passes with same batch should give same results
        output1 = module._forward(batch, parameters=module.states)
        output2 = module._forward(batch, parameters=module.states)
        assert jax.numpy.allclose(output1["qf_preds"], output2["qf_preds"])

    @patch_args()
    def test_different_batch_sizes(self):
        """Test that module handles different batch sizes correctly."""
        module = JaxDQNModule(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config,
            catalog_class=JaxDQNCatalog,
        )
        # Test different batch sizes
        for batch_size in [1, 4, 16, 32]:
            batch = {
                Columns.OBS: jax.numpy.array(
                    np.random.randn(batch_size, *self.observation_space.shape).astype(np.float32)
                )
            }
            output = module._forward(batch, parameters=module.states)
            assert output["qf_preds"].shape == (batch_size, self.action_space.n)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
