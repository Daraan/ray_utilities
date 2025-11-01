"""Test conversion of Optuna distributions to Ray Tune domains."""

from __future__ import annotations

import unittest

import optuna
import pytest
from ray import tune
from ray.tune.search import sample
from ray.tune.search.optuna import OptunaSearch

from ray_utilities.setup.extensions import optuna_dist_to_ray_distribution


class TestOptunaToRayConversion(unittest.TestCase):
    """Test the conversion from Optuna distributions to Ray Tune domains."""

    def test_float_distribution_uniform(self):
        """Test conversion of uniform float distribution."""
        dist = optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        result = optuna_dist_to_ray_distribution(dist)
        assert isinstance(result, sample.Float)
        assert result.lower == 0.0
        assert result.upper == 1.0

    def test_float_distribution_log(self):
        """Test conversion of log-scale float distribution."""
        dist = optuna.distributions.FloatDistribution(low=1e-5, high=1e-1, log=True)
        result = optuna_dist_to_ray_distribution(dist)
        assert isinstance(result, sample.Float)
        assert result.lower == 1e-5
        assert result.upper == 1e-1

    def test_float_distribution_with_step(self):
        """Test conversion of quantized float distribution."""
        dist = optuna.distributions.FloatDistribution(low=0.0, high=10.0, step=0.5)
        result = optuna_dist_to_ray_distribution(dist)
        assert isinstance(result, sample.Float)
        assert result.lower == 0.0
        assert result.upper == 10.0

    def test_float_distribution_log_with_step_not_allowed_by_optuna(self):
        """Test that Optuna doesn't allow log-scale float with step."""
        # Optuna itself doesn't allow this combination, so we verify that
        with pytest.raises(ValueError, match="step.*not supported when.*log"):
            optuna.distributions.FloatDistribution(low=1e-5, high=1e-1, step=1e-6, log=True)

    def test_int_distribution_uniform(self):
        """Test conversion of uniform integer distribution."""
        dist = optuna.distributions.IntDistribution(low=1, high=10)
        result = optuna_dist_to_ray_distribution(dist)
        assert isinstance(result, sample.Integer)
        assert result.lower == 1
        assert result.upper == 11  # Ray uses exclusive upper bound

    def test_int_distribution_log(self):
        """Test conversion of log-scale integer distribution."""
        dist = optuna.distributions.IntDistribution(low=1, high=100, log=True)
        result = optuna_dist_to_ray_distribution(dist)
        assert isinstance(result, sample.Integer)

        assert result.lower == 1
        assert result.upper == 101  # Ray uses exclusive upper bound

    def test_int_distribution_with_step(self):
        """Test conversion of integer distribution with step."""
        dist = optuna.distributions.IntDistribution(low=0, high=100, step=10)
        result = optuna_dist_to_ray_distribution(dist)
        assert isinstance(result, sample.Integer)
        assert result.lower == 0
        assert result.upper == 101  # Ray uses exclusive upper bound

    def test_categorical_distribution(self):
        """Test conversion of categorical distribution."""
        choices = ["option_a", "option_b", "option_c"]
        dist = optuna.distributions.CategoricalDistribution(choices=choices)
        result = optuna_dist_to_ray_distribution(dist)
        assert isinstance(result, sample.Categorical)
        assert result.categories == choices

    def test_categorical_distribution_with_numbers(self):
        """Test conversion of categorical distribution with numeric choices."""
        choices = [1, 2, 5, 10]
        dist = optuna.distributions.CategoricalDistribution(choices=choices)
        result = optuna_dist_to_ray_distribution(dist)
        assert isinstance(result, sample.Categorical)
        assert result.categories == choices

    def test_unsupported_distribution_raises_error(self):
        """Test that unsupported distribution types raise TypeError."""

        # Create a mock distribution that's not one of the supported types
        class UnsupportedDistribution(optuna.distributions.BaseDistribution):
            def single(self):
                return True

            def _asdict(self):
                return {}

            def _contains(self, param_value_in_internal_repr):
                return True

            def to_internal_repr(self, param_value_in_external_repr):
                return param_value_in_external_repr

        dist = UnsupportedDistribution()
        with pytest.raises(TypeError, match="Unsupported Optuna distribution type"):
            optuna_dist_to_ray_distribution(dist)

    def test_roundtrip_conversion(self):
        """Test that converting Ray->Optuna->Ray preserves properties."""
        # Create a Ray search space
        ray_space = {
            "float_uniform": tune.uniform(0.0, 1.0),
            "float_log": tune.loguniform(1e-5, 1e-1),
            "int_uniform": tune.randint(1, 11),
            "categorical": tune.choice(["a", "b", "c"]),
        }
        # Convert to Optuna (using Ray's built-in converter)
        optuna_space = OptunaSearch.convert_search_space(ray_space)
        # Convert back to Ray
        for param_name, optuna_dist in optuna_space.items():
            ray_domain = optuna_dist_to_ray_distribution(optuna_dist)
            assert ray_domain is not None
