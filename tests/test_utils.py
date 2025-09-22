"""Tests for utility functions."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import ldx


class TestDataUtils:
    """Test data handling utilities."""

    def test_normalize_data_standard(self):
        """Test standard normalization."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (100, 5)) * 2.0 + 1.0

        normalized, params = ldx.utils.normalize_data(data, method="standard")

        # Check that normalized data has zero mean and unit variance
        assert jnp.allclose(jnp.mean(normalized, axis=0), 0.0, atol=1e-6)
        assert jnp.allclose(jnp.std(normalized, axis=0), 1.0, atol=1e-6)

        # Check parameters
        assert "mean" in params
        assert "std" in params
        assert params["method"] == "standard"

        # Test denormalization
        denormalized = ldx.utils.denormalize_data(normalized, params)
        assert jnp.allclose(denormalized, data, atol=1e-6)

    def test_normalize_data_minmax(self):
        """Test min-max normalization."""
        key = jax.random.PRNGKey(42)
        data = jax.random.uniform(key, (100, 3)) * 10.0 + 5.0

        normalized, params = ldx.utils.normalize_data(data, method="minmax")

        # Check that normalized data is in [0, 1]
        assert jnp.all(normalized >= 0.0)
        assert jnp.all(normalized <= 1.0)

        # Check parameters
        assert "min" in params
        assert "max" in params
        assert params["method"] == "minmax"

        # Test denormalization
        denormalized = ldx.utils.denormalize_data(normalized, params)
        assert jnp.allclose(denormalized, data, atol=1e-6)

    def test_create_sliding_windows(self):
        """Test sliding window creation."""
        seq_len = 100
        obs_dim = 3
        window_size = 10
        stride = 5

        key = jax.random.PRNGKey(42)
        observations = jax.random.normal(key, (seq_len, obs_dim))
        times = jnp.linspace(0, 10, seq_len)

        windowed_obs, windowed_times = ldx.utils.create_sliding_windows(
            observations, times, window_size, stride
        )

        expected_n_windows = (seq_len - window_size) // stride + 1
        assert windowed_obs.shape == (expected_n_windows, window_size, obs_dim)
        assert windowed_times.shape == (expected_n_windows, window_size)

    def test_pad_sequences(self):
        """Test sequence padding."""
        # Create sequences of different lengths
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        sequences = [
            jax.random.normal(keys[0], (10, 4)),
            jax.random.normal(keys[1], (15, 4)),
            jax.random.normal(keys[2], (8, 4))
        ]

        times = [
            jnp.linspace(0, 1, 10),
            jnp.linspace(0, 1.5, 15),
            jnp.linspace(0, 0.8, 8)
        ]

        padded_sequences, padded_times, mask = ldx.utils.pad_sequences(
            sequences, times, pad_value=-999.0
        )

        max_len = 15
        assert padded_sequences.shape == (3, max_len, 4)
        assert padded_times.shape == (3, max_len)
        assert mask.shape == (3, max_len)

        # Check mask correctness
        assert jnp.sum(mask[0]) == 10
        assert jnp.sum(mask[1]) == 15
        assert jnp.sum(mask[2]) == 8

    def test_create_train_val_split(self):
        """Test train/validation split."""
        key = jax.random.PRNGKey(42)
        n_sequences = 100
        observations = jax.random.normal(key, (n_sequences, 20, 3))
        times = jnp.tile(jnp.linspace(0, 2, 20), (n_sequences, 1))

        (train_obs, train_times), (val_obs, val_times) = ldx.utils.create_train_val_split(
            observations, times, val_fraction=0.2, shuffle=True
        )

        assert train_obs.shape[0] == 80
        assert val_obs.shape[0] == 20
        assert train_obs.shape[1:] == observations.shape[1:]
        assert val_obs.shape[1:] == observations.shape[1:]

    def test_time_series_dataset(self):
        """Test TimeSeriesDataset."""
        key = jax.random.PRNGKey(42)
        n_sequences = 20
        seq_len = 50
        obs_dim = 3

        observations = jax.random.normal(key, (n_sequences, seq_len, obs_dim))
        times = jnp.tile(jnp.linspace(0, 5, seq_len), (n_sequences, 1))

        dataset = ldx.utils.TimeSeriesDataset(
            observations=np.array(observations),
            times=np.array(times),
            batch_size=5,
            shuffle=True
        )

        assert len(dataset) == 4  # 20 sequences / 5 batch_size

        # Test iteration
        batch_count = 0
        for batch_obs, batch_times in dataset:
            assert batch_obs.shape[0] <= 5  # Last batch might be smaller
            assert batch_obs.shape[1:] == (seq_len, obs_dim)
            batch_count += 1

        assert batch_count == 4


class TestMetrics:
    """Test evaluation metrics."""

    def test_basic_metrics(self):
        """Test basic regression metrics."""
        key = jax.random.PRNGKey(42)
        predictions = jax.random.normal(key, (100, 10, 3))
        targets = predictions + 0.1 * jax.random.normal(jax.random.split(key)[1], (100, 10, 3))

        # Test individual metrics
        mse = ldx.utils.mse(predictions, targets)
        mae = ldx.utils.mae(predictions, targets)
        rmse = ldx.utils.rmse(predictions, targets)
        r2 = ldx.utils.r2_score(predictions, targets)

        assert mse > 0
        assert mae > 0
        assert rmse > 0
        assert r2 <= 1.0

        # RMSE should be sqrt of MSE
        assert jnp.allclose(rmse, jnp.sqrt(mse))

    def test_metrics_with_mask(self):
        """Test metrics with missing data mask."""
        key = jax.random.PRNGKey(42)
        predictions = jax.random.normal(key, (50, 20, 2))
        targets = jax.random.normal(jax.random.split(key)[1], (50, 20, 2))

        # Create random mask
        mask = jax.random.bernoulli(jax.random.split(key)[0], 0.8, (50, 20))

        mse_masked = ldx.utils.mse(predictions, targets, mask)
        mse_full = ldx.utils.mse(predictions, targets)

        # Masked MSE should be different from full MSE
        assert not jnp.allclose(mse_masked, mse_full)

    def test_gaussian_metrics(self):
        """Test metrics for Gaussian predictions."""
        key = jax.random.PRNGKey(42)
        mean = jax.random.normal(key, (50, 10, 3))
        logvar = jax.random.normal(jax.random.split(key)[1], (50, 10, 3)) * 0.1
        targets = mean + 0.1 * jax.random.normal(jax.random.split(key)[0], (50, 10, 3))

        predictions = (mean, logvar)

        # Test log-likelihood
        log_lik = ldx.utils.log_likelihood_gaussian(predictions, targets)
        assert jnp.isfinite(log_lik)

        # Test prediction interval coverage
        coverage_95 = ldx.utils.prediction_interval_coverage(
            predictions, targets, confidence_level=0.95
        )
        coverage_90 = ldx.utils.prediction_interval_coverage(
            predictions, targets, confidence_level=0.90
        )

        assert 0.0 <= coverage_95 <= 1.0
        assert 0.0 <= coverage_90 <= 1.0
        assert coverage_90 <= coverage_95  # 90% interval should have lower coverage

    def test_compute_all_metrics(self):
        """Test comprehensive metrics computation."""
        key = jax.random.PRNGKey(42)
        predictions = jax.random.normal(key, (30, 15, 2))
        targets = predictions + 0.05 * jax.random.normal(jax.random.split(key)[1], (30, 15, 2))

        # Test non-probabilistic metrics
        metrics = ldx.utils.compute_all_metrics(predictions, targets)

        expected_keys = ["mse", "mae", "rmse", "mape", "r2", "correlation"]
        for key in expected_keys:
            assert key in metrics
            assert jnp.isfinite(metrics[key])

        # Test probabilistic metrics
        mean = predictions
        logvar = jnp.ones_like(predictions) * (-2.0)  # Small variance
        prob_predictions = (mean, logvar)

        prob_metrics = ldx.utils.compute_all_metrics(
            prob_predictions, targets, probabilistic=True
        )

        expected_prob_keys = ["mse", "mae", "rmse", "r2", "log_likelihood", "coverage_95", "coverage_90"]
        for key in expected_prob_keys:
            assert key in prob_metrics

    def test_forecast_metrics(self):
        """Test forecast horizon metrics."""
        key = jax.random.PRNGKey(42)
        horizon_steps = 10
        predictions = jax.random.normal(key, (20, horizon_steps, 2))
        targets = jax.random.normal(jax.random.split(key)[1], (20, horizon_steps, 2))

        forecast_metrics = ldx.utils.forecast_metrics(
            predictions, targets, horizon_steps
        )

        assert "mse_by_step" in forecast_metrics
        assert "mae_by_step" in forecast_metrics
        assert "rmse_by_step" in forecast_metrics

        for metric_name, values in forecast_metrics.items():
            assert values.shape == (horizon_steps,)
            assert jnp.all(jnp.isfinite(values))


class TestDatasets:
    """Test synthetic dataset generation."""

    def test_spiral_trajectory(self):
        """Test spiral trajectory generation."""
        key = jax.random.PRNGKey(42)
        times, observations = ldx.datasets.generate_spiral_trajectory(
            n_points=100, n_dims=2, noise_std=0.1, key=key
        )

        assert times.shape == (100,)
        assert observations.shape == (100, 2)
        assert jnp.all(jnp.isfinite(times))
        assert jnp.all(jnp.isfinite(observations))

    def test_lorenz_trajectory(self):
        """Test Lorenz attractor generation."""
        key = jax.random.PRNGKey(42)
        times, observations = ldx.datasets.generate_lorenz_trajectory(
            n_points=200, dt=0.01, key=key
        )

        assert times.shape == (200,)
        assert observations.shape == (200, 3)
        assert jnp.all(jnp.isfinite(observations))

    def test_pendulum_trajectory(self):
        """Test pendulum trajectory generation."""
        key = jax.random.PRNGKey(42)
        times, observations = ldx.datasets.generate_pendulum_trajectory(
            n_points=100, dt=0.05, key=key
        )

        assert times.shape == (100,)
        assert observations.shape == (100, 2)  # x, y coordinates

    def test_oscillator_trajectory(self):
        """Test harmonic oscillator generation."""
        key = jax.random.PRNGKey(42)
        times, observations = ldx.datasets.generate_oscillator_trajectory(
            n_points=150, dt=0.1, key=key
        )

        assert times.shape == (150,)
        assert observations.shape == (150, 2)  # position, velocity

    def test_irregular_observations(self):
        """Test irregular observation generation."""
        key = jax.random.PRNGKey(42)

        # First generate regular trajectory
        times, observations = ldx.datasets.generate_spiral_trajectory(
            n_points=100, key=key
        )

        # Create irregular observations
        observed_times, observed_data, mask = ldx.datasets.generate_irregular_observations(
            times, observations, observation_prob=0.7, key=key
        )

        assert len(observed_times) <= len(times)
        assert observed_data.shape[0] == len(observed_times)
        assert observed_data.shape[1] == observations.shape[1]
        assert mask.shape == (len(times),)
        assert jnp.sum(mask) == len(observed_times)

    def test_partially_observed_system(self):
        """Test partially observed system generation."""
        key = jax.random.PRNGKey(42)

        true_dim = 5
        obs_dim = 3
        n_points = 100

        times, true_states, observations, obs_matrix = ldx.datasets.generate_partially_observed_system(
            true_dim=true_dim,
            obs_dim=obs_dim,
            n_points=n_points,
            key=key
        )

        assert times.shape == (n_points,)
        assert true_states.shape == (n_points, true_dim)
        assert observations.shape == (n_points, obs_dim)
        assert obs_matrix.shape == (obs_dim, true_dim)

    def test_synthetic_dataset_class(self):
        """Test SyntheticDataset convenience class."""
        # Test that all methods work
        times, obs = ldx.datasets.SyntheticDataset.spiral(n_points=50)
        assert times.shape == (50,)

        times, obs = ldx.datasets.SyntheticDataset.lorenz(n_points=100)
        assert obs.shape == (100, 3)

        times, obs = ldx.datasets.SyntheticDataset.pendulum(n_points=75)
        assert obs.shape == (75, 2)

        times, obs = ldx.datasets.SyntheticDataset.oscillator(n_points=80)
        assert obs.shape == (80, 2)

        result = ldx.datasets.SyntheticDataset.partially_observed(
            true_dim=4, obs_dim=2, n_points=60
        )
        times, true_states, observations, obs_matrix = result
        assert true_states.shape == (60, 4)
        assert observations.shape == (60, 2)


if __name__ == "__main__":
    pytest.main([__file__])