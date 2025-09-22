"""Tests for inference methods."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import ldx


class TestVariationalInference:
    """Test variational inference components."""

    def test_kl_divergence_gaussian(self):
        """Test KL divergence computation."""
        batch_size, latent_dim = 10, 5

        # Standard normal should have zero KL
        mean = jnp.zeros((batch_size, latent_dim))
        logvar = jnp.zeros((batch_size, latent_dim))

        kl = ldx.inference.kl_divergence_gaussian(mean, logvar)

        assert kl.shape == (batch_size,)
        assert jnp.allclose(kl, 0.0, atol=1e-6)

        # Non-zero mean should increase KL
        mean_nonzero = jnp.ones((batch_size, latent_dim))
        kl_nonzero = ldx.inference.kl_divergence_gaussian(mean_nonzero, logvar)

        assert jnp.all(kl_nonzero > kl)

    def test_reconstruction_loss_mse(self):
        """Test MSE reconstruction loss."""
        batch_size, seq_len, obs_dim = 5, 20, 3

        predictions = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_len, obs_dim))
        targets = jax.random.normal(jax.random.PRNGKey(43), (batch_size, seq_len, obs_dim))

        loss = ldx.inference.reconstruction_loss_mse(predictions, targets)

        assert loss.shape == (batch_size,)
        assert jnp.all(loss >= 0)

        # Perfect reconstruction should give zero loss
        perfect_loss = ldx.inference.reconstruction_loss_mse(targets, targets)
        assert jnp.allclose(perfect_loss, 0.0, atol=1e-6)

    def test_reconstruction_loss_gaussian(self):
        """Test Gaussian reconstruction loss."""
        batch_size, seq_len, obs_dim = 5, 20, 3

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)

        mean = jax.random.normal(key1, (batch_size, seq_len, obs_dim))
        logvar = jax.random.normal(key2, (batch_size, seq_len, obs_dim)) * 0.1
        targets = jax.random.normal(key3, (batch_size, seq_len, obs_dim))

        predictions = (mean, logvar)
        loss = ldx.inference.reconstruction_loss_gaussian(predictions, targets)

        assert loss.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(loss))

    def test_beta_schedules(self):
        """Test beta scheduling functions."""
        # Constant schedule
        constant_schedule = ldx.inference.create_beta_schedule("constant", beta=0.5)
        assert constant_schedule(0) == 0.5
        assert constant_schedule(100) == 0.5

        # Linear schedule
        linear_schedule = ldx.inference.create_beta_schedule(
            "linear", start_beta=0.0, end_beta=1.0, warmup_steps=100
        )
        assert linear_schedule(0) == 0.0
        assert linear_schedule(50) == 0.5
        assert linear_schedule(100) == 1.0
        assert linear_schedule(150) == 1.0

        # Cyclical schedule
        cyclical_schedule = ldx.inference.create_beta_schedule(
            "cyclical", beta_max=1.0, cycle_length=100
        )
        assert cyclical_schedule(0) == 0.0
        assert cyclical_schedule(25) == 0.5
        assert cyclical_schedule(50) == 1.0
        assert cyclical_schedule(75) == 0.5

    def test_variational_trainer_creation(self):
        """Test VariationalTrainer creation."""
        key = jax.random.PRNGKey(42)
        from flax import nnx
        import optax

        # Create dummy model
        rngs = nnx.Rngs(key)
        model = ldx.models.create_latent_ode(
            latent_dim=4,
            obs_dim=2,
            hidden_dims=(8,),
            rngs=rngs
        )

        optimizer = optax.adam(1e-3)

        trainer = ldx.inference.VariationalTrainer(
            model=model,
            optimizer=optimizer,
            beta_schedule=ldx.inference.create_beta_schedule("linear")
        )

        assert trainer.model == model
        assert trainer.optimizer == optimizer


class TestFiltering:
    """Test filtering methods."""

    def test_filter_state(self):
        """Test FilterState structure."""
        latent_dim = 4
        mean = jnp.zeros(latent_dim)
        cov = jnp.eye(latent_dim)
        time = 0.0

        state = ldx.inference.FilterState(mean=mean, cov=cov, time=time)

        assert state.mean.shape == (latent_dim,)
        assert state.cov.shape == (latent_dim, latent_dim)
        assert state.time == 0.0

    def test_extended_kalman_filter(self):
        """Test Extended Kalman Filter."""
        latent_dim = 2
        obs_dim = 1

        # Simple linear dynamics and observation functions
        def dynamics_func(x, t):
            return jnp.array([-0.1 * x[0], -0.1 * x[1]])

        def observation_func(x, t):
            return jnp.array([x[0]])

        # Noise covariances
        Q = 0.01 * jnp.eye(latent_dim)
        R = 0.1 * jnp.eye(obs_dim)

        ekf = ldx.inference.ExtendedKalmanFilter(
            dynamics_func, observation_func, Q, R
        )

        # Test prediction step
        initial_state = ldx.inference.FilterState(
            mean=jnp.zeros(latent_dim),
            cov=jnp.eye(latent_dim),
            time=0.0
        )

        predicted_state = ekf.predict(initial_state, dt=0.1)

        assert predicted_state.mean.shape == (latent_dim,)
        assert predicted_state.cov.shape == (latent_dim, latent_dim)
        assert predicted_state.time == 0.1

        # Test update step
        observation = jnp.array([0.1])
        updated_state = ekf.update(predicted_state, observation)

        assert updated_state.mean.shape == (latent_dim,)
        assert updated_state.cov.shape == (latent_dim, latent_dim)

    def test_particle_filter(self):
        """Test Particle Filter."""
        latent_dim = 2
        n_particles = 50

        def dynamics_func(x, t):
            return jnp.array([-0.1 * x[0], -0.1 * x[1]])

        def observation_func(x, t):
            return x[0:1]  # Observe first dimension

        pf = ldx.inference.ParticleFilter(
            dynamics_func=dynamics_func,
            observation_func=observation_func,
            process_noise_std=0.1,
            observation_noise_std=0.1,
            n_particles=n_particles
        )

        # Test particle prediction
        key = jax.random.PRNGKey(42)
        particles = jax.random.normal(key, (n_particles, latent_dim))
        time = 0.0
        dt = 0.1

        key, pred_key = jax.random.split(key)
        predicted_particles = pf.predict_particles(particles, time, dt, pred_key)

        assert predicted_particles.shape == (n_particles, latent_dim)

        # Test weight computation
        observation = jnp.array([0.0])
        weights = pf.compute_weights(predicted_particles, observation, time + dt)

        assert weights.shape == (n_particles,)
        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert jnp.all(weights >= 0)


class TestSmoothing:
    """Test smoothing methods."""

    def test_rauch_tung_striebel(self):
        """Test Rauch-Tung-Striebel smoother."""
        latent_dim = 2
        seq_len = 10

        # Create EKF for smoother
        def dynamics_func(x, t):
            return -0.1 * x

        def observation_func(x, t):
            return x

        Q = 0.01 * jnp.eye(latent_dim)
        R = 0.1 * jnp.eye(latent_dim)

        ekf = ldx.inference.ExtendedKalmanFilter(
            dynamics_func, observation_func, Q, R
        )

        smoother = ldx.inference.RauchTungStriebel(ekf)

        # Create dummy filtered estimates
        key = jax.random.PRNGKey(42)
        filtered_means = jax.random.normal(key, (seq_len, latent_dim)) * 0.1
        filtered_covs = jnp.tile(jnp.eye(latent_dim), (seq_len, 1, 1)) * 0.1
        times = jnp.linspace(0, 1, seq_len)

        # Test smoothing
        result = smoother.smooth_sequence(filtered_means, filtered_covs, times)

        assert isinstance(result, ldx.inference.SmoothingResult)
        assert result.means.shape == (seq_len, latent_dim)
        assert result.covariances.shape == (seq_len, latent_dim, latent_dim)
        assert result.times.shape == (seq_len,)

    def test_particle_smoother(self):
        """Test Particle Smoother."""
        latent_dim = 2
        n_particles = 20
        seq_len = 5

        def dynamics_func(x, t):
            return -0.1 * x

        def observation_func(x, t):
            return x[0:1]

        smoother = ldx.inference.ParticleSmoother(
            dynamics_func=dynamics_func,
            observation_func=observation_func,
            process_noise_std=0.1,
            observation_noise_std=0.1,
            n_particles=n_particles
        )

        # Create dummy forward filter results
        key = jax.random.PRNGKey(42)
        forward_particles = jax.random.normal(key, (seq_len, n_particles, latent_dim))
        forward_weights = jnp.ones((seq_len, n_particles)) / n_particles
        times = jnp.linspace(0, 1, seq_len)

        # Test smoothing
        key, smooth_key = jax.random.split(key)
        smoothed_means, trajectories = smoother.smooth_sequence(
            forward_particles, forward_weights, times, smooth_key
        )

        assert smoothed_means.shape == (seq_len, latent_dim)
        assert trajectories.shape == (n_particles, seq_len, latent_dim)


class TestInferenceIntegration:
    """Integration tests for inference methods."""

    def test_ekf_with_synthetic_data(self):
        """Test EKF with synthetic data."""
        # Generate simple linear system data
        key = jax.random.PRNGKey(42)
        times, observations = ldx.datasets.generate_oscillator_trajectory(
            n_points=50, dt=0.1, key=key
        )

        # Simple dynamics for testing
        def dynamics_func(x, t):
            return jnp.array([x[1], -x[0]])  # Harmonic oscillator

        def observation_func(x, t):
            return x  # Fully observed

        latent_dim = observations.shape[1]
        Q = 0.01 * jnp.eye(latent_dim)
        R = 0.1 * jnp.eye(latent_dim)

        ekf = ldx.inference.ExtendedKalmanFilter(
            dynamics_func, observation_func, Q, R
        )

        # Initial state
        initial_state = ldx.inference.FilterState(
            mean=observations[0],
            cov=jnp.eye(latent_dim),
            time=times[0]
        )

        # Run filter
        filtered_means, filtered_covs = ekf.filter_sequence(
            initial_state, observations, times
        )

        assert filtered_means.shape == observations.shape
        assert filtered_covs.shape == (len(times), latent_dim, latent_dim)

        # Check that covariances are positive definite
        for i in range(len(times)):
            eigenvals = jnp.linalg.eigvals(filtered_covs[i])
            assert jnp.all(eigenvals > 0)


if __name__ == "__main__":
    pytest.main([__file__])