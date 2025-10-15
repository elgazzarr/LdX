"""Tests for model components."""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

import ldx


class TestEncoders:
    """Test encoder components."""

    def test_mlp_encoder(self):
        """Test MLP encoder."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        encoder = ldx.models.MLPEncoder(
            hidden_dims=(32, 16),
            latent_dim=8,
            obs_dim=4,
            rngs=rngs
        )

        # Test with batch of observations
        batch_size = 10
        obs = jax.random.normal(key, (batch_size, 4))

        mean, logvar = encoder(obs)

        assert mean.shape == (batch_size, 8)
        assert logvar.shape == (batch_size, 8)

    def test_rnn_encoder(self):
        """Test RNN encoder."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        encoder = ldx.models.RNNEncoder(
            hidden_dim=32,
            latent_dim=8,
            obs_dim=4,
            rngs=rngs
        )

        # Test with sequential observations
        batch_size, seq_len = 5, 20
        obs = jax.random.normal(key, (batch_size, seq_len, 4))

        mean, logvar = encoder(obs)

        assert mean.shape == (batch_size, 8)
        assert logvar.shape == (batch_size, 8)


class TestDecoders:
    """Test decoder components."""

    def test_mlp_decoder(self):
        """Test MLP decoder."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        decoder = ldx.models.MLPDecoder(
            hidden_dims=(16, 32),
            obs_dim=4,
            latent_dim=8,
            rngs=rngs
        )

        # Test with batch of latent states
        batch_size = 10
        latent = jax.random.normal(key, (batch_size, 8))

        output = decoder(latent)

        assert output.shape == (batch_size, 4)

    def test_gaussian_decoder(self):
        """Test Gaussian decoder."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        decoder = ldx.models.GaussianDecoder(
            hidden_dims=(16, 32),
            obs_dim=4,
            latent_dim=8,
            rngs=rngs
        )

        # Test with batch of latent states
        batch_size = 10
        latent = jax.random.normal(key, (batch_size, 8))

        mean, logvar = decoder(latent)

        assert mean.shape == (batch_size, 4)
        assert logvar.shape == (batch_size, 4)


class TestDynamics:
    """Test dynamics functions."""

    def test_mlp_dynamics(self):
        """Test MLP dynamics function."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        latent_dim = 6
        dynamics = ldx.models.MLPDynamics(
            hidden_dims=(32, 32),
            latent_dim=latent_dim,
            rngs=rngs
        )

        # Test with batch of states and times
        batch_size, latent_dim = 10, 6
        z = jax.random.normal(key, (batch_size, latent_dim))
        t = jnp.linspace(0, 1, batch_size)

        dz_dt = dynamics(z, t)

        assert dz_dt.shape == (batch_size, latent_dim)

    def test_mlp_dynamics_scalar_time(self):
        """Test MLP dynamics with scalar time."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        latent_dim = 4
        dynamics = ldx.models.MLPDynamics(
            hidden_dims=(16,),
            latent_dim=latent_dim,
            rngs=rngs
        )

        # Test with scalar time
        batch_size, latent_dim = 5, 4
        z = jax.random.normal(key, (batch_size, latent_dim))
        t = 0.5

        dz_dt = dynamics(z, t)

        assert dz_dt.shape == (batch_size, latent_dim)


class TestLatentODE:
    """Test LatentODE model."""

    def test_create_latent_ode(self):
        """Test LatentODE creation."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        model = ldx.models.create_latent_ode(
            latent_dim=6,
            obs_dim=3,
            hidden_dims=(32, 16),
            dynamics_hidden_dims=(32,),
            rngs=rngs
        )

        assert isinstance(model, ldx.models.LatentODE)
        assert model.latent_dim == 6

    def test_latent_ode_forward(self):
        """Test LatentODE forward pass."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        model = ldx.models.create_latent_ode(
            latent_dim=4,
            obs_dim=2,
            hidden_dims=(16,),
            dynamics_hidden_dims=(16,),
            rngs=rngs
        )

        # Create test data
        batch_size, seq_len = 3, 20
        observations = jax.random.normal(key, (batch_size, seq_len, 2))
        times = jnp.linspace(0, 2, seq_len)

        key, forward_key = jax.random.split(key)

        # Forward pass
        reconstructed, mean, logvar, trajectory = model.forward(
            observations, times, forward_key
        )

        assert reconstructed.shape == (batch_size, seq_len, 2)
        assert mean.shape == (batch_size, 4)
        assert logvar.shape == (batch_size, 4)
        assert trajectory.shape == (batch_size, seq_len, 4)


class TestLatentSDE:
    """Test LatentSDE model."""

    def test_create_latent_sde(self):
        """Test LatentSDE creation."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        model = ldx.models.create_latent_sde(
            latent_dim=4,
            obs_dim=2,
            hidden_dims=(16,),
            posterior_hidden_dims=(16,),
            prior_hidden_dims=(16,),
            diffusion_hidden_dims=(8,),
            context_dim=16,
            rngs=rngs
        )

        assert isinstance(model, ldx.models.LatentSDE)
        assert model.latent_dim == 4
        assert hasattr(model, 'posterior_drift')
        assert hasattr(model, 'prior_drift')
        assert hasattr(model, 'obs_encoder')

    def test_diagonal_diffusion(self):
        """Test diagonal diffusion function."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        latent_dim = 4
        diffusion = ldx.models.DiagonalDiffusion(
            hidden_dims=(16,),
            latent_dim=latent_dim,
            rngs=rngs
        )

        # Test with batch of states
        batch_size, latent_dim = 5, 4
        z = jax.random.normal(key, (batch_size, latent_dim))
        t = jnp.linspace(0, 1, batch_size)

        diffusion_coeffs = diffusion(z, t)

        assert diffusion_coeffs.shape == (batch_size, latent_dim)
        # Should be positive (softplus activation)
        assert jnp.all(diffusion_coeffs > 0)

    def test_latent_sde_encoder(self):
        """Test LatentSDE observation encoder."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        encoder = ldx.models.LatentSDEEncoder(
            obs_dim=3,
            hidden_dim=32,
            output_dim=16,
            rngs=rngs
        )

        # Test encoding
        batch_size = 5
        obs = jax.random.normal(key, (batch_size, 3))
        times = jnp.linspace(0, 1, batch_size)

        context = encoder(obs, times)

        assert context.shape == (batch_size, 16)
        assert jnp.isfinite(context).all()

    def test_posterior_prior_drift(self):
        """Test posterior and prior drift functions."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        latent_dim = 4
        context_dim = 8

        # Test posterior drift
        posterior = ldx.models.PosteriorDrift(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dims=(16,),
            rngs=rngs
        )

        batch_size = 3
        z = jax.random.normal(key, (batch_size, latent_dim))
        t = jnp.array([0.0, 0.5, 1.0])
        context = jax.random.normal(key, (batch_size, context_dim))

        drift_post = posterior(z, t, context)
        assert drift_post.shape == (batch_size, latent_dim)
        assert jnp.isfinite(drift_post).all()

        # Test prior drift
        prior = ldx.models.PriorDrift(
            latent_dim=latent_dim,
            hidden_dims=(16,),
            use_linear=False,
            rngs=rngs
        )

        drift_prior = prior(z, t)
        assert drift_prior.shape == (batch_size, latent_dim)
        assert jnp.isfinite(drift_prior).all()

    def test_latent_sde_forward_with_kl(self):
        """Test LatentSDE forward pass with KL computation."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        model = ldx.models.create_latent_sde(
            latent_dim=4,
            obs_dim=2,
            hidden_dims=(16,),
            posterior_hidden_dims=(16,),
            prior_hidden_dims=(16,),
            diffusion_hidden_dims=(8,),
            context_dim=8,
            rngs=rngs
        )

        # Create test data
        seq_len = 20
        observations = jax.random.normal(key, (seq_len, 2))
        times = jnp.linspace(0, 1, seq_len)

        key, forward_key = jax.random.split(key)

        # Forward pass with KL
        reconstructed, mean, logvar, trajectory, kl_div = model.forward(
            observations, times, forward_key, return_kl=True
        )

        assert reconstructed.shape == (seq_len, 2)
        assert mean.shape == (4,)
        assert logvar.shape == (4,)
        assert trajectory.shape == (seq_len, 4)
        assert kl_div.shape == ()  # scalar KL
        assert jnp.isfinite(kl_div)
        assert kl_div >= 0  # KL should be non-negative

    def test_latent_sde_posterior_vs_prior_sampling(self):
        """Test that posterior and prior sampling produce different results."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        model = ldx.models.create_latent_sde(
            latent_dim=4,
            obs_dim=2,
            hidden_dims=(16,),
            posterior_hidden_dims=(16,),
            prior_hidden_dims=(16,),
            diffusion_hidden_dims=(8,),
            context_dim=8,
            rngs=rngs
        )

        # Create test data
        seq_len = 15
        observations = jax.random.normal(key, (seq_len, 2))
        times = jnp.linspace(0, 1, seq_len)

        # Get initial state - need to add batch dimension
        key, enc_key, post_key, prior_key = jax.random.split(key, 4)
        mean, logvar = model.encode(observations[None, ...], times)  # Add batch dim
        initial_state = model.reparameterize(mean[0], logvar[0], enc_key)  # Remove batch dim

        # Sample from posterior (conditioned on observations)
        trajectory_post = model.sample_posterior(
            initial_state, times, observations, post_key, return_kl=False
        )

        # Sample from prior (unconditional)
        trajectory_prior = model.sample_trajectory(
            initial_state, times, prior_key
        )

        # Check shapes
        assert trajectory_post.shape == (seq_len, 4)
        assert trajectory_prior.shape == (seq_len, 4)

        # Trajectories should be different (posterior is conditioned on observations)
        # Note: They start from the same point but should diverge
        assert not jnp.allclose(trajectory_post, trajectory_prior, atol=0.1)


class TestModelIntegration:
    """Integration tests for complete models."""

    def test_end_to_end_latent_ode(self):
        """Test end-to-end LatentODE pipeline."""
        key = jax.random.PRNGKey(42)

        # Generate synthetic data
        times, observations = ldx.datasets.generate_spiral_trajectory(
            n_points=50, n_dims=2, noise_std=0.1, key=key
        )

        # Normalize data
        normalized_obs, _ = ldx.utils.normalize_data(observations)

        # Create model
        key, model_key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)

        model = ldx.models.create_latent_ode(
            latent_dim=4,
            obs_dim=2,
            hidden_dims=(16,),
            dynamics_hidden_dims=(16,),
            rngs=rngs
        )

        # Test forward pass
        key, forward_key = jax.random.split(key)
        batch_obs = normalized_obs[None, ...]  # Add batch dimension
        batch_times = times[None, ...]

        reconstructed, mean, logvar, trajectory = model.forward(
            batch_obs, batch_times, forward_key
        )

        # Check shapes
        assert reconstructed.shape == batch_obs.shape
        assert mean.shape == (1, 4)
        assert logvar.shape == (1, 4)
        assert trajectory.shape == (1, len(times), 4)

        # Check that output is reasonable
        assert jnp.isfinite(reconstructed).all()
        assert jnp.isfinite(trajectory).all()

    def test_variational_training_step(self):
        """Test variational training step."""
        key = jax.random.PRNGKey(42)

        # Create simple model and data
        rngs = nnx.Rngs(key)
        model = ldx.models.create_latent_ode(
            latent_dim=3,
            obs_dim=2,
            hidden_dims=(8,),
            dynamics_hidden_dims=(8,),
            rngs=rngs
        )

        # Create trainer
        import optax
        optimizer = optax.adam(1e-3)
        trainer = ldx.inference.VariationalTrainer(
            model=model,
            optimizer=optimizer
        )

        # Create test data
        seq_len = 20
        observations = jax.random.normal(key, (seq_len, 2))
        times = jnp.linspace(0, 1, seq_len)

        key, train_key = jax.random.split(key)

        # Test training step
        params = model  # Placeholder
        updated_params, metrics = trainer.train_step(
            params, observations, times, train_key, 0
        )

        # Check that metrics are returned
        assert "total_loss" in metrics
        assert "reconstruction_loss" in metrics
        assert "kl_loss" in metrics
        assert jnp.isfinite(metrics["total_loss"])

    def test_variational_training_step_latent_sde(self):
        """Test variational training step with LatentSDE."""
        key = jax.random.PRNGKey(42)

        # Create LatentSDE model
        rngs = nnx.Rngs(key)
        model = ldx.models.create_latent_sde(
            latent_dim=3,
            obs_dim=2,
            hidden_dims=(8,),
            posterior_hidden_dims=(8,),
            prior_hidden_dims=(8,),
            diffusion_hidden_dims=(8,),
            context_dim=8,
            rngs=rngs
        )

        # Create trainer
        import optax
        optimizer = optax.adam(1e-3)
        trainer = ldx.inference.VariationalTrainer(
            model=model,
            optimizer=optimizer
        )

        # Create test data
        seq_len = 15
        observations = jax.random.normal(key, (seq_len, 2))
        times = jnp.linspace(0, 1, seq_len)

        key, train_key = jax.random.split(key)

        # Test training step
        updated_model, metrics = trainer.train_step(
            model, observations, times, train_key, 0
        )

        # Check that metrics are returned
        assert "total_loss" in metrics
        assert "reconstruction_loss" in metrics
        assert "kl_loss" in metrics
        assert jnp.isfinite(metrics["total_loss"])
        assert jnp.isfinite(metrics["kl_loss"])
        # KL from augmented dynamics should be non-negative
        assert metrics["kl_loss"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])