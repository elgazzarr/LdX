"""Latent ODE models for continuous-time dynamics."""

from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
import diffrax

from .base import LatentModel, DynamicsFunction
from .encoders import MLPEncoder, RNNEncoder
from .decoders import MLPDecoder, GaussianDecoder


class LatentODE(LatentModel):
    """Latent ODE model for learning dynamics in latent space."""

    def __init__(
        self,
        dynamics_func: DynamicsFunction,
        encoder: nnx.Module,
        decoder: nnx.Module,
        latent_dim: int,
        solver: diffrax.AbstractSolver = diffrax.Dopri5(),
        dt0: float = 0.1,
        rngs: nnx.Rngs = None
    ):
        super().__init__(latent_dim, rngs)
        self.dynamics_func = dynamics_func
        self.encoder = encoder
        self.decoder = decoder
        self.solver = solver
        self.dt0 = dt0

    def encode(self, observations: jnp.ndarray, times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode observations to latent distribution parameters."""
        return self.encoder(observations)

    def decode(self, latent_states: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
        """Decode latent states to observations."""
        return self.decoder(latent_states)

    def sample_trajectory(
        self,
        initial_state: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Sample trajectory by solving ODE from initial state."""
        def ode_func(t, y, args):
            return self.dynamics_func(y, t)

        # Set up ODE term
        ode_term = diffrax.ODETerm(ode_func)

        # Solve ODE
        solution = diffrax.diffeqsolve(
            ode_term,
            self.solver,
            t0=times[0],
            t1=times[-1],
            dt0=self.dt0,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=times)
        )

        return solution.ys

    def reparameterize(self, mean: jnp.ndarray, logvar: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Reparameterization trick for sampling from latent distribution."""
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mean.shape)
        return mean + std * eps

    def forward(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass through the model.

        Args:
            observations: Shape (batch_size, seq_len, obs_dim)
            times: Shape (batch_size, seq_len) or (seq_len,)
            key: Random key

        Returns:
            Tuple of (reconstructed_obs, latent_mean, latent_logvar, latent_trajectory)
        """
        # Encode observations
        latent_mean, latent_logvar = self.encode(observations, times)

        # Sample initial latent state
        key, subkey = jax.random.split(key)
        initial_state = self.reparameterize(latent_mean, latent_logvar, subkey)

        # Generate trajectory
        if times.ndim == 2:
            times_flat = times[0]  # Assume same times for all batch elements
        else:
            times_flat = times

        # Vectorize over batch dimension
        trajectory_fn = jax.vmap(
            lambda z0: self.sample_trajectory(z0, times_flat, key)
        )
        latent_trajectory = trajectory_fn(initial_state)

        # Decode trajectory
        reconstructed_obs = self.decode(latent_trajectory, times)

        return reconstructed_obs, latent_mean, latent_logvar, latent_trajectory

    def __call__(self, observations: jnp.ndarray, times: jnp.ndarray, key: jax.random.PRNGKey):
        """Call method for NNX compatibility."""
        return self.forward(observations, times, key)


def create_latent_ode(
    latent_dim: int,
    obs_dim: int,
    hidden_dims: Tuple[int, ...] = (64, 64),
    dynamics_hidden_dims: Tuple[int, ...] = (64, 64),
    encoder_type: str = "mlp",
    decoder_type: str = "mlp",
    rngs: nnx.Rngs = None
) -> LatentODE:
    """Factory function to create a LatentODE model.

    Args:
        latent_dim: Dimension of latent space
        obs_dim: Dimension of observations
        hidden_dims: Hidden dimensions for encoder/decoder
        dynamics_hidden_dims: Hidden dimensions for dynamics function
        encoder_type: Type of encoder ("mlp" or "rnn")
        decoder_type: Type of decoder ("mlp" or "gaussian")
        rngs: Random number generators

    Returns:
        Configured LatentODE model
    """
    from .base import MLPDynamics

    # Create dynamics function
    dynamics_func = MLPDynamics(dynamics_hidden_dims, latent_dim=latent_dim, rngs=rngs)

    # Create encoder
    if encoder_type == "mlp":
        encoder = MLPEncoder(hidden_dims, latent_dim, obs_dim, rngs)
    elif encoder_type == "rnn":
        encoder = RNNEncoder(hidden_dims[0], latent_dim, obs_dim, rngs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Create decoder
    if decoder_type == "mlp":
        decoder = MLPDecoder(hidden_dims, obs_dim, latent_dim, rngs)
    elif decoder_type == "gaussian":
        decoder = GaussianDecoder(hidden_dims, obs_dim, latent_dim, rngs)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    return LatentODE(dynamics_func, encoder, decoder, latent_dim, rngs=rngs)