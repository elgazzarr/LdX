"""Latent SDE models for stochastic dynamics."""

from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
import diffrax

from .base import LatentModel, DynamicsFunction
from .encoders import MLPEncoder, RNNEncoder
from .decoders import MLPDecoder, GaussianDecoder


class DriftDiffusionSDE(nnx.Module):
    """SDE with separate drift and diffusion functions."""

    def __init__(
        self,
        drift_func: DynamicsFunction,
        diffusion_func: DynamicsFunction,
        rngs: nnx.Rngs = None
    ):
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, args) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return drift and diffusion terms.

        Args:
            t: Time
            y: State
            args: Additional arguments

        Returns:
            Tuple of (drift, diffusion)
        """
        drift = self.drift_func(y, t)
        diffusion = self.diffusion_func(y, t)
        return drift, diffusion


class LatentSDE(LatentModel):
    """Latent SDE model for learning stochastic dynamics in latent space."""

    def __init__(
        self,
        drift_func: DynamicsFunction,
        diffusion_func: DynamicsFunction,
        encoder: nnx.Module,
        decoder: nnx.Module,
        latent_dim: int,
        solver: diffrax.AbstractSolver = diffrax.EulerHeun(),
        dt0: float = 0.01,
        rngs: nnx.Rngs = None
    ):
        super().__init__(latent_dim, rngs)
        self.sde_func = DriftDiffusionSDE(drift_func, diffusion_func, rngs)
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
        """Sample trajectory by solving SDE from initial state."""
        def drift_diffusion(t, y, args):
            return self.sde_func(t, y, args)

        # Set up SDE terms
        drift_term = diffrax.ODETerm(lambda t, y, args: drift_diffusion(t, y, args)[0])
        diffusion_term = diffrax.WeaklyDiagonalControlTerm(
            lambda t, y, args: drift_diffusion(t, y, args)[1],
            diffrax.VirtualBrownianTree(
                t0=times[0],
                t1=times[-1],
                tol=1e-3,
                shape=(self.latent_dim,),
                key=key
            )
        )

        # Solve SDE
        solution = diffrax.diffeqsolve(
            diffrax.MultiTerm(drift_term, diffusion_term),
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

        # Vectorize over batch dimension with different keys
        keys = jax.random.split(key, initial_state.shape[0])
        trajectory_fn = jax.vmap(
            lambda z0, k: self.sample_trajectory(z0, times_flat, k)
        )
        latent_trajectory = trajectory_fn(initial_state, keys)

        # Decode trajectory
        reconstructed_obs = self.decode(latent_trajectory, times)

        return reconstructed_obs, latent_mean, latent_logvar, latent_trajectory

    def __call__(self, observations: jnp.ndarray, times: jnp.ndarray, key: jax.random.PRNGKey):
        """Call method for NNX compatibility."""
        return self.forward(observations, times, key)


class DiagonalDiffusion(DynamicsFunction):
    """Diagonal diffusion function for SDE."""

    def __init__(self, hidden_dims: Tuple[int, ...], latent_dim: int, rngs: nnx.Rngs = None):
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Initialize layers - input size is latent_dim + 1 (for time)
        input_dim = latent_dim + 1
        self.layers = []

        for hidden_dim in hidden_dims:
            self.layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = nnx.Linear(input_dim, latent_dim, rngs=rngs)

    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute diagonal diffusion coefficients.

        Args:
            z: Latent state of shape (batch_size, latent_dim)
            t: Time of shape (batch_size,) or scalar

        Returns:
            Diagonal diffusion matrix of shape (batch_size, latent_dim)
        """
        # Concatenate state and time
        t = jnp.asarray(t)  # Ensure t is a JAX array
        if t.ndim == 0:  # scalar time
            t = jnp.broadcast_to(t, z.shape[:-1] + (1,))
        elif t.ndim == 1:  # batch of times
            t = t[..., None]

        x = jnp.concatenate([z, t], axis=-1)

        for layer in self.layers:
            x = layer(x)
            x = jax.nn.tanh(x)

        # Output layer
        x = self.output_layer(x)

        # Apply softplus to ensure positive diffusion
        x = jax.nn.softplus(x)
        return x


def create_latent_sde(
    latent_dim: int,
    obs_dim: int,
    hidden_dims: Tuple[int, ...] = (64, 64),
    dynamics_hidden_dims: Tuple[int, ...] = (64, 64),
    diffusion_hidden_dims: Tuple[int, ...] = (32, 32),
    encoder_type: str = "mlp",
    decoder_type: str = "mlp",
    rngs: nnx.Rngs = None
) -> LatentSDE:
    """Factory function to create a LatentSDE model.

    Args:
        latent_dim: Dimension of latent space
        obs_dim: Dimension of observations
        hidden_dims: Hidden dimensions for encoder/decoder
        dynamics_hidden_dims: Hidden dimensions for drift function
        diffusion_hidden_dims: Hidden dimensions for diffusion function
        encoder_type: Type of encoder ("mlp" or "rnn")
        decoder_type: Type of decoder ("mlp" or "gaussian")
        rngs: Random number generators

    Returns:
        Configured LatentSDE model
    """
    from .base import MLPDynamics

    # Create drift and diffusion functions
    drift_func = MLPDynamics(dynamics_hidden_dims, latent_dim=latent_dim, rngs=rngs)
    diffusion_func = DiagonalDiffusion(diffusion_hidden_dims, latent_dim=latent_dim, rngs=rngs)

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

    return LatentSDE(drift_func, diffusion_func, encoder, decoder, latent_dim, rngs=rngs)