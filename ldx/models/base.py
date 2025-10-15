"""Base classes for latent dynamical models."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


class DynamicsFunction(nnx.Module, ABC):
    """Abstract base class for dynamics functions f(z, t, θ)."""

    @abstractmethod
    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute dynamics f(z, t).

        Args:
            z: Latent state of shape (batch_size, latent_dim)
            t: Time of shape (batch_size,) or scalar

        Returns:
            Dynamics of shape (batch_size, latent_dim)
        """
        pass


class LatentModel(nnx.Module, ABC):
    """Abstract base class for latent dynamical models."""

    def __init__(self, latent_dim: int, rngs: nnx.Rngs):
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, observations: jnp.ndarray, times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode observations to latent space.

        Args:
            observations: Observations of shape (batch_size, seq_len, obs_dim)
            times: Times of shape (batch_size, seq_len)

        Returns:
            Tuple of (latent_mean, latent_logvar) each of shape (batch_size, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, latent_states: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
        """Decode latent states to observation space.

        Args:
            latent_states: Latent states of shape (batch_size, seq_len, latent_dim)
            times: Times of shape (batch_size, seq_len)

        Returns:
            Decoded observations of shape (batch_size, seq_len, obs_dim)
        """
        pass

    @abstractmethod
    def sample_trajectory(
        self,
        initial_state: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Sample trajectory from initial state.

        Args:
            initial_state: Initial latent state of shape (batch_size, latent_dim)
            times: Times to sample at of shape (seq_len,)
            key: Random key

        Returns:
            Trajectory of shape (batch_size, seq_len, latent_dim)
        """
        pass


class MLPDynamics(DynamicsFunction):
    """MLP-based dynamics function."""

    def __init__(self, hidden_dims: Tuple[int, ...], latent_dim: int, activation: Callable = jax.nn.tanh, rngs: nnx.Rngs = None):
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.activation = activation

        # Initialize layers - input size is latent_dim + 1 (for time)
        input_dim = latent_dim + 1
        self.layers = []

        for hidden_dim in hidden_dims:
            self.layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = nnx.Linear(input_dim, latent_dim, rngs=rngs)

    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # Concatenate state and time
        t = jnp.asarray(t)  # Ensure t is a JAX array
        if t.ndim == 0:  # scalar time
            t = jnp.broadcast_to(t, z.shape[:-1] + (1,))
        elif t.ndim == 1:  # batch of times
            t = t[..., None]

        x = jnp.concatenate([z, t], axis=-1)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        # Output layer
        x = self.output_layer(x)
        x = nnx.tanh(x)  # Ensure bounded output
        return x