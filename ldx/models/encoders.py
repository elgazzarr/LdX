"""Encoder modules for mapping observations to latent representations."""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx


class MLPEncoder(nnx.Module):
    """MLP-based encoder for observations."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...],
        latent_dim: int,
        obs_dim: int,
        rngs: nnx.Rngs
    ):
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Build hidden layers
        self.hidden_layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        # Output layers for mean and log variance
        self.mean_layer = nnx.Linear(input_dim, latent_dim, rngs=rngs)
        self.logvar_layer = nnx.Linear(input_dim, latent_dim, rngs=rngs)

    def __call__(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode observations to latent distribution parameters.

        Args:
            observations: Shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)

        Returns:
            Tuple of (mean, logvar) each of shape (batch_size, latent_dim)
        """
        # If we have a sequence, take the last observation or aggregate somehow
        if observations.ndim == 3:
            x = observations[:, -1]  # Take last observation
        else:
            x = observations

        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.tanh(x)

        # Output mean and log variance
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)

        return mean, logvar


class RNNEncoder(nnx.Module):
    """RNN-based encoder for sequential observations."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        obs_dim: int,
        rngs: nnx.Rngs
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # GRU cell for processing sequences
        self.gru_cell = nnx.GRUCell(
            in_features=obs_dim,
            hidden_features=hidden_dim,
            keep_rngs=True,  # Keep rngs for initialization
            rngs=rngs
        )

        # Output layers
        self.mean_layer = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)
        self.logvar_layer = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)

    def __call__(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode sequential observations to latent distribution parameters.

        Args:
            observations: Shape (batch_size, seq_len, obs_dim)

        Returns:
            Tuple of (mean, logvar) each of shape (batch_size, latent_dim)
        """
        batch_size, seq_len, obs_dim = observations.shape

        # Initialize carry state with zeros
        carry = jnp.zeros((batch_size, self.hidden_dim))

        # Process sequence step by step using scan
        def scan_fn(carry, x):
            carry, output = self.gru_cell(carry, x)
            return carry, output

        # Transpose to (seq_len, batch_size, obs_dim) for scanning
        observations_transposed = jnp.transpose(observations, (1, 0, 2))

        final_carry, all_outputs = jax.lax.scan(scan_fn, carry, observations_transposed)

        # Output mean and log variance using final hidden state
        mean = self.mean_layer(final_carry)
        logvar = self.logvar_layer(final_carry)

        return mean, logvar


class CNNEncoder(nnx.Module):
    """CNN-based encoder for image observations."""

    def __init__(
        self,
        features: Tuple[int, ...],
        latent_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 3, 3),
        strides: Tuple[int, ...] = (2, 2, 2),
        rngs: nnx.Rngs = None
    ):
        self.features = features
        self.latent_dim = latent_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Build convolutional layers
        self.conv_layers = []
        for i, feature in enumerate(features):
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 3
            stride = strides[i] if i < len(strides) else 1
            self.conv_layers.append(
                nnx.Conv(feature, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), rngs=rngs)
            )

        # Output layers (will be initialized after first call to determine input size)
        self.mean_layer = None
        self.logvar_layer = None

    def __call__(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode image observations to latent distribution parameters.

        Args:
            observations: Shape (batch_size, height, width, channels) or
                         (batch_size, seq_len, height, width, channels)

        Returns:
            Tuple of (mean, logvar) each of shape (batch_size, latent_dim)
        """
        # Handle sequential images by taking the last frame
        if observations.ndim == 5:
            x = observations[:, -1]  # Take last image in sequence
        else:
            x = observations

        # Forward through conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = jax.nn.relu(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Initialize output layers if not done yet
        if self.mean_layer is None:
            self.mean_layer = nnx.Linear(x.shape[-1], self.latent_dim)
            self.logvar_layer = nnx.Linear(x.shape[-1], self.latent_dim)

        # Output mean and log variance
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)

        return mean, logvar