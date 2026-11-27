"""Decoder modules for mapping latent states to observations."""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from ._compat import ModuleList


class MLPDecoder(nnx.Module):
    """MLP-based decoder for observations."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...],
        obs_dim: int,
        latent_dim: int,
        rngs: nnx.Rngs
    ):
        self.hidden_dims = hidden_dims
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # Build hidden layers
        self.hidden_layers = ModuleList()
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = nnx.Linear(input_dim, obs_dim, rngs=rngs)

    def __call__(self, latent_states: jnp.ndarray) -> jnp.ndarray:
        """Decode latent states to observations.

        Args:
            latent_states: Shape (batch_size, latent_dim) or (batch_size, seq_len, latent_dim)

        Returns:
            Decoded observations of shape (batch_size, obs_dim) or (batch_size, seq_len, obs_dim)
        """
        # Handle both single states and sequences
        original_shape = latent_states.shape
        if latent_states.ndim == 3:
            # Reshape to (batch_size * seq_len, latent_dim)
            batch_size, seq_len, latent_dim = latent_states.shape
            x = latent_states.reshape(-1, latent_dim)
        else:
            x = latent_states

        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.tanh(x)

        # Output layer
        x = self.output_layer(x)

        # Reshape back if needed
        if len(original_shape) == 3:
            x = x.reshape(batch_size, seq_len, -1)

        return x


class CNNDecoder(nnx.Module):
    """CNN-based decoder for image observations."""

    def __init__(
        self,
        features: Tuple[int, ...],
        output_shape: Tuple[int, int, int],  # (height, width, channels)
        latent_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 3, 3),
        strides: Tuple[int, ...] = (2, 2, 2),
        rngs: nnx.Rngs = None
    ):
        self.features = features
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Calculate the size after flattening the initial conv transpose layers
        # This is a rough estimate - may need adjustment based on actual conv transpose shapes
        self.initial_height = output_shape[0] // (2 ** len(features))
        self.initial_width = output_shape[1] // (2 ** len(features))
        self.initial_features = features[0]
        self.initial_size = self.initial_height * self.initial_width * self.initial_features

        # Dense layer to map from latent to initial conv shape
        self.dense_layer = nnx.Linear(latent_dim, self.initial_size, rngs=rngs)

        # Build transpose convolutional layers
        self.conv_transpose_layers = ModuleList()
        for i, feature in enumerate(features[1:] + (output_shape[2],)):
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 3
            stride = strides[i] if i < len(strides) else 1
            self.conv_transpose_layers.append(
                nnx.ConvTranspose(
                    feature,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(stride, stride),
                    rngs=rngs
                )
            )

    def __call__(self, latent_states: jnp.ndarray) -> jnp.ndarray:
        """Decode latent states to image observations.

        Args:
            latent_states: Shape (batch_size, latent_dim) or (batch_size, seq_len, latent_dim)

        Returns:
            Decoded images of shape (batch_size, height, width, channels) or
                                   (batch_size, seq_len, height, width, channels)
        """
        # Handle both single states and sequences
        original_shape = latent_states.shape
        if latent_states.ndim == 3:
            # Reshape to (batch_size * seq_len, latent_dim)
            batch_size, seq_len, latent_dim = latent_states.shape
            x = latent_states.reshape(-1, latent_dim)
        else:
            x = latent_states
            batch_size = x.shape[0]

        # Map to initial conv shape
        x = self.dense_layer(x)
        x = nnx.relu(x)

        # Reshape to conv format
        x = x.reshape(-1, self.initial_height, self.initial_width, self.initial_features)

        # Forward through transpose conv layers
        for i, conv_layer in enumerate(self.conv_transpose_layers):
            x = conv_layer(x)
            # Apply activation except for the last layer
            if i < len(self.conv_transpose_layers) - 1:
                x = jax.nn.relu(x)
            else:
                x = jax.nn.sigmoid(x)  # Assuming pixel values in [0, 1]

        # Reshape back if needed
        if len(original_shape) == 3:
            x = x.reshape(batch_size, seq_len, *self.output_shape)

        return x


class GaussianDecoder(nnx.Module):
    """Decoder that outputs Gaussian distribution parameters."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...],
        obs_dim: int,
        latent_dim: int,
        rngs: nnx.Rngs
    ):
        self.hidden_dims = hidden_dims
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # Build hidden layers
        self.hidden_layers = ModuleList()
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        # Output layers for mean and log variance
        self.mean_layer = nnx.Linear(input_dim, obs_dim, rngs=rngs)
        self.logvar_layer = nnx.Linear(input_dim, obs_dim, rngs=rngs)

    def __call__(self, latent_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decode latent states to observation distribution parameters.

        Args:
            latent_states: Shape (batch_size, latent_dim) or (batch_size, seq_len, latent_dim)

        Returns:
            Tuple of (mean, logvar) for observations
        """
        # Handle both single states and sequences
        original_shape = latent_states.shape
        if latent_states.ndim == 3:
            # Reshape to (batch_size * seq_len, latent_dim)
            batch_size, seq_len, latent_dim = latent_states.shape
            x = latent_states.reshape(-1, latent_dim)
        else:
            x = latent_states

        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.tanh(x)

        # Output mean and log variance
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)

        # Reshape back if needed
        if len(original_shape) == 3:
            mean = mean.reshape(batch_size, seq_len, -1)
            logvar = logvar.reshape(batch_size, seq_len, -1)

        return mean, logvar