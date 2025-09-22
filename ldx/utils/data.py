"""Data handling utilities for latent dynamical systems."""

from typing import Tuple, Iterator, Optional, List
import functools

import jax
import jax.numpy as jnp
import numpy as np


class TimeSeriesDataset:
    """Dataset for time series data with irregular sampling."""

    def __init__(
        self,
        observations: np.ndarray,
        times: np.ndarray,
        batch_size: int = 32,
        sequence_length: Optional[int] = None,
        shuffle: bool = True
    ):
        """Initialize time series dataset.

        Args:
            observations: Observations of shape (n_sequences, max_seq_len, obs_dim)
            times: Times of shape (n_sequences, max_seq_len)
            batch_size: Batch size for iteration
            sequence_length: Fixed sequence length (if None, use variable lengths)
            shuffle: Whether to shuffle data
        """
        self.observations = observations
        self.times = times
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.n_sequences = observations.shape[0]

        # Create indices for iteration
        self.indices = np.arange(self.n_sequences)
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return (self.n_sequences + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Iterate over batches."""
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, self.n_sequences, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_obs = self.observations[batch_indices]
            batch_times = self.times[batch_indices]

            # Convert to JAX arrays
            batch_obs = jnp.array(batch_obs)
            batch_times = jnp.array(batch_times)

            # Handle sequence length
            if self.sequence_length is not None:
                batch_obs = batch_obs[:, :self.sequence_length]
                batch_times = batch_times[:, :self.sequence_length]

            yield batch_obs, batch_times


def create_sliding_windows(
    observations: jnp.ndarray,
    times: jnp.ndarray,
    window_size: int,
    stride: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create sliding windows from time series data.

    Args:
        observations: Observations of shape (seq_len, obs_dim)
        times: Times of shape (seq_len,)
        window_size: Size of each window
        stride: Stride between windows

    Returns:
        Tuple of (windowed_obs, windowed_times)
    """
    seq_len = observations.shape[0]
    n_windows = (seq_len - window_size) // stride + 1

    windowed_obs = []
    windowed_times = []

    for i in range(0, seq_len - window_size + 1, stride):
        windowed_obs.append(observations[i:i + window_size])
        windowed_times.append(times[i:i + window_size])

    return jnp.stack(windowed_obs), jnp.stack(windowed_times)


def pad_sequences(
    sequences: List[jnp.ndarray],
    times: List[jnp.ndarray],
    pad_value: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pad sequences to same length.

    Args:
        sequences: List of sequences with different lengths
        times: List of time arrays with different lengths
        pad_value: Value to use for padding

    Returns:
        Tuple of (padded_sequences, padded_times, mask)
    """
    max_len = max(seq.shape[0] for seq in sequences)
    n_sequences = len(sequences)
    obs_dim = sequences[0].shape[1]

    # Initialize padded arrays
    padded_sequences = jnp.full((n_sequences, max_len, obs_dim), pad_value)
    padded_times = jnp.zeros((n_sequences, max_len))
    mask = jnp.zeros((n_sequences, max_len), dtype=bool)

    for i, (seq, time) in enumerate(zip(sequences, times)):
        seq_len = seq.shape[0]
        padded_sequences = padded_sequences.at[i, :seq_len].set(seq)
        padded_times = padded_times.at[i, :seq_len].set(time)
        mask = mask.at[i, :seq_len].set(True)

    return padded_sequences, padded_times, mask


def normalize_data(
    data: jnp.ndarray,
    method: str = "standard",
    axis: int = 0,
    eps: float = 1e-8
) -> Tuple[jnp.ndarray, dict]:
    """Normalize data.

    Args:
        data: Data to normalize
        method: Normalization method ("standard", "minmax", "robust")
        axis: Axis along which to normalize
        eps: Small constant for numerical stability

    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == "standard":
        mean = jnp.mean(data, axis=axis, keepdims=True)
        std = jnp.std(data, axis=axis, keepdims=True) + eps
        normalized = (data - mean) / std
        params = {"mean": mean, "std": std, "method": "standard"}

    elif method == "minmax":
        min_val = jnp.min(data, axis=axis, keepdims=True)
        max_val = jnp.max(data, axis=axis, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + eps)
        params = {"min": min_val, "max": max_val, "method": "minmax"}

    elif method == "robust":
        median = jnp.median(data, axis=axis, keepdims=True)
        mad = jnp.median(jnp.abs(data - median), axis=axis, keepdims=True) + eps
        normalized = (data - median) / mad
        params = {"median": median, "mad": mad, "method": "robust"}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def denormalize_data(data: jnp.ndarray, params: dict) -> jnp.ndarray:
    """Denormalize data using stored parameters.

    Args:
        data: Normalized data
        params: Normalization parameters from normalize_data

    Returns:
        Denormalized data
    """
    method = params["method"]

    if method == "standard":
        return data * params["std"] + params["mean"]
    elif method == "minmax":
        return data * (params["max"] - params["min"]) + params["min"]
    elif method == "robust":
        return data * params["mad"] + params["median"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def interpolate_missing(
    observations: jnp.ndarray,
    times: jnp.ndarray,
    mask: jnp.ndarray,
    method: str = "linear"
) -> jnp.ndarray:
    """Interpolate missing observations.

    Args:
        observations: Observations with missing values
        times: Times
        mask: Boolean mask indicating valid observations
        method: Interpolation method ("linear", "nearest")

    Returns:
        Interpolated observations
    """
    if method == "linear":
        # Simple linear interpolation
        interpolated = observations.copy()

        for i in range(observations.shape[0]):  # For each sequence
            for j in range(observations.shape[2]):  # For each dimension
                obs_seq = observations[i, :, j]
                time_seq = times[i, :]
                mask_seq = mask[i, :]

                # Find valid points
                valid_indices = jnp.where(mask_seq)[0]

                if len(valid_indices) > 1:
                    # Interpolate between valid points
                    interpolated = interpolated.at[i, :, j].set(
                        jnp.interp(time_seq, time_seq[valid_indices], obs_seq[valid_indices])
                    )

        return interpolated

    elif method == "nearest":
        # Forward fill then backward fill
        interpolated = observations.copy()

        # Forward fill
        for i in range(observations.shape[1] - 1):
            next_obs = jnp.where(mask[:, i + 1:i + 2], observations[:, i + 1:i + 2], interpolated[:, i:i + 1])
            interpolated = interpolated.at[:, i + 1:i + 2].set(next_obs)

        # Backward fill
        for i in range(observations.shape[1] - 1, 0, -1):
            prev_obs = jnp.where(mask[:, i - 1:i], observations[:, i - 1:i], interpolated[:, i:i + 1])
            interpolated = interpolated.at[:, i - 1:i].set(prev_obs)

        return interpolated

    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def create_train_val_split(
    observations: jnp.ndarray,
    times: jnp.ndarray,
    val_fraction: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Split data into training and validation sets.

    Args:
        observations: Observations array
        times: Times array
        val_fraction: Fraction of data to use for validation
        shuffle: Whether to shuffle before splitting
        random_state: Random seed

    Returns:
        Tuple of ((train_obs, train_times), (val_obs, val_times))
    """
    n_sequences = observations.shape[0]
    n_val = int(n_sequences * val_fraction)

    # Create indices
    indices = np.arange(n_sequences)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    # Split indices
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Split data
    train_obs = observations[train_indices]
    train_times = times[train_indices]
    val_obs = observations[val_indices]
    val_times = times[val_indices]

    return (train_obs, train_times), (val_obs, val_times)