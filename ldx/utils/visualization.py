"""Visualization utilities for latent dynamical systems."""

from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import jax.numpy as jnp
import numpy as np


def plot_trajectory(
    times: np.ndarray,
    observations: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    dims: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Trajectory"
) -> plt.Figure:
    """Plot trajectory with optional predictions and uncertainty.

    Args:
        times: Time points of shape (seq_len,)
        observations: Observations of shape (seq_len, obs_dim)
        predictions: Optional predictions of shape (seq_len, obs_dim)
        uncertainty: Optional uncertainty (std) of shape (seq_len, obs_dim)
        mask: Optional mask for valid observations
        dims: Which dimensions to plot (if None, plot all)
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if dims is None:
        dims = list(range(observations.shape[1]))

    n_dims = len(dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, dim in enumerate(dims):
        ax = axes[i]

        # Plot observations
        obs_times = times
        obs_values = observations[:, dim]

        if mask is not None:
            obs_times = times[mask]
            obs_values = observations[mask, dim]

        ax.scatter(obs_times, obs_values, alpha=0.7, label='Observations', s=20)

        # Plot predictions
        if predictions is not None:
            pred_values = predictions[:, dim]
            ax.plot(times, pred_values, label='Predictions', linewidth=2)

            # Plot uncertainty bands
            if uncertainty is not None:
                lower = pred_values - 1.96 * uncertainty[:, dim]
                upper = pred_values + 1.96 * uncertainty[:, dim]
                ax.fill_between(times, lower, upper, alpha=0.3, label='95% CI')

        ax.set_xlabel('Time')
        ax.set_ylabel(f'Dimension {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_latent_trajectory(
    times: np.ndarray,
    latent_states: np.ndarray,
    dims: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Latent Trajectory"
) -> plt.Figure:
    """Plot latent space trajectory.

    Args:
        times: Time points of shape (seq_len,)
        latent_states: Latent states of shape (seq_len, latent_dim)
        dims: Which latent dimensions to plot
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if dims is None:
        dims = list(range(min(latent_states.shape[1], 4)))  # Plot first 4 dims by default

    n_dims = len(dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, dim in enumerate(dims):
        ax = axes[i]
        ax.plot(times, latent_states[:, dim], linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Latent Dim {dim}')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_phase_portrait(
    latent_states: np.ndarray,
    dims: Tuple[int, int] = (0, 1),
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Phase Portrait"
) -> plt.Figure:
    """Plot 2D phase portrait of latent dynamics.

    Args:
        latent_states: Latent states of shape (seq_len, latent_dim)
        dims: Which two dimensions to plot
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = latent_states[:, dims[0]]
    y = latent_states[:, dims[1]]

    # Create colormap based on time
    colors = np.linspace(0, 1, len(x))
    scatter = ax.scatter(x, y, c=colors, cmap='viridis', alpha=0.7)

    # Add trajectory line
    ax.plot(x, y, alpha=0.3, linewidth=1, color='gray')

    # Mark start and end
    ax.scatter(x[0], y[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], color='red', s=100, marker='s', label='End')

    ax.set_xlabel(f'Latent Dim {dims[0]}')
    ax.set_ylabel(f'Latent Dim {dims[1]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Time')
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_reconstruction_error(
    times: np.ndarray,
    observations: np.ndarray,
    reconstructions: np.ndarray,
    mask: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot reconstruction error over time.

    Args:
        times: Time points
        observations: True observations
        reconstructions: Reconstructed observations
        mask: Optional mask for valid observations
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Compute reconstruction error
    errors = np.abs(observations - reconstructions)
    if mask is not None:
        errors = errors[mask]
        times_masked = times[mask]
    else:
        times_masked = times

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot mean error over time
    if errors.ndim > 1:
        mean_error = np.mean(errors, axis=1)
    else:
        mean_error = errors

    ax1.plot(times_masked, mean_error, linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Reconstruction Error')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Reconstruction Error Over Time')

    # Plot error distribution
    ax2.hist(mean_error, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Error Distribution')

    plt.tight_layout()
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    metrics: Optional[dict] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot training curves.

    Args:
        train_losses: Training losses
        val_losses: Optional validation losses
        metrics: Optional dictionary of additional metrics
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_plots = 1 + (len(metrics) if metrics else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Plot losses
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Training Loss', linewidth=2)

    if val_losses is not None:
        ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Training Curves')

    # Plot additional metrics
    if metrics:
        for i, (metric_name, values) in enumerate(metrics.items()):
            if i + 1 < len(axes):
                ax = axes[i + 1]
                ax.plot(epochs, values, linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')

    plt.tight_layout()
    return fig


def plot_uncertainty_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot uncertainty calibration.

    Args:
        predictions: Predicted values
        targets: True values
        uncertainties: Prediction uncertainties (standard deviations)
        n_bins: Number of bins for calibration
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Compute absolute errors
    errors = np.abs(predictions - targets)

    # Calibration plot
    # Bin by uncertainty
    uncertainty_bins = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
    bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2

    mean_uncertainties = []
    mean_errors = []

    for i in range(n_bins):
        mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i + 1])
        if np.sum(mask) > 0:
            mean_uncertainties.append(np.mean(uncertainties[mask]))
            mean_errors.append(np.mean(errors[mask]))

    ax1.scatter(mean_uncertainties, mean_errors, alpha=0.7, s=50)
    ax1.plot([0, max(mean_uncertainties)], [0, max(mean_uncertainties)], 'r--', label='Perfect Calibration')
    ax1.set_xlabel('Predicted Uncertainty')
    ax1.set_ylabel('Observed Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Uncertainty Calibration')

    # Error vs uncertainty scatter
    ax2.scatter(uncertainties, errors, alpha=0.5, s=10)
    ax2.set_xlabel('Predicted Uncertainty')
    ax2.set_ylabel('Absolute Error')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Error vs Uncertainty')

    plt.tight_layout()
    return fig


def plot_forecast_horizon(
    forecast_metrics: dict,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot metrics as a function of forecast horizon.

    Args:
        forecast_metrics: Dictionary with metrics by horizon step
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_metrics = len(forecast_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (metric_name, values) in enumerate(forecast_metrics.items()):
        ax = axes[i]
        horizon_steps = range(1, len(values) + 1)
        ax.plot(horizon_steps, values, marker='o', linewidth=2)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel(metric_name.upper())
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{metric_name.upper()} vs Horizon')

    plt.tight_layout()
    return fig


def animate_trajectory(
    times: np.ndarray,
    observations: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    interval: int = 100
):
    """Create animated trajectory plot.

    Args:
        times: Time points
        observations: True observations
        predictions: Optional predictions
        save_path: Path to save animation
        interval: Animation interval in milliseconds
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib.animation not available, skipping animation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    def animate(frame):
        ax.clear()
        end_idx = min(frame + 1, len(times))

        # Plot observations up to current frame
        ax.scatter(times[:end_idx], observations[:end_idx, 0], alpha=0.7, label='Observations')

        # Plot predictions if available
        if predictions is not None:
            ax.plot(times[:end_idx], predictions[:end_idx, 0], label='Predictions', linewidth=2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Trajectory (t = {times[end_idx-1]:.2f})')

    anim = FuncAnimation(fig, animate, frames=len(times), interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow')

    return anim