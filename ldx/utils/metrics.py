"""Evaluation metrics for latent dynamical systems."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


def mse(predictions: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
    """Mean squared error.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional mask for valid observations

    Returns:
        MSE value
    """
    squared_errors = (predictions - targets) ** 2

    if mask is not None:
        squared_errors = jnp.where(mask[..., None], squared_errors, 0.0)
        return jnp.sum(squared_errors) / jnp.sum(mask)
    else:
        return jnp.mean(squared_errors)


def mae(predictions: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
    """Mean absolute error.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional mask for valid observations

    Returns:
        MAE value
    """
    absolute_errors = jnp.abs(predictions - targets)

    if mask is not None:
        absolute_errors = jnp.where(mask[..., None], absolute_errors, 0.0)
        return jnp.sum(absolute_errors) / jnp.sum(mask)
    else:
        return jnp.mean(absolute_errors)


def rmse(predictions: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
    """Root mean squared error.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional mask for valid observations

    Returns:
        RMSE value
    """
    return jnp.sqrt(mse(predictions, targets, mask))


def mape(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray = None,
    eps: float = 1e-8
) -> jnp.ndarray:
    """Mean absolute percentage error.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional mask for valid observations
        eps: Small constant to avoid division by zero

    Returns:
        MAPE value
    """
    percentage_errors = jnp.abs((predictions - targets) / (jnp.abs(targets) + eps))

    if mask is not None:
        percentage_errors = jnp.where(mask[..., None], percentage_errors, 0.0)
        return jnp.sum(percentage_errors) / jnp.sum(mask)
    else:
        return jnp.mean(percentage_errors)


def r2_score(predictions: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
    """R-squared score.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional mask for valid observations

    Returns:
        R² value
    """
    if mask is not None:
        # Compute means only over valid observations
        target_mean = jnp.sum(targets * mask[..., None]) / jnp.sum(mask)
        ss_res = jnp.sum(jnp.where(mask[..., None], (targets - predictions) ** 2, 0.0))
        ss_tot = jnp.sum(jnp.where(mask[..., None], (targets - target_mean) ** 2, 0.0))
    else:
        target_mean = jnp.mean(targets)
        ss_res = jnp.sum((targets - predictions) ** 2)
        ss_tot = jnp.sum((targets - target_mean) ** 2)

    return 1 - ss_res / (ss_tot + 1e-8)


def pearson_correlation(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Pearson correlation coefficient.

    Args:
        predictions: Predicted values
        targets: Target values

    Returns:
        Correlation coefficient
    """
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Compute correlation
    pred_centered = pred_flat - jnp.mean(pred_flat)
    target_centered = target_flat - jnp.mean(target_flat)

    numerator = jnp.sum(pred_centered * target_centered)
    denominator = jnp.sqrt(jnp.sum(pred_centered ** 2) * jnp.sum(target_centered ** 2))

    return numerator / (denominator + 1e-8)


def log_likelihood_gaussian(
    predictions: Tuple[jnp.ndarray, jnp.ndarray],
    targets: jnp.ndarray,
    mask: jnp.ndarray = None
) -> jnp.ndarray:
    """Log-likelihood for Gaussian predictions.

    Args:
        predictions: Tuple of (mean, logvar)
        targets: Target values
        mask: Optional mask for valid observations

    Returns:
        Log-likelihood value
    """
    mean, logvar = predictions
    var = jnp.exp(logvar)

    # Gaussian log-likelihood
    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + logvar + (targets - mean) ** 2 / var)

    if mask is not None:
        log_prob = jnp.where(mask[..., None], log_prob, 0.0)
        return jnp.sum(log_prob) / jnp.sum(mask)
    else:
        return jnp.mean(log_prob)


def prediction_interval_coverage(
    predictions: Tuple[jnp.ndarray, jnp.ndarray],
    targets: jnp.ndarray,
    confidence_level: float = 0.95,
    mask: jnp.ndarray = None
) -> jnp.ndarray:
    """Prediction interval coverage probability.

    Args:
        predictions: Tuple of (mean, logvar)
        targets: Target values
        confidence_level: Confidence level for intervals
        mask: Optional mask for valid observations

    Returns:
        Coverage probability
    """
    mean, logvar = predictions
    std = jnp.exp(0.5 * logvar)

    # Compute prediction intervals
    z_score = jax.scipy.stats.norm.ppf(0.5 + confidence_level / 2)
    lower = mean - z_score * std
    upper = mean + z_score * std

    # Check coverage
    in_interval = (targets >= lower) & (targets <= upper)

    if mask is not None:
        in_interval = jnp.where(mask[..., None], in_interval, False)
        return jnp.sum(in_interval) / jnp.sum(mask)
    else:
        return jnp.mean(in_interval)


def trajectory_mse(
    predicted_trajectories: jnp.ndarray,
    true_trajectories: jnp.ndarray,
    times: jnp.ndarray = None
) -> jnp.ndarray:
    """MSE for trajectory predictions.

    Args:
        predicted_trajectories: Shape (batch_size, seq_len, latent_dim)
        true_trajectories: Shape (batch_size, seq_len, latent_dim)
        times: Optional times for weighted MSE

    Returns:
        Trajectory MSE
    """
    if times is None:
        return jnp.mean((predicted_trajectories - true_trajectories) ** 2)
    else:
        # Weight by time intervals
        dt = jnp.diff(times, axis=-1, prepend=0.0)
        weights = dt / jnp.sum(dt, axis=-1, keepdims=True)
        weighted_errors = weights[..., None] * (predicted_trajectories - true_trajectories) ** 2
        return jnp.mean(jnp.sum(weighted_errors, axis=1))


def compute_all_metrics(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray = None,
    probabilistic: bool = False
) -> Dict[str, float]:
    """Compute all regression metrics.

    Args:
        predictions: Predicted values (or tuple for probabilistic)
        targets: Target values
        mask: Optional mask for valid observations
        probabilistic: Whether predictions are probabilistic

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if probabilistic:
        mean_pred, logvar_pred = predictions
        metrics.update({
            "mse": float(mse(mean_pred, targets, mask)),
            "mae": float(mae(mean_pred, targets, mask)),
            "rmse": float(rmse(mean_pred, targets, mask)),
            "r2": float(r2_score(mean_pred, targets, mask)),
            "log_likelihood": float(log_likelihood_gaussian(predictions, targets, mask)),
            "coverage_95": float(prediction_interval_coverage(predictions, targets, 0.95, mask)),
            "coverage_90": float(prediction_interval_coverage(predictions, targets, 0.90, mask)),
        })
    else:
        metrics.update({
            "mse": float(mse(predictions, targets, mask)),
            "mae": float(mae(predictions, targets, mask)),
            "rmse": float(rmse(predictions, targets, mask)),
            "mape": float(mape(predictions, targets, mask)),
            "r2": float(r2_score(predictions, targets, mask)),
            "correlation": float(pearson_correlation(predictions, targets)),
        })

    return metrics


def forecast_metrics(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    horizon_steps: int,
    mask: jnp.ndarray = None
) -> Dict[str, jnp.ndarray]:
    """Compute metrics as a function of forecast horizon.

    Args:
        predictions: Predicted values of shape (batch_size, seq_len, obs_dim)
        targets: Target values of shape (batch_size, seq_len, obs_dim)
        horizon_steps: Number of steps to compute metrics for
        mask: Optional mask for valid observations

    Returns:
        Dictionary of metrics per horizon step
    """
    metrics = {
        "mse_by_step": [],
        "mae_by_step": [],
        "rmse_by_step": []
    }

    for step in range(min(horizon_steps, predictions.shape[1])):
        step_mask = mask[:, step] if mask is not None else None

        step_mse = mse(predictions[:, step], targets[:, step], step_mask)
        step_mae = mae(predictions[:, step], targets[:, step], step_mask)
        step_rmse = rmse(predictions[:, step], targets[:, step], step_mask)

        metrics["mse_by_step"].append(step_mse)
        metrics["mae_by_step"].append(step_mae)
        metrics["rmse_by_step"].append(step_rmse)

    # Convert to arrays
    for key in metrics:
        metrics[key] = jnp.array(metrics[key])

    return metrics