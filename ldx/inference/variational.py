"""Variational inference methods for latent dynamical systems."""

from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import optax


def kl_divergence_gaussian(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """Compute KL divergence between Gaussian and standard normal.

    Args:
        mean: Mean of shape (batch_size, latent_dim)
        logvar: Log variance of shape (batch_size, latent_dim)

    Returns:
        KL divergence of shape (batch_size,)
    """
    return -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar), axis=-1)


def reconstruction_loss_mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error reconstruction loss.

    Args:
        predictions: Predicted observations of shape (batch_size, seq_len, obs_dim)
        targets: Target observations of shape (batch_size, seq_len, obs_dim)

    Returns:
        MSE loss of shape (batch_size,)
    """
    return jnp.mean((predictions - targets)**2, axis=(-2, -1))


def reconstruction_loss_gaussian(
    predictions: Tuple[jnp.ndarray, jnp.ndarray],
    targets: jnp.ndarray
) -> jnp.ndarray:
    """Negative log-likelihood for Gaussian decoder.

    Args:
        predictions: Tuple of (mean, logvar) each of shape (batch_size, seq_len, obs_dim)
        targets: Target observations of shape (batch_size, seq_len, obs_dim)

    Returns:
        Negative log-likelihood of shape (batch_size,)
    """
    mean, logvar = predictions
    var = jnp.exp(logvar)

    # Negative log-likelihood
    nll = 0.5 * (jnp.log(2 * jnp.pi) + logvar + (targets - mean)**2 / var)
    return jnp.sum(nll, axis=(-2, -1))


def elbo_loss(
    model,
    params,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    key: jax.random.PRNGKey,
    beta: float = 1.0,
    reconstruction_loss_fn=reconstruction_loss_mse
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute Evidence Lower BOund (ELBO) loss.

    Args:
        model: Latent dynamical model
        params: Model parameters
        observations: Observations of shape (batch_size, seq_len, obs_dim)
        times: Times of shape (batch_size, seq_len) or (seq_len,)
        key: Random key
        beta: KL weighting factor
        reconstruction_loss_fn: Function to compute reconstruction loss

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Forward pass (in NNX, call model directly)
    reconstructed_obs, latent_mean, latent_logvar, latent_trajectory = model(
        observations, times, key
    )

    # Reconstruction loss
    recon_loss = reconstruction_loss_fn(reconstructed_obs, observations)
    recon_loss = jnp.mean(recon_loss)

    # KL divergence
    kl_loss = kl_divergence_gaussian(latent_mean, latent_logvar)
    kl_loss = jnp.mean(kl_loss)

    # Total ELBO loss
    total_loss = recon_loss + beta * kl_loss

    metrics = {
        "total_loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss,
        "beta": beta
    }

    return total_loss, metrics


class VariationalTrainer:
    """Trainer for variational latent dynamical models."""

    def __init__(
        self,
        model,
        optimizer: optax.GradientTransformation,
        reconstruction_loss_fn=reconstruction_loss_mse,
        beta_schedule=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.beta_schedule = beta_schedule or (lambda step: 1.0)

        # Initialize optimizer state
        self.opt_state = None

    def loss_fn(
        self,
        params,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey,
        step: int
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute loss for training step."""
        beta = self.beta_schedule(step)
        return elbo_loss(
            self.model,
            params,
            observations,
            times,
            key,
            beta,
            self.reconstruction_loss_fn
        )

    def train_step(
        self,
        params,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey,
        step: int
    ) -> Tuple[Any, Dict[str, jnp.ndarray]]:
        """Single training step.

        Args:
            params: Model parameters
            observations: Batch of observations
            times: Batch of times
            key: Random key
            step: Training step number

        Returns:
            Tuple of (updated_params, metrics)
        """
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True
        )(params, observations, times, key, step)

        # Initialize optimizer state if needed
        if self.opt_state is None:
            self.opt_state = self.optimizer.init(params)

        # Update parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, metrics

    def evaluate(
        self,
        params,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Dict[str, jnp.ndarray]:
        """Evaluate model on batch.

        Args:
            params: Model parameters
            observations: Batch of observations
            times: Batch of times
            key: Random key

        Returns:
            Dictionary of evaluation metrics
        """
        loss, metrics = self.loss_fn(params, observations, times, key, 0)
        return metrics


def create_beta_schedule(schedule_type: str = "constant", **kwargs) -> callable:
    """Create beta scheduling function for KL annealing.

    Args:
        schedule_type: Type of schedule ("constant", "linear", "cyclical")
        **kwargs: Additional arguments for schedule

    Returns:
        Beta scheduling function
    """
    if schedule_type == "constant":
        beta = kwargs.get("beta", 1.0)
        return lambda step: beta

    elif schedule_type == "linear":
        start_beta = kwargs.get("start_beta", 0.0)
        end_beta = kwargs.get("end_beta", 1.0)
        warmup_steps = kwargs.get("warmup_steps", 1000)

        def linear_schedule(step):
            if step < warmup_steps:
                return start_beta + (end_beta - start_beta) * step / warmup_steps
            else:
                return end_beta

        return linear_schedule

    elif schedule_type == "cyclical":
        beta_max = kwargs.get("beta_max", 1.0)
        cycle_length = kwargs.get("cycle_length", 1000)
        n_cycles = kwargs.get("n_cycles", 4)

        def cyclical_schedule(step):
            cycle_step = step % cycle_length
            if cycle_step < cycle_length // 2:
                return beta_max * 2 * cycle_step / cycle_length
            else:
                return beta_max * (2 - 2 * cycle_step / cycle_length)

        return cyclical_schedule

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")