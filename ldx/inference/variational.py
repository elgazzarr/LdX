"""Variational inference methods for latent dynamical systems."""

from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx


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
        params: Model parameters (unused, for compatibility)
        observations: Observations of shape (batch_size, seq_len, obs_dim)
        times: Times of shape (batch_size, seq_len) or (seq_len,)
        key: Random key
        beta: KL weighting factor
        reconstruction_loss_fn: Function to compute reconstruction loss

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Check if model is LatentSDE (supports augmented dynamics)
    from ldx.models.latent_sde import LatentSDE

    is_latent_sde = isinstance(model, LatentSDE)

    # Forward pass (in NNX, call model directly)
    if is_latent_sde:
        # LatentSDE returns KL from augmented dynamics
        forward_result = model(observations, times, key, return_kl=True)
        reconstructed_obs, latent_mean, latent_logvar, latent_trajectory, kl_divergence = forward_result
    else:
        # Fallback for LatentODE or other models
        forward_result = model(observations, times, key)
        if len(forward_result) == 5:
            reconstructed_obs, latent_mean, latent_logvar, latent_trajectory, kl_divergence = forward_result
        else:
            reconstructed_obs, latent_mean, latent_logvar, latent_trajectory = forward_result
            kl_divergence = None

    # Reconstruction loss
    recon_loss = reconstruction_loss_fn(reconstructed_obs, observations)
    if recon_loss.ndim > 0:
        recon_loss = jnp.mean(recon_loss)

    # KL divergence
    if kl_divergence is not None:
        # Use KL from augmented dynamics (for LatentSDE)
        if kl_divergence.ndim > 0:
            kl_loss = jnp.mean(kl_divergence)
        else:
            kl_loss = kl_divergence
    else:
        # Use analytical Gaussian KL (for LatentODE)
        kl_loss = kl_divergence_gaussian(latent_mean, latent_logvar)
        if kl_loss.ndim > 0:
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


def _nnx_optimizer_update(optimizer, model, grads) -> None:
    """Apply ``grads`` with an ``nnx.Optimizer`` across Flax versions.

    Flax >= 0.11 expects ``optimizer.update(model, grads)``; Flax 0.10 expects
    ``optimizer.update(grads)`` (the model is bound at construction).
    """
    import inspect

    if "model" in inspect.signature(optimizer.update).parameters:
        optimizer.update(model, grads)
    else:
        optimizer.update(grads)


def _ensure_batched(
    observations: jnp.ndarray, times: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return observations as (batch, seq_len, obs_dim) and times as (batch, seq_len).

    A single unbatched sequence of shape (seq_len, obs_dim) is treated as a
    batch of one, and shared 1-D times are broadcast across the batch.
    """
    if observations.ndim == 2:
        observations = observations[None, ...]
    if times.ndim == 1:
        times = jnp.broadcast_to(times, observations.shape[:2])
    return observations, times


class VariationalTrainer:
    """Trainer for variational latent dynamical models."""

    def __init__(
        self,
        model,
        optimizer,  # Can be optax or nnx.Optimizer
        reconstruction_loss_fn=reconstruction_loss_mse,
        beta_schedule=None
    ):
        from flax import nnx

        self.model = model
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.beta_schedule = beta_schedule or (lambda step: 1.0)

        # Handle both optax and NNX optimizers
        if isinstance(optimizer, nnx.Optimizer):
            self.optimizer = optimizer
            self.use_nnx = True
            self.opt_state = None
        else:
            self.optimizer = optimizer
            self.use_nnx = False
            self.opt_state = None

        # Note: JIT compilation disabled for NNX compatibility
        # self.__post_init__()

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
        model,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey,
        step: int
    ) -> Tuple[Any, Dict[str, jnp.ndarray]]:
        """JIT-compiled training step.

        Args:
            model: Model (NNX style)
            observations: Batch of observations (batch_size, seq_len, obs_dim)
            times: Batch of times (batch_size, seq_len)
            key: Random key
            step: Training step number

        Returns:
            Tuple of (updated_model, metrics)
        """
        from flax import nnx

        def loss_fn(model):
            beta = self.beta_schedule(step)

            # Always expect batched input (batch_size, seq_len, obs_dim)
            # Use vmap to process batch efficiently
            keys = jax.random.split(key, observations.shape[0])

            def single_loss(obs, times_seq, key_seq):
                return elbo_loss(
                    model,
                    model,
                    obs[None, ...],  # Add batch dim for single sequence
                    times_seq[None, ...],
                    key_seq,
                    beta,
                    self.reconstruction_loss_fn
                )

            # Vectorize across batch
            batch_losses_and_metrics = jax.vmap(single_loss, in_axes=(0, 0, 0))(
                observations, times, keys
            )

            # Average across batch
            losses, all_metrics = batch_losses_and_metrics
            avg_loss = jnp.mean(losses)

            # Average metrics
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = jnp.mean(jnp.array([m[key] for m in all_metrics]))

            return avg_loss, avg_metrics

        # Compute gradients using NNX
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Update using NNX optimizer
        if self.use_nnx:
            _nnx_optimizer_update(self.optimizer, model, grads)
        else:
            # Fallback to old optax pattern (for backward compatibility)
            model_params = nnx.state(model, nnx.Param)
            # nnx.value_and_grad already returns a State holding only the
            # Param gradients; calling nnx.state() on it again does not
            # yield a tree that matches model_params.
            grad_params = grads

            if self.opt_state is None:
                self.opt_state = self.optimizer.init(model_params)

            updates, self.opt_state = self.optimizer.update(grad_params, self.opt_state, model_params)
            updated_params = optax.apply_updates(model_params, updates)
            nnx.update(model, updated_params)

        return model, metrics

    def __post_init__(self):
        """Create JIT-compiled versions of train_step and evaluate."""
        # Create JIT-compiled train step
        def _train_step_fn(model, observations, times, key, step):
            from flax import nnx

            def loss_fn(model):
                beta = self.beta_schedule(step)

                # Always expect batched input (batch_size, seq_len, obs_dim)
                # Use vmap to process batch efficiently
                keys = jax.random.split(key, observations.shape[0])

                def single_loss(obs, times_seq, key_seq):
                    return elbo_loss(
                        model,
                        model,
                        obs[None, ...],  # Add batch dim for single sequence
                        times_seq[None, ...],
                        key_seq,
                        beta,
                        self.reconstruction_loss_fn
                    )

                # Vectorize across batch
                batch_losses_and_metrics = jax.vmap(single_loss, in_axes=(0, 0, 0))(
                    observations, times, keys
                )

                # Average across batch
                losses, all_metrics = batch_losses_and_metrics
                avg_loss = jnp.mean(losses)

                # Average metrics (all_metrics is already a dict with arrays)
                avg_metrics = {}
                for metric_key in all_metrics.keys():
                    avg_metrics[metric_key] = jnp.mean(all_metrics[metric_key])

                return avg_loss, avg_metrics

            # Compute gradients using NNX
            (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

            # Update using NNX optimizer
            if self.use_nnx:
                _nnx_optimizer_update(self.optimizer, model, grads)
            else:
                # Fallback to old optax pattern (for backward compatibility)
                model_params = nnx.state(model, nnx.Param)
                grad_params = nnx.state(grads, nnx.Param)

                if self.opt_state is None:
                    self.opt_state = self.optimizer.init(model_params)

                updates, self.opt_state = self.optimizer.update(grad_params, self.opt_state, model_params)
                updated_params = optax.apply_updates(model_params, updates)
                nnx.update(model, updated_params)

            return model, metrics

        # Create JIT-compiled evaluate function
        def _evaluate_fn(model, observations, times, key):
            from flax import nnx

            def eval_fn(model):
                beta = 1.0  # Use full beta for evaluation

                # Always expect batched input (batch_size, seq_len, obs_dim)
                # Use vmap to process batch efficiently
                keys = jax.random.split(key, observations.shape[0])

                def single_eval(obs, times_seq, key_seq):
                    return elbo_loss(
                        model,
                        model,
                        obs[None, ...],  # Add batch dim for single sequence
                        times_seq[None, ...],
                        key_seq,
                        beta,
                        self.reconstruction_loss_fn
                    )

                # Vectorize across batch
                batch_losses_and_metrics = jax.vmap(single_eval, in_axes=(0, 0, 0))(
                    observations, times, keys
                )

                # Average across batch
                losses, all_metrics = batch_losses_and_metrics
                avg_loss = jnp.mean(losses)

                # Average metrics (all_metrics is already a dict with arrays)
                avg_metrics = {}
                for metric_key in all_metrics.keys():
                    avg_metrics[metric_key] = jnp.mean(all_metrics[metric_key])

                return avg_loss, avg_metrics

            loss, metrics = eval_fn(model)
            return metrics

        # JIT compile the functions
        self._jit_train_step = jax.jit(_train_step_fn, static_argnums=(4,))  # step is static
        self._jit_evaluate = jax.jit(_evaluate_fn)

    def train_step(
        self,
        model,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey,
        step: int
    ) -> Tuple[Any, Dict[str, jnp.ndarray]]:
        """Training step with efficient vmap processing."""
        from flax import nnx

        observations, times = _ensure_batched(observations, times)

        def loss_fn(model):
            beta = self.beta_schedule(step)

            # Always expect batched input (batch_size, seq_len, obs_dim)
            # Use vmap to process batch efficiently
            keys = jax.random.split(key, observations.shape[0])

            def single_loss(obs, times_seq, key_seq):
                return elbo_loss(
                    model,
                    model,
                    obs[None, ...],  # Add batch dim for single sequence
                    times_seq[None, ...],
                    key_seq,
                    beta,
                    self.reconstruction_loss_fn
                )

            # Vectorize across batch
            batch_losses_and_metrics = jax.vmap(single_loss, in_axes=(0, 0, 0))(
                observations, times, keys
            )

            # Average across batch
            losses, all_metrics = batch_losses_and_metrics
            avg_loss = jnp.mean(losses)

            # Average metrics (all_metrics is already a dict with arrays)
            avg_metrics = {}
            for metric_key in all_metrics.keys():
                avg_metrics[metric_key] = jnp.mean(all_metrics[metric_key])

            return avg_loss, avg_metrics

        # Compute gradients using NNX
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Update using NNX optimizer
        if self.use_nnx:
            _nnx_optimizer_update(self.optimizer, model, grads)
        else:
            # Fallback to old optax pattern (for backward compatibility)
            model_params = nnx.state(model, nnx.Param)
            # nnx.value_and_grad already returns a State holding only the
            # Param gradients; calling nnx.state() on it again does not
            # yield a tree that matches model_params.
            grad_params = grads

            if self.opt_state is None:
                self.opt_state = self.optimizer.init(model_params)

            updates, self.opt_state = self.optimizer.update(grad_params, self.opt_state, model_params)
            updated_params = optax.apply_updates(model_params, updates)
            nnx.update(model, updated_params)

        return model, metrics

    def evaluate(
        self,
        model,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Dict[str, jnp.ndarray]:
        """Evaluation with efficient vmap processing."""
        from flax import nnx

        observations, times = _ensure_batched(observations, times)

        def eval_fn(model):
            beta = 1.0  # Use full beta for evaluation

            # Always expect batched input (batch_size, seq_len, obs_dim)
            # Use vmap to process batch efficiently
            keys = jax.random.split(key, observations.shape[0])

            def single_eval(obs, times_seq, key_seq):
                return elbo_loss(
                    model,
                    model,
                    obs[None, ...],  # Add batch dim for single sequence
                    times_seq[None, ...],
                    key_seq,
                    beta,
                    self.reconstruction_loss_fn
                )

            # Vectorize across batch
            batch_losses_and_metrics = jax.vmap(single_eval, in_axes=(0, 0, 0))(
                observations, times, keys
            )

            # Average across batch
            losses, all_metrics = batch_losses_and_metrics
            avg_loss = jnp.mean(losses)

            # Average metrics (all_metrics is already a dict with arrays)
            avg_metrics = {}
            for metric_key in all_metrics.keys():
                avg_metrics[metric_key] = jnp.mean(all_metrics[metric_key])

            return avg_loss, avg_metrics

        loss, metrics = eval_fn(model)
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