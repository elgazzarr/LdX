"""Inference methods for latent dynamical systems.

Currently supports:
- Variational inference with ELBO loss for latent ODE/SDE models
"""

from .variational import (
    kl_divergence_gaussian,
    reconstruction_loss_mse,
    reconstruction_loss_gaussian,
    elbo_loss,
    VariationalTrainer,
    create_beta_schedule
)

__all__ = [
    "kl_divergence_gaussian",
    "reconstruction_loss_mse",
    "reconstruction_loss_gaussian",
    "elbo_loss",
    "VariationalTrainer",
    "create_beta_schedule",
]