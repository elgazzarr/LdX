"""Inference methods for latent dynamical systems.

Currently supports:
- Variational inference with ELBO loss for latent ODE/SDE models
- Filtering (extended Kalman filter, particle filter)
- Smoothing (Rauch-Tung-Striebel, two-filter, particle smoother)
"""

from .variational import (
    kl_divergence_gaussian,
    reconstruction_loss_mse,
    reconstruction_loss_gaussian,
    elbo_loss,
    VariationalTrainer,
    create_beta_schedule
)
from .filtering import (
    FilterState,
    ExtendedKalmanFilter,
    ParticleFilter
)
from .smoothing import (
    SmoothingResult,
    RauchTungStriebel,
    TwoFilterSmoother,
    ParticleSmoother
)

__all__ = [
    "kl_divergence_gaussian",
    "reconstruction_loss_mse",
    "reconstruction_loss_gaussian",
    "elbo_loss",
    "VariationalTrainer",
    "create_beta_schedule",
    "FilterState",
    "ExtendedKalmanFilter",
    "ParticleFilter",
    "SmoothingResult",
    "RauchTungStriebel",
    "TwoFilterSmoother",
    "ParticleSmoother",
]
