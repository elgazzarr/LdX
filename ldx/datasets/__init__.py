"""Benchmark datasets and data generators."""

from .synthetic import (
    generate_spiral_trajectory,
    generate_lorenz_trajectory,
    generate_pendulum_trajectory,
    generate_oscillator_trajectory,
    generate_van_der_pol_trajectory,
    generate_controlled_oscillator_trajectory,
    generate_irregular_observations,
    generate_partially_observed_system,
    SyntheticDataset
)

__all__ = [
    "generate_spiral_trajectory",
    "generate_lorenz_trajectory",
    "generate_pendulum_trajectory",
    "generate_oscillator_trajectory",
    "generate_van_der_pol_trajectory",
    "generate_controlled_oscillator_trajectory",
    "generate_irregular_observations",
    "generate_partially_observed_system",
    "SyntheticDataset",
]