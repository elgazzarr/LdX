"""Filtering methods for state estimation in latent dynamical systems."""

from typing import Tuple, NamedTuple, Optional

import jax
import jax.numpy as jnp


class FilterState(NamedTuple):
    """State of a filter at a given time."""
    mean: jnp.ndarray  # Shape (latent_dim,)
    cov: jnp.ndarray   # Shape (latent_dim, latent_dim)
    time: float


class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear latent dynamics."""

    def __init__(
        self,
        dynamics_func,
        observation_func,
        process_noise_cov: jnp.ndarray,
        observation_noise_cov: jnp.ndarray
    ):
        self.dynamics_func = dynamics_func
        self.observation_func = observation_func
        self.Q = process_noise_cov  # Process noise covariance
        self.R = observation_noise_cov  # Observation noise covariance

    def predict(
        self,
        state: FilterState,
        dt: float
    ) -> FilterState:
        """Prediction step of EKF.

        Args:
            state: Current filter state
            dt: Time step

        Returns:
            Predicted state
        """
        # Predict mean using dynamics
        next_time = state.time + dt
        predicted_mean = state.mean + dt * self.dynamics_func(state.mean, state.time)

        # Compute Jacobian of dynamics function
        jacobian_f = jax.jacfwd(
            lambda x: self.dynamics_func(x, state.time)
        )(state.mean)

        # Predict covariance
        F = jnp.eye(state.mean.shape[0]) + dt * jacobian_f
        predicted_cov = F @ state.cov @ F.T + dt * self.Q

        return FilterState(
            mean=predicted_mean,
            cov=predicted_cov,
            time=next_time
        )

    def update(
        self,
        predicted_state: FilterState,
        observation: jnp.ndarray
    ) -> FilterState:
        """Update step of EKF.

        Args:
            predicted_state: Predicted state from prediction step
            observation: Observed data

        Returns:
            Updated state
        """
        # Predicted observation
        predicted_obs = self.observation_func(predicted_state.mean, predicted_state.time)

        # Compute Jacobian of observation function
        jacobian_h = jax.jacfwd(
            lambda x: self.observation_func(x, predicted_state.time)
        )(predicted_state.mean)

        # Innovation covariance
        S = jacobian_h @ predicted_state.cov @ jacobian_h.T + self.R

        # Kalman gain
        K = predicted_state.cov @ jacobian_h.T @ jnp.linalg.inv(S)

        # Innovation
        innovation = observation - predicted_obs

        # Update mean and covariance
        updated_mean = predicted_state.mean + K @ innovation
        updated_cov = (jnp.eye(predicted_state.mean.shape[0]) - K @ jacobian_h) @ predicted_state.cov

        return FilterState(
            mean=updated_mean,
            cov=updated_cov,
            time=predicted_state.time
        )

    def filter_sequence(
        self,
        initial_state: FilterState,
        observations: jnp.ndarray,
        times: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Filter a sequence of observations.

        Args:
            initial_state: Initial filter state
            observations: Observations of shape (seq_len, obs_dim)
            times: Times of shape (seq_len,)

        Returns:
            Tuple of (filtered_means, filtered_covs)
        """
        seq_len = observations.shape[0]
        latent_dim = initial_state.mean.shape[0]

        # Storage for results
        filtered_means = jnp.zeros((seq_len, latent_dim))
        filtered_covs = jnp.zeros((seq_len, latent_dim, latent_dim))

        current_state = initial_state

        for i in range(seq_len):
            # Prediction step
            if i > 0:
                dt = times[i] - times[i-1]
                current_state = self.predict(current_state, dt)

            # Update step
            current_state = self.update(current_state, observations[i])

            # Store results
            filtered_means = filtered_means.at[i].set(current_state.mean)
            filtered_covs = filtered_covs.at[i].set(current_state.cov)

        return filtered_means, filtered_covs


class ParticleFilter:
    """Particle filter for nonlinear state estimation."""

    def __init__(
        self,
        dynamics_func,
        observation_func,
        process_noise_std: float,
        observation_noise_std: float,
        n_particles: int = 100
    ):
        self.dynamics_func = dynamics_func
        self.observation_func = observation_func
        self.process_noise_std = process_noise_std
        self.observation_noise_std = observation_noise_std
        self.n_particles = n_particles

    def predict_particles(
        self,
        particles: jnp.ndarray,
        time: float,
        dt: float,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Predict particles forward in time.

        Args:
            particles: Current particles of shape (n_particles, latent_dim)
            time: Current time
            dt: Time step
            key: Random key

        Returns:
            Predicted particles
        """
        # Apply dynamics to each particle
        dynamics_vmap = jax.vmap(lambda x: self.dynamics_func(x, time))
        predicted_particles = particles + dt * dynamics_vmap(particles)

        # Add process noise
        noise = jax.random.normal(key, particles.shape) * self.process_noise_std
        predicted_particles = predicted_particles + jnp.sqrt(dt) * noise

        return predicted_particles

    def compute_weights(
        self,
        particles: jnp.ndarray,
        observation: jnp.ndarray,
        time: float
    ) -> jnp.ndarray:
        """Compute particle weights based on observation likelihood.

        Args:
            particles: Particles of shape (n_particles, latent_dim)
            observation: Observed data of shape (obs_dim,)
            time: Current time

        Returns:
            Weights of shape (n_particles,)
        """
        # Compute predicted observations for each particle
        obs_vmap = jax.vmap(lambda x: self.observation_func(x, time))
        predicted_obs = obs_vmap(particles)

        # Compute likelihood (assuming Gaussian observation noise)
        log_weights = -0.5 * jnp.sum(
            (predicted_obs - observation)**2 / self.observation_noise_std**2,
            axis=-1
        )

        # Normalize weights
        log_weights = log_weights - jnp.max(log_weights)  # For numerical stability
        weights = jnp.exp(log_weights)
        weights = weights / jnp.sum(weights)

        return weights

    def resample(
        self,
        particles: jnp.ndarray,
        weights: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Resample particles based on weights.

        Args:
            particles: Particles of shape (n_particles, latent_dim)
            weights: Weights of shape (n_particles,)
            key: Random key

        Returns:
            Resampled particles
        """
        # Systematic resampling
        indices = jax.random.choice(
            key, self.n_particles, shape=(self.n_particles,), p=weights
        )
        return particles[indices]

    def filter_sequence(
        self,
        initial_particles: jnp.ndarray,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Filter a sequence of observations using particle filter.

        Args:
            initial_particles: Initial particles of shape (n_particles, latent_dim)
            observations: Observations of shape (seq_len, obs_dim)
            times: Times of shape (seq_len,)
            key: Random key

        Returns:
            Tuple of (filtered_means, particle_history)
        """
        seq_len = observations.shape[0]
        latent_dim = initial_particles.shape[1]

        # Storage for results
        filtered_means = jnp.zeros((seq_len, latent_dim))
        particle_history = jnp.zeros((seq_len, self.n_particles, latent_dim))

        particles = initial_particles
        keys = jax.random.split(key, seq_len * 2)

        for i in range(seq_len):
            # Prediction step
            if i > 0:
                dt = times[i] - times[i-1]
                particles = self.predict_particles(
                    particles, times[i-1], dt, keys[2*i]
                )

            # Update step
            weights = self.compute_weights(particles, observations[i], times[i])

            # Resample
            particles = self.resample(particles, weights, keys[2*i + 1])

            # Store results
            filtered_mean = jnp.average(particles, weights=weights, axis=0)
            filtered_means = filtered_means.at[i].set(filtered_mean)
            particle_history = particle_history.at[i].set(particles)

        return filtered_means, particle_history