"""Smoothing algorithms for latent dynamical systems."""

from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp

from .filtering import FilterState, ExtendedKalmanFilter


class SmoothingResult(NamedTuple):
    """Result of smoothing algorithm."""
    means: jnp.ndarray      # Shape (seq_len, latent_dim)
    covariances: jnp.ndarray  # Shape (seq_len, latent_dim, latent_dim)
    times: jnp.ndarray      # Shape (seq_len,)


class RauchTungStriebel:
    """Rauch-Tung-Striebel smoother for nonlinear systems."""

    def __init__(self, ekf: ExtendedKalmanFilter):
        self.ekf = ekf

    def smooth_sequence(
        self,
        filtered_means: jnp.ndarray,
        filtered_covs: jnp.ndarray,
        times: jnp.ndarray
    ) -> SmoothingResult:
        """Apply RTS smoothing to filtered estimates.

        Args:
            filtered_means: Filtered means of shape (seq_len, latent_dim)
            filtered_covs: Filtered covariances of shape (seq_len, latent_dim, latent_dim)
            times: Times of shape (seq_len,)

        Returns:
            Smoothing result
        """
        seq_len, latent_dim = filtered_means.shape

        # Initialize storage
        smoothed_means = jnp.zeros_like(filtered_means)
        smoothed_covs = jnp.zeros_like(filtered_covs)

        # Initialize with final filtered estimates
        smoothed_means = smoothed_means.at[-1].set(filtered_means[-1])
        smoothed_covs = smoothed_covs.at[-1].set(filtered_covs[-1])

        # Backward pass
        for k in range(seq_len - 2, -1, -1):
            dt = times[k + 1] - times[k]

            # Predict next state
            filter_state = FilterState(
                mean=filtered_means[k],
                cov=filtered_covs[k],
                time=times[k]
            )
            predicted_state = self.ekf.predict(filter_state, dt)

            # Compute smoother gain
            jacobian_f = jax.jacfwd(
                lambda x: self.ekf.dynamics_func(x, times[k])
            )(filtered_means[k])
            F = jnp.eye(latent_dim) + dt * jacobian_f

            # Avoid division by zero
            predicted_cov_inv = jnp.linalg.pinv(predicted_state.cov)
            A = filtered_covs[k] @ F.T @ predicted_cov_inv

            # Smooth
            smoothed_means = smoothed_means.at[k].set(
                filtered_means[k] + A @ (smoothed_means[k + 1] - predicted_state.mean)
            )
            smoothed_covs = smoothed_covs.at[k].set(
                filtered_covs[k] + A @ (smoothed_covs[k + 1] - predicted_state.cov) @ A.T
            )

        return SmoothingResult(
            means=smoothed_means,
            covariances=smoothed_covs,
            times=times
        )


class TwoFilterSmoother:
    """Two-filter smoother combining forward and backward filters."""

    def __init__(
        self,
        forward_ekf: ExtendedKalmanFilter,
        backward_ekf: ExtendedKalmanFilter
    ):
        self.forward_ekf = forward_ekf
        self.backward_ekf = backward_ekf

    def smooth_sequence(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        initial_state: FilterState,
        final_state: FilterState
    ) -> SmoothingResult:
        """Apply two-filter smoothing.

        Args:
            observations: Observations of shape (seq_len, obs_dim)
            times: Times of shape (seq_len,)
            initial_state: Initial state for forward filter
            final_state: Final state for backward filter

        Returns:
            Smoothing result
        """
        # Forward filtering
        forward_means, forward_covs = self.forward_ekf.filter_sequence(
            initial_state, observations, times
        )

        # Backward filtering (reverse time and observations)
        backward_times = times[::-1]
        backward_observations = observations[::-1]
        backward_means, backward_covs = self.backward_ekf.filter_sequence(
            final_state, backward_observations, backward_times
        )

        # Reverse backward results
        backward_means = backward_means[::-1]
        backward_covs = backward_covs[::-1]

        # Combine forward and backward estimates
        seq_len, latent_dim = forward_means.shape
        smoothed_means = jnp.zeros_like(forward_means)
        smoothed_covs = jnp.zeros_like(forward_covs)

        for k in range(seq_len):
            # Information form combination
            forward_info = jnp.linalg.inv(forward_covs[k])
            backward_info = jnp.linalg.inv(backward_covs[k])

            # Combined information matrix and vector
            combined_info = forward_info + backward_info
            combined_info_vec = (
                forward_info @ forward_means[k] +
                backward_info @ backward_means[k]
            )

            # Convert back to moment form
            smoothed_cov = jnp.linalg.inv(combined_info)
            smoothed_mean = smoothed_cov @ combined_info_vec

            smoothed_means = smoothed_means.at[k].set(smoothed_mean)
            smoothed_covs = smoothed_covs.at[k].set(smoothed_cov)

        return SmoothingResult(
            means=smoothed_means,
            covariances=smoothed_covs,
            times=times
        )


class ParticleSmoother:
    """Particle smoother using forward-backward sampling."""

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

    def backward_sampling(
        self,
        forward_particles: jnp.ndarray,
        forward_weights: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Perform backward sampling to get smoothed trajectories.

        Args:
            forward_particles: Forward particles of shape (seq_len, n_particles, latent_dim)
            forward_weights: Forward weights of shape (seq_len, n_particles)
            times: Times of shape (seq_len,)
            key: Random key

        Returns:
            Smoothed trajectories of shape (n_trajectories, seq_len, latent_dim)
        """
        seq_len, n_particles, latent_dim = forward_particles.shape
        n_trajectories = n_particles

        # Storage for trajectories
        trajectories = jnp.zeros((n_trajectories, seq_len, latent_dim))

        # Sample final particles
        keys = jax.random.split(key, seq_len)
        final_indices = jax.random.choice(
            keys[0], n_particles, shape=(n_trajectories,), p=forward_weights[-1]
        )
        trajectories = trajectories.at[:, -1].set(forward_particles[-1, final_indices])

        # Backward pass
        for k in range(seq_len - 2, -1, -1):
            dt = times[k + 1] - times[k]

            # For each trajectory, sample from previous particles
            def sample_previous(current_state, key_k):
                # Compute transition probabilities
                def transition_prob(prev_state):
                    predicted_state = prev_state + dt * self.dynamics_func(prev_state, times[k])
                    return jnp.exp(-0.5 * jnp.sum(
                        (current_state - predicted_state)**2 / (dt * self.process_noise_std**2)
                    ))

                transition_probs = jax.vmap(transition_prob)(forward_particles[k])
                # Combine with forward weights
                combined_weights = forward_weights[k] * transition_probs
                combined_weights = combined_weights / jnp.sum(combined_weights)

                # Sample previous particle
                prev_idx = jax.random.choice(key_k, n_particles, p=combined_weights)
                return forward_particles[k, prev_idx]

            # Sample for all trajectories
            prev_states = jax.vmap(sample_previous)(
                trajectories[:, k + 1], jax.random.split(keys[k + 1], n_trajectories)
            )
            trajectories = trajectories.at[:, k].set(prev_states)

        return trajectories

    def smooth_sequence(
        self,
        forward_particles: jnp.ndarray,
        forward_weights: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey,
        n_smooth_particles: int = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate smoothed particle trajectories.

        Args:
            forward_particles: Forward particles of shape (seq_len, n_particles, latent_dim)
            forward_weights: Forward weights of shape (seq_len, n_particles)
            times: Times of shape (seq_len,)
            key: Random key
            n_smooth_particles: Number of smoothed particles to generate

        Returns:
            Tuple of (smoothed_means, smoothed_trajectories)
        """
        if n_smooth_particles is None:
            n_smooth_particles = self.n_particles

        # Generate smoothed trajectories
        smoothed_trajectories = self.backward_sampling(
            forward_particles, forward_weights, times, key
        )

        # Compute smoothed means
        smoothed_means = jnp.mean(smoothed_trajectories, axis=0)

        return smoothed_means, smoothed_trajectories[:n_smooth_particles]