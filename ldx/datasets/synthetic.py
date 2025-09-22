"""Synthetic datasets for testing latent dynamical systems."""

from typing import Tuple, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np


def generate_spiral_trajectory(
    n_points: int = 100,
    n_dims: int = 2,
    noise_std: float = 0.1,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate spiral trajectory in 2D or 3D.

    Args:
        n_points: Number of time points
        n_dims: Number of dimensions (2 or 3)
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    times = jnp.linspace(0, 4 * jnp.pi, n_points)
    t_scaled = times / (4 * jnp.pi)

    if n_dims == 2:
        # 2D spiral
        x = t_scaled * jnp.cos(times)
        y = t_scaled * jnp.sin(times)
        observations = jnp.stack([x, y], axis=1)
    elif n_dims == 3:
        # 3D spiral
        x = t_scaled * jnp.cos(times)
        y = t_scaled * jnp.sin(times)
        z = t_scaled
        observations = jnp.stack([x, y, z], axis=1)
    else:
        raise ValueError("n_dims must be 2 or 3")

    # Add noise
    noise = jax.random.normal(key, observations.shape) * noise_std
    observations = observations + noise

    return times, observations


def generate_lorenz_trajectory(
    n_points: int = 1000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0,
    initial_state: Optional[jnp.ndarray] = None,
    noise_std: float = 0.1,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Lorenz attractor trajectory.

    Args:
        n_points: Number of time points
        dt: Time step
        sigma, rho, beta: Lorenz system parameters
        initial_state: Initial state (if None, use default)
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    if initial_state is None:
        initial_state = jnp.array([1.0, 1.0, 1.0])

    def lorenz_dynamics(state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return jnp.array([dx, dy, dz])

    # Integrate using Euler method
    states = [initial_state]
    current_state = initial_state

    for _ in range(n_points - 1):
        current_state = current_state + dt * lorenz_dynamics(current_state)
        states.append(current_state)

    observations = jnp.stack(states)
    times = jnp.arange(n_points) * dt

    # Add noise
    noise = jax.random.normal(key, observations.shape) * noise_std
    observations = observations + noise

    return times, observations


def generate_pendulum_trajectory(
    n_points: int = 200,
    dt: float = 0.05,
    length: float = 1.0,
    gravity: float = 9.81,
    damping: float = 0.1,
    initial_angle: float = jnp.pi/4,
    initial_velocity: float = 0.0,
    noise_std: float = 0.05,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate damped pendulum trajectory.

    Args:
        n_points: Number of time points
        dt: Time step
        length: Pendulum length
        gravity: Gravitational acceleration
        damping: Damping coefficient
        initial_angle: Initial angle
        initial_velocity: Initial angular velocity
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations) where observations are [x, y] positions
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    def pendulum_dynamics(state):
        angle, velocity = state
        acceleration = -(gravity / length) * jnp.sin(angle) - damping * velocity
        return jnp.array([velocity, acceleration])

    # Integrate using Euler method
    states = [jnp.array([initial_angle, initial_velocity])]
    current_state = jnp.array([initial_angle, initial_velocity])

    for _ in range(n_points - 1):
        current_state = current_state + dt * pendulum_dynamics(current_state)
        states.append(current_state)

    angles = jnp.stack([state[0] for state in states])
    times = jnp.arange(n_points) * dt

    # Convert to Cartesian coordinates
    x = length * jnp.sin(angles)
    y = -length * jnp.cos(angles)
    observations = jnp.stack([x, y], axis=1)

    # Add noise
    noise = jax.random.normal(key, observations.shape) * noise_std
    observations = observations + noise

    return times, observations


def generate_oscillator_trajectory(
    n_points: int = 200,
    dt: float = 0.1,
    frequency: float = 1.0,
    damping: float = 0.05,
    amplitude: float = 1.0,
    phase: float = 0.0,
    noise_std: float = 0.1,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate damped harmonic oscillator trajectory.

    Args:
        n_points: Number of time points
        dt: Time step
        frequency: Oscillation frequency
        damping: Damping coefficient
        amplitude: Initial amplitude
        phase: Phase offset
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations) where observations are [position, velocity]
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    times = jnp.arange(n_points) * dt
    omega = 2 * jnp.pi * frequency

    # Analytical solution for damped harmonic oscillator
    exp_decay = jnp.exp(-damping * times)
    position = amplitude * exp_decay * jnp.cos(omega * times + phase)
    velocity = amplitude * exp_decay * (-damping * jnp.cos(omega * times + phase) - omega * jnp.sin(omega * times + phase))

    observations = jnp.stack([position, velocity], axis=1)

    # Add noise
    noise = jax.random.normal(key, observations.shape) * noise_std
    observations = observations + noise

    return times, observations


def generate_van_der_pol_trajectory(
    n_points: int = 500,
    dt: float = 0.02,
    mu: float = 1.0,
    initial_state: Optional[jnp.ndarray] = None,
    noise_std: float = 0.05,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Van der Pol oscillator trajectory.

    Args:
        n_points: Number of time points
        dt: Time step
        mu: Van der Pol parameter
        initial_state: Initial state [x, dx/dt]
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    if initial_state is None:
        initial_state = jnp.array([2.0, 0.0])

    def van_der_pol_dynamics(state):
        x, y = state
        dx = y
        dy = mu * (1 - x**2) * y - x
        return jnp.array([dx, dy])

    # Integrate using Euler method
    states = [initial_state]
    current_state = initial_state

    for _ in range(n_points - 1):
        current_state = current_state + dt * van_der_pol_dynamics(current_state)
        states.append(current_state)

    observations = jnp.stack(states)
    times = jnp.arange(n_points) * dt

    # Add noise
    noise = jax.random.normal(key, observations.shape) * noise_std
    observations = observations + noise

    return times, observations


def generate_irregular_observations(
    times: jnp.ndarray,
    observations: jnp.ndarray,
    observation_prob: float = 0.7,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create irregular observations by randomly masking some time points.

    Args:
        times: Full time array
        observations: Full observations
        observation_prob: Probability of observing at each time point
        key: Random key

    Returns:
        Tuple of (observed_times, observed_data, mask)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # Create observation mask
    mask = jax.random.bernoulli(key, observation_prob, shape=(len(times),))

    # Extract observed data
    observed_times = times[mask]
    observed_data = observations[mask]

    return observed_times, observed_data, mask


def generate_partially_observed_system(
    true_dim: int,
    obs_dim: int,
    n_points: int = 200,
    dynamics_type: str = "linear",
    noise_std: float = 0.1,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate partially observed dynamical system.

    Args:
        true_dim: Dimension of true latent system
        obs_dim: Dimension of observations
        n_points: Number of time points
        dynamics_type: Type of dynamics ("linear", "nonlinear")
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, true_states, observations, observation_matrix)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    keys = jax.random.split(key, 4)

    times = jnp.linspace(0, 10, n_points)
    dt = times[1] - times[0]

    # Generate dynamics matrix
    if dynamics_type == "linear":
        # Stable linear dynamics
        A = jax.random.normal(keys[0], (true_dim, true_dim)) * 0.1
        A = A - jnp.eye(true_dim) * 0.05  # Make stable
    else:
        raise NotImplementedError("Only linear dynamics implemented")

    # Generate observation matrix
    C = jax.random.normal(keys[1], (obs_dim, true_dim))

    # Generate initial state
    initial_state = jax.random.normal(keys[2], (true_dim,))

    # Integrate dynamics
    states = [initial_state]
    current_state = initial_state

    for _ in range(n_points - 1):
        if dynamics_type == "linear":
            current_state = current_state + dt * (A @ current_state)
        states.append(current_state)

    true_states = jnp.stack(states)

    # Generate observations
    observations = (C @ true_states.T).T

    # Add noise
    noise = jax.random.normal(keys[3], observations.shape) * noise_std
    observations = observations + noise

    return times, true_states, observations, C


class SyntheticDataset:
    """Collection of synthetic datasets for testing."""

    @staticmethod
    def spiral(n_points: int = 100, **kwargs):
        """Generate spiral dataset."""
        return generate_spiral_trajectory(n_points=n_points, **kwargs)

    @staticmethod
    def lorenz(n_points: int = 1000, **kwargs):
        """Generate Lorenz attractor dataset."""
        return generate_lorenz_trajectory(n_points=n_points, **kwargs)

    @staticmethod
    def pendulum(n_points: int = 200, **kwargs):
        """Generate pendulum dataset."""
        return generate_pendulum_trajectory(n_points=n_points, **kwargs)

    @staticmethod
    def oscillator(n_points: int = 200, **kwargs):
        """Generate harmonic oscillator dataset."""
        return generate_oscillator_trajectory(n_points=n_points, **kwargs)

    @staticmethod
    def van_der_pol(n_points: int = 500, **kwargs):
        """Generate Van der Pol oscillator dataset."""
        return generate_van_der_pol_trajectory(n_points=n_points, **kwargs)

    @staticmethod
    def partially_observed(true_dim: int = 5, obs_dim: int = 3, **kwargs):
        """Generate partially observed system dataset."""
        return generate_partially_observed_system(true_dim=true_dim, obs_dim=obs_dim, **kwargs)