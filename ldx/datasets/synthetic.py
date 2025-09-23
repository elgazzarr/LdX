"""Synthetic datasets for testing latent dynamical systems."""

from typing import Tuple, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np


def generate_spiral_trajectory(
    n_points: int = 100,
    n_sequences: int = 1,
    n_dims: int = 2,
    noise_std: float = 0.1,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate spiral trajectory in 2D or 3D.

    Args:
        n_points: Number of time points
        n_sequences: Number of sequences to generate
        n_dims: Number of dimensions (2 or 3)
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations)
        - times: Shape (n_sequences, n_points) or (n_points,) if n_sequences=1
        - observations: Shape (n_sequences, n_points, n_dims) or (n_points, n_dims) if n_sequences=1
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    def generate_single_spiral(phase_offset, radius_scale, noise_key):
        """Generate a single spiral with given parameters."""
        times = jnp.linspace(0, 4 * jnp.pi, n_points)
        t_scaled = (times + phase_offset) / (4 * jnp.pi) * radius_scale

        if n_dims == 2:
            # 2D spiral
            x = t_scaled * jnp.cos(times + phase_offset)
            y = t_scaled * jnp.sin(times + phase_offset)
            observations = jnp.stack([x, y], axis=1)
        elif n_dims == 3:
            # 3D spiral
            x = t_scaled * jnp.cos(times + phase_offset)
            y = t_scaled * jnp.sin(times + phase_offset)
            z = t_scaled
            observations = jnp.stack([x, y, z], axis=1)
        else:
            raise ValueError("n_dims must be 2 or 3")

        # Add noise
        noise = jax.random.normal(noise_key, observations.shape) * noise_std
        return observations + noise

    # Generate different parameters for each sequence
    key, param_key, noise_key = jax.random.split(key, 3)

    # Random phase offsets and radius scales for variety
    phase_offsets = jax.random.uniform(param_key, (n_sequences,), minval=0, maxval=jnp.pi)
    radius_scales = jax.random.uniform(param_key, (n_sequences,), minval=0.8, maxval=1.2)
    noise_keys = jax.random.split(noise_key, n_sequences)

    # Always use vmap for consistent batch processing
    batch_generate = jax.vmap(generate_single_spiral, in_axes=(0, 0, 0))
    observations = batch_generate(phase_offsets, radius_scales, noise_keys)
    times = jnp.tile(jnp.linspace(0, 4 * jnp.pi, n_points), (n_sequences, 1))

    return times, observations


def generate_lorenz_trajectory(
    n_points: int = 100,
    n_sequences: int = 1,
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
        n_sequences: Number of sequences to generate
        dt: Time step
        sigma, rho, beta: Lorenz system parameters
        initial_state: Initial state(s). Shape can be:
            - (3,) for single initial state used for all sequences
            - (n_sequences, 3) for different initial states
            - None for random initial states
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations)
        - times: Shape (n_sequences, n_points) or (n_points,) if n_sequences=1
        - observations: Shape (n_sequences, n_points, 3) or (n_points, 3) if n_sequences=1
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # Handle initial states
    if initial_state is None:
        # Generate random initial states for each sequence
        key, init_key = jax.random.split(key)
        initial_state = jax.random.normal(init_key, (n_sequences, 3)) * 5.0
    elif initial_state.ndim == 1:
        # Single initial state provided, replicate for all sequences
        initial_state = jnp.tile(initial_state[None, :], (n_sequences, 1))

    import diffrax

    def lorenz_dynamics(t, state, args):
        """Lorenz dynamics for diffrax."""
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return jnp.array([dx, dy, dz])

    def integrate_single_trajectory(init_state, noise_key):
        """Integrate a single trajectory using diffrax."""
        # Set up ODE problem
        term = diffrax.ODETerm(lorenz_dynamics)
        solver = diffrax.Dopri5()  # 4/5th order Runge-Kutta

        # Time points
        t0 = 0.0
        t1 = dt * (n_points - 1)
        saveat = diffrax.SaveAt(ts=jnp.arange(n_points) * dt)

        # Solve ODE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=init_state,
            saveat=saveat,
            max_steps=n_points * 10,
        )

        states = solution.ys  # (n_points, 3)

        # Add observation noise
        noise = jax.random.normal(noise_key, states.shape) * noise_std
        return states + noise

    # Split keys for noise
    key, noise_key = jax.random.split(key)
    noise_keys = jax.random.split(noise_key, n_sequences)

    # Always use vmap for consistent batch processing
    batch_integrate = jax.vmap(integrate_single_trajectory, in_axes=(0, 0))
    observations = batch_integrate(initial_state, noise_keys)  # (n_sequences, n_points, 3)
    times = jnp.tile(jnp.arange(n_points) * dt, (n_sequences, 1))  # (n_sequences, n_points)

    return times, observations


def generate_pendulum_trajectory(
    n_points: int = 200,
    n_sequences: int = 1,
    dt: float = 0.05,
    length: float = 1.0,
    gravity: float = 9.81,
    damping: float = 0.1,
    initial_angle: float = None,
    initial_velocity: float = None,
    noise_std: float = 0.05,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate damped pendulum trajectory.

    Args:
        n_points: Number of time points
        n_sequences: Number of sequences to generate
        dt: Time step
        length: Pendulum length
        gravity: Gravitational acceleration
        damping: Damping coefficient
        initial_angle: Initial angle (None for random)
        initial_velocity: Initial angular velocity (None for random)
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations) where observations are [x, y] positions
        - times: Shape (n_sequences, n_points) or (n_points,) if n_sequences=1
        - observations: Shape (n_sequences, n_points, 2) or (n_points, 2) if n_sequences=1
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    import diffrax

    # Generate initial conditions
    key, init_key = jax.random.split(key)
    if initial_angle is None:
        initial_angles = jax.random.uniform(init_key, (n_sequences,), minval=-jnp.pi, maxval=jnp.pi)
    else:
        initial_angles = jnp.full((n_sequences,), initial_angle)

    if initial_velocity is None:
        key, vel_key = jax.random.split(key)
        initial_velocities = jax.random.normal(vel_key, (n_sequences,)) * 2.0
    else:
        initial_velocities = jnp.full((n_sequences,), initial_velocity)

    def pendulum_dynamics(t, state, args):
        """Pendulum dynamics for diffrax."""
        angle, velocity = state
        acceleration = -(gravity / length) * jnp.sin(angle) - damping * velocity
        return jnp.array([velocity, acceleration])

    def integrate_single_pendulum(init_angle, init_vel, noise_key):
        """Integrate a single pendulum trajectory."""
        term = diffrax.ODETerm(pendulum_dynamics)
        solver = diffrax.Dopri5()

        t0 = 0.0
        t1 = dt * (n_points - 1)
        saveat = diffrax.SaveAt(ts=jnp.arange(n_points) * dt)

        y0 = jnp.array([init_angle, init_vel])

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=y0,
            saveat=saveat,
            max_steps=n_points * 10,
        )

        states = solution.ys  # (n_points, 2)
        angles = states[:, 0]

        # Convert to Cartesian coordinates
        x = length * jnp.sin(angles)
        y = -length * jnp.cos(angles)
        observations = jnp.stack([x, y], axis=1)

        # Add noise
        noise = jax.random.normal(noise_key, observations.shape) * noise_std
        return observations + noise

    # Split keys for noise
    key, noise_key = jax.random.split(key)
    noise_keys = jax.random.split(noise_key, n_sequences)

    # Always use vmap for consistent batch processing
    batch_integrate = jax.vmap(integrate_single_pendulum, in_axes=(0, 0, 0))
    observations = batch_integrate(initial_angles, initial_velocities, noise_keys)
    times = jnp.tile(jnp.arange(n_points) * dt, (n_sequences, 1))

    return times, observations


def generate_oscillator_trajectory(
    n_points: int = 200,
    n_sequences: int = 1,
    dt: float = 0.1,
    frequency: float = None,
    damping: float = None,
    amplitude: float = None,
    phase: float = None,
    noise_std: float = 0.1,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate damped harmonic oscillator trajectory.

    Args:
        n_points: Number of time points
        n_sequences: Number of sequences to generate
        dt: Time step
        frequency: Oscillation frequency (None for random)
        damping: Damping coefficient (None for random)
        amplitude: Initial amplitude (None for random)
        phase: Phase offset (None for random)
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations) where observations are [position, velocity]
        - times: Shape (n_sequences, n_points) or (n_points,) if n_sequences=1
        - observations: Shape (n_sequences, n_points, 2) or (n_points, 2) if n_sequences=1
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # Generate parameters for each sequence
    key, param_key = jax.random.split(key)

    if frequency is None:
        frequencies = jax.random.uniform(param_key, (n_sequences,), minval=0.5, maxval=2.0)
    else:
        frequencies = jnp.full((n_sequences,), frequency)

    if damping is None:
        key, damp_key = jax.random.split(key)
        dampings = jax.random.uniform(damp_key, (n_sequences,), minval=0.01, maxval=0.1)
    else:
        dampings = jnp.full((n_sequences,), damping)

    if amplitude is None:
        key, amp_key = jax.random.split(key)
        amplitudes = jax.random.uniform(amp_key, (n_sequences,), minval=0.5, maxval=2.0)
    else:
        amplitudes = jnp.full((n_sequences,), amplitude)

    if phase is None:
        key, phase_key = jax.random.split(key)
        phases = jax.random.uniform(phase_key, (n_sequences,), minval=0, maxval=2*jnp.pi)
    else:
        phases = jnp.full((n_sequences,), phase)

    def generate_single_oscillator(freq, damp, amp, ph, noise_key):
        """Generate a single oscillator trajectory."""
        times = jnp.arange(n_points) * dt
        omega = 2 * jnp.pi * freq

        # Analytical solution for damped harmonic oscillator
        exp_decay = jnp.exp(-damp * times)
        position = amp * exp_decay * jnp.cos(omega * times + ph)
        velocity = amp * exp_decay * (-damp * jnp.cos(omega * times + ph) - omega * jnp.sin(omega * times + ph))

        observations = jnp.stack([position, velocity], axis=1)

        # Add noise
        noise = jax.random.normal(noise_key, observations.shape) * noise_std
        return observations + noise

    # Split keys for noise
    key, noise_key = jax.random.split(key)
    noise_keys = jax.random.split(noise_key, n_sequences)

    # Always use vmap for consistent batch processing
    batch_generate = jax.vmap(generate_single_oscillator, in_axes=(0, 0, 0, 0, 0))
    observations = batch_generate(frequencies, dampings, amplitudes, phases, noise_keys)
    times = jnp.tile(jnp.arange(n_points) * dt, (n_sequences, 1))

    return times, observations


def generate_van_der_pol_trajectory(
    n_points: int = 500,
    n_sequences: int = 1,
    dt: float = 0.02,
    mu: float = None,
    initial_state: Optional[jnp.ndarray] = None,
    noise_std: float = 0.05,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Van der Pol oscillator trajectory.

    Args:
        n_points: Number of time points
        n_sequences: Number of sequences to generate
        dt: Time step
        mu: Van der Pol parameter (None for random)
        initial_state: Initial state(s) [x, dx/dt] (None for random)
        noise_std: Standard deviation of observation noise
        key: Random key

    Returns:
        Tuple of (times, observations)
        - times: Shape (n_sequences, n_points) or (n_points,) if n_sequences=1
        - observations: Shape (n_sequences, n_points, 2) or (n_points, 2) if n_sequences=1
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    import diffrax

    # Generate parameters for each sequence
    key, param_key = jax.random.split(key)

    if mu is None:
        mus = jax.random.uniform(param_key, (n_sequences,), minval=0.5, maxval=2.0)
    else:
        mus = jnp.full((n_sequences,), mu)

    # Handle initial states
    if initial_state is None:
        key, init_key = jax.random.split(key)
        initial_states = jax.random.uniform(init_key, (n_sequences, 2), minval=-2.0, maxval=2.0)
    elif initial_state.ndim == 1:
        initial_states = jnp.tile(initial_state[None, :], (n_sequences, 1))
    else:
        initial_states = initial_state

    def van_der_pol_dynamics(t, state, args):
        """Van der Pol dynamics for diffrax."""
        mu_val = args
        x, y = state
        dx = y
        dy = mu_val * (1 - x**2) * y - x
        return jnp.array([dx, dy])

    def integrate_single_vdp(mu_val, init_state, noise_key):
        """Integrate a single Van der Pol trajectory."""
        term = diffrax.ODETerm(van_der_pol_dynamics)
        solver = diffrax.Dopri5()

        t0 = 0.0
        t1 = dt * (n_points - 1)
        saveat = diffrax.SaveAt(ts=jnp.arange(n_points) * dt)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=init_state,
            args=mu_val,
            saveat=saveat,
            max_steps=n_points * 10,
        )

        states = solution.ys  # (n_points, 2)

        # Add noise
        noise = jax.random.normal(noise_key, states.shape) * noise_std
        return states + noise

    # Split keys for noise
    key, noise_key = jax.random.split(key)
    noise_keys = jax.random.split(noise_key, n_sequences)

    # Always use vmap for consistent batch processing
    batch_integrate = jax.vmap(integrate_single_vdp, in_axes=(0, 0, 0))
    observations = batch_integrate(mus, initial_states, noise_keys)
    times = jnp.tile(jnp.arange(n_points) * dt, (n_sequences, 1))

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