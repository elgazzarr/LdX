"""Latent SDE models for stochastic dynamics.

This implementation follows the variational inference approach from:
https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py

Key improvements:
- Separate posterior (conditioned on observations) and prior drift functions
- Observation encoder that provides context to posterior drift
- Augmented dynamics for computing KL divergence during SDE solving
"""

from typing import Tuple, Optional, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from ._compat import ModuleList
import diffrax
import lineax

from .base import LatentModel, DynamicsFunction
from .encoders import MLPEncoder, RNNEncoder
from .decoders import MLPDecoder, GaussianDecoder


class LatentSDEEncoder(nnx.Module):
    """Encoder for observations to provide context to posterior drift.

    Encodes observations at each time step using positional time encoding
    and a neural network. This provides context for the posterior drift function
    during variational inference.
    """

    def __init__(
        self,
        obs_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        time_encoding_dim: int = 16,
        rngs: nnx.Rngs = None
    ):
        """Initialize the encoder.

        Args:
            obs_dim: Dimension of observations
            output_dim: Output context dimension
            hidden_dims: Hidden layer dimensions for MLP
            time_encoding_dim: Dimension of positional time encoding
            rngs: Random number generators
        """
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.time_encoding_dim = time_encoding_dim

        # Build hidden layers
        self.hidden_layers = ModuleList()
        input_dim = obs_dim + time_encoding_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = nnx.Linear(input_dim, output_dim, rngs=rngs)

    def positional_encoding(self, t: jnp.ndarray) -> jnp.ndarray:
        """Create positional encoding for time using sin/cos functions.

        Args:
            t: Time values of shape (...,)

        Returns:
            Positional encoding of shape (..., time_encoding_dim)
        """
        # Create frequencies for positional encoding
        freq = jnp.arange(self.time_encoding_dim // 2)
        freq = 2.0 * jnp.pi * (2.0 ** freq)  # Exponentially spaced frequencies

        # Reshape t for broadcasting
        t_expanded = jnp.expand_dims(t, axis=-1)  # (..., 1)

        # Compute sin and cos encodings
        sin_encoding = jnp.sin(freq * t_expanded)  # (..., time_encoding_dim // 2)
        cos_encoding = jnp.cos(freq * t_expanded)  # (..., time_encoding_dim // 2)

        # Concatenate sin and cos
        encoding = jnp.concatenate([sin_encoding, cos_encoding], axis=-1)
        return encoding

    def __call__(self, obs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Encode observations and time to context vector.

        Args:
            obs: Observations of shape (..., obs_dim)
            t: Time values of shape (...,)

        Returns:
            Context vectors of shape (..., output_dim)
        """
        # Get positional encoding of time
        time_encoding = self.positional_encoding(t)

        # Concatenate observation and time encoding
        x = jnp.concatenate([obs, time_encoding], axis=-1)

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.tanh(x)

        # Output layer
        x = self.output_layer(x)

        return x


class PosteriorDrift(nnx.Module):
    """Posterior drift function conditioned on observations and optionally control.

    This drift function is used during training and is conditioned on
    the encoded observations at each time step and optional control inputs.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        control_dim: int = 0,
        hidden_dims: Tuple[int, ...] = (64, 64),
        rngs: nnx.Rngs = None
    ):
        """Initialize posterior drift.

        Args:
            latent_dim: Dimension of latent state
            context_dim: Dimension of context from encoder
            control_dim: Dimension of control inputs (0 for no control)
            hidden_dims: Hidden layer dimensions
            rngs: Random number generators
        """
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.control_dim = control_dim
        self.hidden_dims = hidden_dims

        # MLP takes [latent_state, time, context, control] -> drift
        input_dim = latent_dim + 1 + context_dim + control_dim
        self.layers = ModuleList()

        for hidden_dim in hidden_dims:
            self.layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        self.output_layer = nnx.Linear(input_dim, latent_dim, rngs=rngs)

    def __call__(
        self,
        z: jnp.ndarray,
        t: jnp.ndarray,
        context: jnp.ndarray,
        u: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Compute posterior drift conditioned on context and control.

        Args:
            z: Latent state of shape (..., latent_dim)
            t: Time of shape (...,) or scalar
            context: Context from encoder of shape (..., context_dim)
            u: Control input of shape (..., control_dim) or None

        Returns:
            Drift of shape (..., latent_dim)
        """
        # Prepare time for concatenation
        t = jnp.asarray(t)
        if t.ndim == 0:  # scalar time
            t = jnp.broadcast_to(t, z.shape[:-1] + (1,))
        elif t.ndim == 1:  # batch of times
            t = t[..., None]

        # Concatenate [z, t, context]
        x = jnp.concatenate([z, t, context], axis=-1)

        # Add control if provided
        if u is not None:
            if u.ndim == 1 and self.control_dim > 1:
                u = u[None, :]
            x = jnp.concatenate([x, u], axis=-1)
        elif self.control_dim > 0:
            # If control expected but not provided, use zeros
            u_zeros = jnp.zeros(z.shape[:-1] + (self.control_dim,))
            x = jnp.concatenate([x, u_zeros], axis=-1)

        # Pass through MLP
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.tanh(x)

        x = self.output_layer(x)
        return x


class PriorDrift(nnx.Module):
    """Prior drift function with optional control inputs.

    This is the generative model drift, used during sampling without
    observations. Often parameterized as linear mean-reverting dynamics
    or a learned unconditional neural network. Can be conditioned on control.
    """

    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        hidden_dims: Tuple[int, ...] = (64,),
        use_linear: bool = False,
        rngs: nnx.Rngs = None
    ):
        """Initialize prior drift.

        Args:
            latent_dim: Dimension of latent state
            control_dim: Dimension of control inputs (0 for no control)
            hidden_dims: Hidden layer dimensions (if not using linear)
            use_linear: If True, use simple linear mean-reverting dynamics
            rngs: Random number generators
        """
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.hidden_dims = hidden_dims
        self.use_linear = use_linear

        if use_linear:
            # Linear mean-reverting: dz = -theta * z * dt + B * u * dt + sigma * dW
            # Parameterize theta and control matrix B
            self.theta = nnx.Param(jnp.ones(latent_dim) * 0.1)
            if control_dim > 0:
                self.B = nnx.Param(jax.random.normal(rngs(), (latent_dim, control_dim)) * 0.1)
        else:
            # Neural network drift
            input_dim = latent_dim + 1 + control_dim  # state + time + control
            self.layers = ModuleList()

            for hidden_dim in hidden_dims:
                self.layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
                input_dim = hidden_dim

            self.output_layer = nnx.Linear(input_dim, latent_dim, rngs=rngs)

    def __call__(self, z: jnp.ndarray, t: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute prior drift.

        Args:
            z: Latent state of shape (..., latent_dim)
            t: Time of shape (...,) or scalar
            u: Control input of shape (..., control_dim) or None

        Returns:
            Drift of shape (..., latent_dim)
        """
        if self.use_linear:
            # Mean-reverting: -theta * z + B * u
            drift = -self.theta.value * z
            if u is not None and self.control_dim > 0:
                drift = drift + jnp.dot(u, self.B.value.T)
            return drift
        else:
            # Neural network drift
            t = jnp.asarray(t)
            if t.ndim == 0:  # scalar time
                t = jnp.broadcast_to(t, z.shape[:-1] + (1,))
            elif t.ndim == 1:  # batch of times
                t = t[..., None]

            x = jnp.concatenate([z, t], axis=-1)

            # Add control if provided
            if u is not None:
                if u.ndim == 1 and self.control_dim > 1:
                    u = u[None, :]
                x = jnp.concatenate([x, u], axis=-1)
            elif self.control_dim > 0:
                # If control expected but not provided, use zeros
                u_zeros = jnp.zeros(z.shape[:-1] + (self.control_dim,))
                x = jnp.concatenate([x, u_zeros], axis=-1)

            for layer in self.layers:
                x = layer(x)
                x = jax.nn.tanh(x)

            x = self.output_layer(x)
            return x


class DriftDiffusionSDE(nnx.Module):
    """SDE with separate drift and diffusion functions."""

    def __init__(
        self,
        drift_func: DynamicsFunction,
        diffusion_func: DynamicsFunction,
        rngs: nnx.Rngs = None
    ):
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, args) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return drift and diffusion terms.

        Args:
            t: Time
            y: State
            args: Additional arguments

        Returns:
            Tuple of (drift, diffusion)
        """
        drift = self.drift_func(y, t)
        diffusion = self.diffusion_func(y, t)
        return drift, diffusion


class LatentSDE(LatentModel):
    """Latent SDE model with proper variational inference.

    This implementation follows the variational approach where:
    - During training: uses posterior drift conditioned on observations
    - During generation: uses prior drift without observations
    - Computes KL divergence via augmented SDE dynamics
    """

    def __init__(
        self,
        posterior_drift: PosteriorDrift,
        prior_drift: PriorDrift,
        diffusion_func: DynamicsFunction,
        obs_encoder: LatentSDEEncoder,
        initial_encoder: nnx.Module,  # Encodes initial observation to latent distribution
        decoder: nnx.Module,
        latent_dim: int,
        solver: diffrax.AbstractSolver = diffrax.EulerHeun(),
        dt0: float = 0.01,
        rngs: nnx.Rngs = None
    ):
        """Initialize Latent SDE model.

        Args:
            posterior_drift: Drift conditioned on observations (for training)
            prior_drift: Unconditional drift (for generation)
            diffusion_func: Diffusion function (shared between posterior and prior)
            obs_encoder: Encoder for observations at each time step
            initial_encoder: Encoder for initial observation to get latent distribution
            decoder: Decoder from latent to observations
            latent_dim: Dimension of latent space
            solver: SDE solver
            dt0: Initial time step
            rngs: Random number generators
        """
        super().__init__(latent_dim, rngs)
        self.posterior_drift = posterior_drift
        self.prior_drift = prior_drift
        self.diffusion_func = diffusion_func
        self.obs_encoder = obs_encoder
        self.initial_encoder = initial_encoder
        self.decoder = decoder
        self.solver = solver
        self.dt0 = dt0

    def encode(self, observations: jnp.ndarray, times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode initial observation to latent distribution parameters.

        Args:
            observations: Observations of shape (seq_len, obs_dim) or (batch, seq_len, obs_dim)
            times: Times (not used, for API compatibility)

        Returns:
            Tuple of (mean, logvar) for initial latent distribution
        """
        # Use first observation to get initial latent distribution
        if observations.ndim == 3:
            initial_obs = observations[:, 0, :]  # (batch, obs_dim)
        else:
            initial_obs = observations[0:1, :]  # (1, obs_dim)

        return self.initial_encoder(initial_obs)

    def decode(self, latent_states: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
        """Decode latent states to observations.

        Args:
            latent_states: Latent states of shape (seq_len, latent_dim) or (batch, seq_len, latent_dim)
            times: Times (not used, for API compatibility)

        Returns:
            Decoded observations
        """
        return self.decoder(latent_states)

    def encode_observations(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray
    ) -> jnp.ndarray:
        """Encode all observations to context vectors for posterior drift.

        Args:
            observations: Observations of shape (seq_len, obs_dim)
            times: Times of shape (seq_len,)

        Returns:
            Context vectors of shape (seq_len, context_dim)
        """
        # Use vmap to encode each observation with its time
        return jax.vmap(self.obs_encoder)(observations, times)

    def sample_posterior(
        self,
        initial_state: jnp.ndarray,
        times: jnp.ndarray,
        observations: jnp.ndarray,
        key: jax.random.PRNGKey,
        return_kl: bool = False,
        controls: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Sample from posterior (conditioned on observations) using augmented dynamics.

        This is used during training. The posterior drift is conditioned on observations.

        Args:
            initial_state: Initial latent state of shape (latent_dim,)
            times: Times to sample at of shape (seq_len,)
            observations: Observations of shape (seq_len, obs_dim)
            key: Random key
            return_kl: If True, return KL divergence as well
            controls: Optional control inputs of shape (seq_len, control_dim)

        Returns:
            If return_kl=False: Trajectory of shape (seq_len, latent_dim)
            If return_kl=True: Tuple of (trajectory, kl_divergence)
        """
        # Encode observations to get context at each time step
        contexts = self.encode_observations(observations, times)

        # Create interpolation for contexts using Diffrax LinearInterpolation
        context_interp = diffrax.LinearInterpolation(ts=times, ys=contexts)

        # Create control interpolation using Diffrax LinearInterpolation
        if controls is not None:
            control_interp = diffrax.LinearInterpolation(ts=times, ys=controls)
        else:
            control_interp = None

        if return_kl:
            # Augmented dynamics: state = [z, log_ratio] where log_ratio tracks KL
            augmented_dim = self.latent_dim + 1
            augmented_initial = jnp.concatenate([initial_state, jnp.array([0.0])])

            def augmented_drift(t, y, args):
                z = y[: self.latent_dim]
                context = context_interp.evaluate(t)
                u = control_interp.evaluate(t) if control_interp is not None else None

                # Posterior and prior drifts
                f_post = self.posterior_drift(z, t, context, u)
                f_prior = self.prior_drift(z, t, u)
                g = self.diffusion_func(z, t, u)

                # Compute KL increment: 0.5 * ||g^{-1}(f_post - f_prior)||^2
                diff = f_post - f_prior
                # Assume diagonal diffusion
                kl_increment = 0.5 * jnp.sum((diff / (g + 1e-8)) ** 2)

                # Return augmented drift
                dz = f_post
                dlog_ratio = kl_increment
                return jnp.concatenate([dz, jnp.array([dlog_ratio])])

            def augmented_diffusion(t, y, args):
                z = y[: self.latent_dim]
                u = control_interp.evaluate(t) if control_interp is not None else None
                g = self.diffusion_func(z, t, u)
                # Diffusion only affects z, not log_ratio
                g_full = jnp.concatenate([g, jnp.array([0.0])])
                return lineax.DiagonalLinearOperator(g_full)

            # Set up augmented SDE terms
            drift_term = diffrax.ODETerm(augmented_drift)
            diffusion_term = diffrax.ControlTerm(
                augmented_diffusion,
                diffrax.VirtualBrownianTree(
                    t0=times[0],
                    t1=times[-1],
                    tol=1e-3,
                    shape=(augmented_dim,),
                    key=key
                )
            )

            # Solve augmented SDE
            solution = diffrax.diffeqsolve(
                diffrax.MultiTerm(drift_term, diffusion_term),
                self.solver,
                t0=times[0],
                t1=times[-1],
                dt0=self.dt0,
                y0=augmented_initial,
                saveat=diffrax.SaveAt(ts=times)
            )

            # Extract trajectory and KL
            trajectory = solution.ys[:, : self.latent_dim]
            kl_divergence = solution.ys[-1, self.latent_dim]  # Final log_ratio value
            return trajectory, kl_divergence

        else:
            # Standard posterior sampling without KL tracking
            def drift(t, y, args):
                context = context_interp.evaluate(t)
                u = control_interp.evaluate(t) if control_interp is not None else None
                return self.posterior_drift(y, t, context, u)

            def diffusion(t, y, args):
                u = control_interp.evaluate(t) if control_interp is not None else None
                g = self.diffusion_func(y, t, u)
                return lineax.DiagonalLinearOperator(g)

            drift_term = diffrax.ODETerm(drift)
            diffusion_term = diffrax.ControlTerm(
                diffusion,
                diffrax.VirtualBrownianTree(
                    t0=times[0],
                    t1=times[-1],
                    tol=1e-3,
                    shape=(self.latent_dim,),
                    key=key
                )
            )

            solution = diffrax.diffeqsolve(
                diffrax.MultiTerm(drift_term, diffusion_term),
                self.solver,
                t0=times[0],
                t1=times[-1],
                dt0=self.dt0,
                y0=initial_state,
                saveat=diffrax.SaveAt(ts=times)
            )

            return solution.ys

    def sample_trajectory(
        self,
        initial_state: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey,
        controls: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Sample from prior (unconditional) for generation.

        This is used for generating new trajectories without observations.

        Args:
            initial_state: Initial latent state of shape (latent_dim,)
            times: Times to sample at of shape (seq_len,)
            key: Random key
            controls: Optional control inputs of shape (seq_len, control_dim)

        Returns:
            Trajectory of shape (seq_len, latent_dim)
        """
        # Create control interpolation using Diffrax LinearInterpolation
        if controls is not None:
            control_interp = diffrax.LinearInterpolation(ts=times, ys=controls)
        else:
            control_interp = None

        def drift(t, y, args):
            u = control_interp.evaluate(t) if control_interp is not None else None
            return self.prior_drift(y, t, u)

        def diffusion(t, y, args):
            u = control_interp.evaluate(t) if control_interp is not None else None
            g = self.diffusion_func(y, t, u)
            return lineax.DiagonalLinearOperator(g)

        # Set up SDE terms
        drift_term = diffrax.ODETerm(drift)
        diffusion_term = diffrax.ControlTerm(
            diffusion,
            diffrax.VirtualBrownianTree(
                t0=times[0],
                t1=times[-1],
                tol=1e-3,
                shape=(self.latent_dim,),
                key=key
            )
        )

        # Solve SDE
        solution = diffrax.diffeqsolve(
            diffrax.MultiTerm(drift_term, diffusion_term),
            self.solver,
            t0=times[0],
            t1=times[-1],
            dt0=self.dt0,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=times)
        )

        return solution.ys

    def reparameterize(self, mean: jnp.ndarray, logvar: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Reparameterization trick for sampling from latent distribution."""
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mean.shape)
        return mean + std * eps

    def forward(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey,
        return_kl: bool = True,
        controls: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass through the model using posterior dynamics.

        Args:
            observations: Shape (batch_size, seq_len, obs_dim) or (seq_len, obs_dim)
            times: Shape (seq_len,)
            key: Random key
            return_kl: If True, compute KL divergence via augmented dynamics
            controls: Optional control inputs of shape (batch_size, seq_len, control_dim) or (seq_len, control_dim)

        Returns:
            Tuple of (reconstructed_obs, latent_mean, latent_logvar, latent_trajectory, kl_divergence)
            If return_kl=False, kl_divergence is None
        """
        # Handle both batched and single sequence inputs
        is_batched = observations.ndim == 3
        if not is_batched:
            observations = observations[None, ...]  # Add batch dimension

        batch_size = observations.shape[0]

        # Encode initial observation to get latent distribution
        latent_mean, latent_logvar = self.encode(observations, times)

        # Sample initial latent states
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        initial_states = jax.vmap(self.reparameterize, in_axes=(0, 0, 0))(
            latent_mean, latent_logvar, keys
        )

        # Flatten times if needed
        if times.ndim == 2:
            times_flat = times[0]  # Assume same times for all batch elements
        else:
            times_flat = times

        # Handle control inputs
        if controls is not None:
            if controls.ndim == 3:
                # Batched controls: (batch_size, seq_len, control_dim)
                controls_batched = controls
            else:
                # Single control sequence: (seq_len, control_dim)
                # Expand to batch dimension
                controls_batched = jnp.tile(controls[None, ...], (batch_size, 1, 1))
        else:
            controls_batched = None

        # Sample trajectories using posterior dynamics (conditioned on observations)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)

        if return_kl:
            # Use augmented dynamics to track KL
            if controls_batched is not None:
                def sample_with_kl(z0, obs, ctrl, k):
                    return self.sample_posterior(z0, times_flat, obs, k, return_kl=True, controls=ctrl)

                # Vectorize over batch
                results = jax.vmap(sample_with_kl, in_axes=(0, 0, 0, 0))(
                    initial_states, observations, controls_batched, keys
                )
            else:
                def sample_with_kl(z0, obs, k):
                    return self.sample_posterior(z0, times_flat, obs, k, return_kl=True)

                # Vectorize over batch
                results = jax.vmap(sample_with_kl, in_axes=(0, 0, 0))(
                    initial_states, observations, keys
                )
            latent_trajectory, kl_divergence = results
        else:
            # Standard posterior sampling
            if controls_batched is not None:
                def sample_no_kl(z0, obs, ctrl, k):
                    return self.sample_posterior(z0, times_flat, obs, k, return_kl=False, controls=ctrl)

                latent_trajectory = jax.vmap(sample_no_kl, in_axes=(0, 0, 0, 0))(
                    initial_states, observations, controls_batched, keys
                )
            else:
                def sample_no_kl(z0, obs, k):
                    return self.sample_posterior(z0, times_flat, obs, k, return_kl=False)

                latent_trajectory = jax.vmap(sample_no_kl, in_axes=(0, 0, 0))(
                    initial_states, observations, keys
                )
            kl_divergence = None

        # Decode trajectory
        reconstructed_obs = self.decode(latent_trajectory, times)

        # Remove batch dimension if input wasn't batched
        if not is_batched:
            reconstructed_obs = reconstructed_obs[0]
            latent_mean = latent_mean[0]
            latent_logvar = latent_logvar[0]
            latent_trajectory = latent_trajectory[0]
            if kl_divergence is not None:
                kl_divergence = kl_divergence[0]

        return reconstructed_obs, latent_mean, latent_logvar, latent_trajectory, kl_divergence

    def __call__(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        key: jax.random.PRNGKey,
        return_kl: bool = True,
        controls: Optional[jnp.ndarray] = None
    ):
        """Call method for NNX compatibility."""
        return self.forward(observations, times, key, return_kl, controls)


class DiagonalDiffusion(DynamicsFunction):
    """Diagonal diffusion function for SDE with optional control dependence."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...],
        latent_dim: int,
        control_dim: int = 0,
        rngs: nnx.Rngs = None
    ):
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.control_dim = control_dim

        # Initialize layers - input size is latent_dim + 1 (for time) + control_dim
        input_dim = latent_dim + 1 + control_dim
        self.layers = ModuleList()

        for hidden_dim in hidden_dims:
            self.layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = nnx.Linear(input_dim, latent_dim, rngs=rngs)

    def __call__(self, z: jnp.ndarray, t: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute diagonal diffusion coefficients.

        Args:
            z: Latent state of shape (batch_size, latent_dim)
            t: Time of shape (batch_size,) or scalar
            u: Control input of shape (batch_size, control_dim) or None

        Returns:
            Diagonal diffusion matrix of shape (batch_size, latent_dim)
        """
        # Concatenate state and time
        t = jnp.asarray(t)  # Ensure t is a JAX array
        if t.ndim == 0:  # scalar time
            t = jnp.broadcast_to(t, z.shape[:-1] + (1,))
        elif t.ndim == 1:  # batch of times
            t = t[..., None]

        x = jnp.concatenate([z, t], axis=-1)

        # Add control if provided
        if u is not None:
            if u.ndim == 1 and self.control_dim > 1:
                u = u[None, :]
            x = jnp.concatenate([x, u], axis=-1)
        elif self.control_dim > 0:
            # If control expected but not provided, use zeros
            u_zeros = jnp.zeros(z.shape[:-1] + (self.control_dim,))
            x = jnp.concatenate([x, u_zeros], axis=-1)

        for layer in self.layers:
            x = layer(x)
            x = jax.nn.tanh(x)

        # Output layer
        x = self.output_layer(x)

        # Apply softplus to ensure positive diffusion
        x = jax.nn.softplus(x)
        return x


def create_latent_sde(
    latent_dim: int,
    obs_dim: int,
    hidden_dims: Tuple[int, ...] = (64, 64),
    posterior_hidden_dims: Tuple[int, ...] = (64, 64),
    prior_hidden_dims: Tuple[int, ...] = (64,),
    diffusion_hidden_dims: Tuple[int, ...] = (32, 32),
    control_dim: int = 0,
    context_dim: int = 32,
    obs_encoder_hidden_dims: Tuple[int, ...] = (64, 64),
    obs_encoder_time_encoding_dim: int = 16,
    encoder_type: str = "mlp",
    decoder_type: str = "mlp",
    use_linear_prior: bool = False,
    rngs: nnx.Rngs = None
) -> LatentSDE:
    """Factory function to create a LatentSDE model with variational inference.

    Args:
        latent_dim: Dimension of latent space
        obs_dim: Dimension of observations
        hidden_dims: Hidden dimensions for initial encoder/decoder
        posterior_hidden_dims: Hidden dimensions for posterior drift function
        prior_hidden_dims: Hidden dimensions for prior drift function
        diffusion_hidden_dims: Hidden dimensions for diffusion function
        control_dim: Dimension of control inputs (0 for no control)
        context_dim: Dimension of context vectors from observation encoder
        obs_encoder_hidden_dims: Hidden dimensions for observation encoder
        obs_encoder_time_encoding_dim: Dimension of positional time encoding
        encoder_type: Type of initial encoder ("mlp" or "rnn")
        decoder_type: Type of decoder ("mlp" or "gaussian")
        use_linear_prior: If True, use linear mean-reverting prior drift
        rngs: Random number generators

    Returns:
        Configured LatentSDE model with posterior/prior dynamics
    """
    # Create observation encoder for context
    obs_encoder = LatentSDEEncoder(
        obs_dim=obs_dim,
        output_dim=context_dim,
        hidden_dims=obs_encoder_hidden_dims,
        time_encoding_dim=obs_encoder_time_encoding_dim,
        rngs=rngs
    )

    # Create posterior drift (conditioned on observations)
    posterior_drift = PosteriorDrift(
        latent_dim=latent_dim,
        context_dim=context_dim,
        control_dim=control_dim,
        hidden_dims=posterior_hidden_dims,
        rngs=rngs
    )

    # Create prior drift (unconditional)
    prior_drift = PriorDrift(
        latent_dim=latent_dim,
        control_dim=control_dim,
        hidden_dims=prior_hidden_dims,
        use_linear=use_linear_prior,
        rngs=rngs
    )

    # Create diffusion function (shared between posterior and prior)
    diffusion_func = DiagonalDiffusion(
        diffusion_hidden_dims,
        latent_dim=latent_dim,
        control_dim=control_dim,
        rngs=rngs
    )

    # Create initial encoder (encodes first observation to latent distribution)
    if encoder_type == "mlp":
        initial_encoder = MLPEncoder(hidden_dims, latent_dim, obs_dim, rngs)
    elif encoder_type == "rnn":
        initial_encoder = RNNEncoder(hidden_dims[0], latent_dim, obs_dim, rngs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Create decoder
    if decoder_type == "mlp":
        decoder = MLPDecoder(hidden_dims, obs_dim, latent_dim, rngs)
    elif decoder_type == "gaussian":
        decoder = GaussianDecoder(hidden_dims, obs_dim, latent_dim, rngs)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    return LatentSDE(
        posterior_drift=posterior_drift,
        prior_drift=prior_drift,
        diffusion_func=diffusion_func,
        obs_encoder=obs_encoder,
        initial_encoder=initial_encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        rngs=rngs
    )