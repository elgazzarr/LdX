"""Example of using LatentSDE with external control inputs.

This example demonstrates how to train and use a LatentSDE model with
continuous-time control signals that influence both the drift and diffusion.
We use a controlled Ornstein-Uhlenbeck process as the ground truth system.

System: Controlled Ornstein-Uhlenbeck process
    dz = (θ(μ - z) + B·u(t))dt + σ dW

where u(t) is the external control input that influences the drift.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
import optax
import diffrax
import lineax

import ldx


def generate_controlled_ou_process(
    n_sequences: int = 100,
    n_points: int = 100,
    dt: float = 0.05,
    n_dims: int = 2,
    theta: float = 0.5,
    mu: float = 0.0,
    sigma: float = 0.3,
    control_influence: float = 0.8,
    control_type: str = "sine",
    control_amplitude: float = 1.0,
    control_frequency: float = 0.3,
    obs_noise_std: float = 0.05,
    key: jax.random.PRNGKey = None
):
    """Generate controlled Ornstein-Uhlenbeck process trajectories.

    The dynamics are:
        dz = (theta * (mu - z) + B * u(t)) dt + sigma * dW

    where B is the control influence matrix.

    Args:
        n_sequences: Number of sequences to generate
        n_points: Number of time points per sequence
        dt: Time step
        n_dims: Dimension of the latent space
        theta: Mean-reversion rate
        mu: Mean-reversion level
        sigma: Diffusion coefficient
        control_influence: Strength of control influence
        control_type: Type of control signal ("sine", "step", "random")
        control_amplitude: Amplitude of control signal
        control_frequency: Frequency for sine control
        obs_noise_std: Observation noise standard deviation
        key: Random key

    Returns:
        Tuple of (times, observations, controls)
        - times: (n_sequences, n_points)
        - observations: (n_sequences, n_points, n_dims)
        - controls: (n_sequences, n_points, 1)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    times_array = jnp.arange(n_points) * dt

    # Control influence matrix (first dimension gets most influence)
    key, b_key = jax.random.split(key)
    B = jax.random.normal(b_key, (n_dims,)) * control_influence
    B = B.at[0].set(control_influence)  # First dimension most influenced

    def generate_control_signal(control_key):
        """Generate control signal based on type."""
        if control_type == "sine":
            # Sinusoidal control with random phase
            phase = jax.random.uniform(control_key, minval=0, maxval=2*jnp.pi)
            controls = control_amplitude * jnp.sin(2 * jnp.pi * control_frequency * times_array + phase)
        elif control_type == "step":
            # Multiple step functions
            key1, key2 = jax.random.split(control_key)
            step_times = jax.random.uniform(key1, (2,), minval=0.2, maxval=0.8) * (n_points * dt)
            step_times = jnp.sort(step_times)

            controls = jnp.zeros_like(times_array)
            controls = jnp.where(times_array > step_times[0], control_amplitude, controls)
            controls = jnp.where(times_array > step_times[1], -control_amplitude, controls)
        elif control_type == "random":
            # Smooth random walk control
            increments = jax.random.normal(control_key, (n_points,)) * 0.15
            controls = jnp.cumsum(increments) * control_amplitude / jnp.sqrt(n_points)
            # Apply smoothing
            controls = jnp.convolve(controls, jnp.ones(5)/5, mode='same')
        else:
            raise ValueError(f"Unknown control type: {control_type}")

        return controls

    def integrate_single_trajectory(init_state, controls, noise_key):
        """Integrate a single trajectory with given control using SDE solver."""
        # Create interpolation function for control
        def control_fn(t):
            idx = jnp.searchsorted(times_array, t, side='right') - 1
            idx = jnp.clip(idx, 0, len(times_array) - 2)

            t0, t1 = times_array[idx], times_array[idx + 1]
            u0, u1 = controls[idx], controls[idx + 1]

            alpha = (t - t0) / (t1 - t0 + 1e-8)
            alpha = jnp.clip(alpha, 0.0, 1.0)
            return (1 - alpha) * u0 + alpha * u1

        # Define SDE dynamics
        def drift(t, y, args):
            """Controlled Ornstein-Uhlenbeck drift."""
            u = control_fn(t)
            # Mean-reversion + control influence
            return theta * (mu - y) + B * u

        def diffusion(t, y, args):
            """Constant diffusion."""
            return lineax.DiagonalLinearOperator(jnp.ones(n_dims) * sigma)

        # Set up SDE terms
        drift_term = diffrax.ODETerm(drift)
        diffusion_term = diffrax.ControlTerm(
            diffusion,
            diffrax.VirtualBrownianTree(
                t0=times_array[0],
                t1=times_array[-1],
                tol=1e-3,
                shape=(n_dims,),
                key=noise_key
            )
        )

        # Solve SDE
        solver = diffrax.EulerHeun()
        solution = diffrax.diffeqsolve(
            diffrax.MultiTerm(drift_term, diffusion_term),
            solver,
            t0=times_array[0],
            t1=times_array[-1],
            dt0=dt,
            y0=init_state,
            saveat=diffrax.SaveAt(ts=times_array),
            max_steps=n_points * 10,
        )

        states = solution.ys  # (n_points, n_dims)

        # Add observation noise
        obs_noise = jax.random.normal(noise_key, states.shape) * obs_noise_std
        return states + obs_noise

    # Generate random initial states
    key, init_key = jax.random.split(key)
    initial_states = jax.random.normal(init_key, (n_sequences, n_dims)) * 0.5

    # Generate control signals for each sequence
    key, control_key = jax.random.split(key)
    control_keys = jax.random.split(control_key, n_sequences)
    controls = jax.vmap(generate_control_signal)(control_keys)  # (n_sequences, n_points)

    # Integrate all trajectories
    key, noise_key = jax.random.split(key)
    noise_keys = jax.random.split(noise_key, n_sequences)

    batch_integrate = jax.vmap(integrate_single_trajectory, in_axes=(0, 0, 0))
    observations = batch_integrate(initial_states, controls, noise_keys)  # (n_sequences, n_points, n_dims)

    # Tile times for all sequences
    times = jnp.tile(times_array, (n_sequences, 1))

    # Reshape controls to (n_sequences, n_points, 1)
    controls = controls[..., None]

    return times, observations, controls


def main():
    """Main example function."""
    print("=" * 60)
    print("Controlled Latent SDE Example")
    print("=" * 60)

    # Set random seed
    key = jax.random.PRNGKey(123)

    # Generate synthetic controlled data
    print("\nGenerating controlled Ornstein-Uhlenbeck process...")
    n_sequences = 100
    n_points = 100

    key, data_key = jax.random.split(key)
    times, observations, controls = generate_controlled_ou_process(
        n_sequences=n_sequences,
        n_points=n_points,
        dt=0.05,
        n_dims=2,
        control_type="sine",
        control_amplitude=1.5,
        control_frequency=0.3,
        key=data_key
    )

    print(f"   - Number of sequences: {n_sequences}")
    print(f"   - Time points per sequence: {n_points}")
    print(f"   - Observation shape: {observations.shape}")
    print(f"   - Control shape: {controls.shape}")

    # Normalize data
    normalized_obs, obs_norm_params = ldx.utils.normalize_data(observations, method="standard")
    normalized_controls, control_norm_params = ldx.utils.normalize_data(controls, method="standard")

    # Create train/val split
    (train_obs, train_times), (val_obs, val_times) = ldx.utils.create_train_val_split(
        normalized_obs, times, val_fraction=0.3
    )

    # Split controls similarly
    split_idx = int(n_sequences * 0.7)
    train_controls = normalized_controls[:split_idx]
    val_controls = normalized_controls[split_idx:]

    print(f"   - Training sequences: {train_obs.shape[0]}")
    print(f"   - Validation sequences: {val_obs.shape[0]}")

    # Model configuration
    latent_dim = 5
    obs_dim = observations.shape[2]
    control_dim = controls.shape[2]

    print(f"\nCreating LatentSDE model with control inputs...")
    print(f"   - Latent dimension: {latent_dim}")
    print(f"   - Observation dimension: {obs_dim}")
    print(f"   - Control dimension: {control_dim}")

    # Create controlled SDE model
    key, model_key = jax.random.split(key)
    rngs = nnx.Rngs(model_key)

    model = ldx.models.create_latent_sde(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dims=(32, 32),
        posterior_hidden_dims=(32, 32),
        prior_hidden_dims=(32,),
        diffusion_hidden_dims=(16, 16),
        control_dim=control_dim,  # Enable control inputs
        context_dim=32,
        obs_encoder_hidden_dims=(32, 32),
        encoder_type="mlp",
        decoder_type="mlp",
        use_linear_prior=False,
        rngs=rngs
    )

    print("   ✅ Model created successfully")

    # Training setup
    print(f"\nSetting up training...")
    learning_rate = 1e-3
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Training loop with manual loss computation
    n_epochs = 200
    print(f"\nTraining for {n_epochs} epochs...")

    train_losses = []
    val_losses = []
    kl_values = []

    for epoch in range(n_epochs):
        key, train_key = jax.random.split(key)

        # Training step on a random training sequence
        train_idx = jax.random.randint(train_key, (), 0, train_obs.shape[0])

        # Custom training step for SDE with controls
        def train_step_with_controls(model, obs, times, controls, key, step):
            """Training step that passes controls to the model."""
            def loss_fn(model):
                # Forward pass with controls (returns KL)
                reconstructed_obs, latent_mean, latent_logvar, latent_trajectory, kl_divergence = model(
                    obs, times, key, return_kl=True, controls=controls
                )

                # Compute losses
                reconstruction_loss = ldx.inference.reconstruction_loss_mse(reconstructed_obs, obs)

                # KL divergence from augmented SDE
                kl_loss = kl_divergence

                # Beta schedule for KL
                beta = jnp.minimum(1.0, step / 50.0)  # Linear warmup over 50 epochs
                total_loss = reconstruction_loss + beta * kl_loss

                return total_loss, {
                    "total_loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_loss": kl_loss,
                    "beta": beta
                }

            # Compute gradients and update
            (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)

            return model, metrics

        model, train_metrics = train_step_with_controls(
            model,
            train_obs[train_idx],
            train_times[train_idx],
            train_controls[train_idx],
            train_key,
            epoch
        )

        train_losses.append(train_metrics["total_loss"])
        kl_values.append(train_metrics["kl_loss"])

        # Validation every 10 epochs
        if epoch % 10 == 0:
            key, val_key = jax.random.split(key)
            val_idx = jax.random.randint(val_key, (), 0, val_obs.shape[0])

            # Validation (no gradient computation)
            reconstructed_obs, latent_mean, latent_logvar, _, kl_divergence = model(
                val_obs[val_idx],
                val_times[val_idx],
                val_key,
                return_kl=True,
                controls=val_controls[val_idx]
            )

            val_recon_loss = ldx.inference.reconstruction_loss_mse(reconstructed_obs, val_obs[val_idx])
            val_kl_loss = kl_divergence
            val_total_loss = val_recon_loss + val_kl_loss

            val_losses.append(val_total_loss)

            print(f"   Epoch {epoch:3d}: Train Loss = {train_metrics['total_loss']:.4f}, "
                  f"Val Loss = {val_total_loss:.4f}, KL = {train_metrics['kl_loss']:.4f}")

    print("Training completed!")

    # Test with the validation control and generate multiple samples
    print("\n" + "=" * 60)
    print("Testing Model with Multiple Stochastic Samples")
    print("=" * 60)

    # Get a test sequence
    test_idx = 0
    test_obs = val_obs[test_idx]
    test_times = val_times[test_idx]
    test_control = val_controls[test_idx]

    # Generate alternative control signals
    times_flat = test_times

    print("\nGenerating multiple stochastic samples...")

    # Generate multiple samples with same control to show stochasticity
    n_samples = 5
    samples = []

    for i in range(n_samples):
        key, sample_key = jax.random.split(key)

        # Use posterior sampling (conditioned on observations) for first sample
        if i == 0:
            reconstructed_obs, _, _, latent_trajectory, _ = model(
                test_obs,
                times_flat,
                sample_key,
                return_kl=False,
                controls=test_control
            )
        else:
            # Use prior sampling (generation) for other samples
            # Get initial state from encoder
            latent_mean, latent_logvar = model.encode(test_obs[None, ...], times_flat)

            # Sample initial state
            key, init_key = jax.random.split(key)
            initial_state = model.reparameterize(latent_mean[0], latent_logvar[0], init_key)

            # Sample trajectory using prior
            key, traj_key = jax.random.split(key)
            latent_trajectory = model.sample_trajectory(
                initial_state,
                times_flat,
                traj_key,
                controls=test_control
            )

            # Decode
            reconstructed_obs = model.decode(latent_trajectory[None, ...], times_flat)[0]

        samples.append(reconstructed_obs)

    print(f"   ✅ Generated {n_samples} stochastic samples")

    # Denormalize for visualization
    test_obs_denorm = ldx.utils.denormalize_data(test_obs[None, ...], obs_norm_params)[0]
    samples_denorm = [ldx.utils.denormalize_data(s[None, ...], obs_norm_params)[0] for s in samples]
    control_denorm = ldx.utils.denormalize_data(test_control[None, ...], control_norm_params)[0]

    # Visualization
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    # Plot 1: Training curves
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(train_losses, label='Training Loss', alpha=0.7)
    if val_losses:
        val_epochs = np.arange(0, n_epochs, 10)[:len(val_losses)]
        ax.plot(val_epochs, val_losses, label='Validation Loss', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL divergence
    ax = axes[1]
    ax.plot(kl_values, label='KL Divergence', color='orange', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("controlled_sde_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: controlled_sde_training.png")

    # Plot 2: Multiple stochastic samples
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_samples = plt.cm.viridis(np.linspace(0.2, 0.9, n_samples))

    # Dimension 1
    ax = axes[0]
    ax.plot(times_flat, test_obs_denorm[:, 0], 'k-',
            label='True', linewidth=2.5, alpha=0.8, zorder=10)

    for i, sample in enumerate(samples_denorm):
        label = 'Samples' if i == 0 else None
        ax.plot(times_flat, sample[:, 0], '--',
                color=colors_samples[i], linewidth=1.5, alpha=0.6, label=label)

    ax.set_xlabel('Time')
    ax.set_ylabel('Dimension 1')
    ax.set_title('Dimension 1: Multiple Stochastic Samples')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Dimension 2
    ax = axes[1]
    ax.plot(times_flat, test_obs_denorm[:, 1], 'k-',
            label='True', linewidth=2.5, alpha=0.8, zorder=10)

    for i, sample in enumerate(samples_denorm):
        label = 'Samples' if i == 0 else None
        ax.plot(times_flat, sample[:, 1], '--',
                color=colors_samples[i], linewidth=1.5, alpha=0.6, label=label)

    ax.set_xlabel('Time')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Dimension 2: Multiple Stochastic Samples')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("controlled_sde_stochastic_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: controlled_sde_stochastic_samples.png")

    # Plot 3: Uncertainty quantification (mean and std over samples)
    fig3, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Dimension 1
    ax = axes[0]
    ax.plot(times_flat, test_obs_denorm[:, 0], 'k-', label='True', linewidth=2, alpha=0.8)

    samples_array = np.array([s[:, 0] for s in samples_denorm])
    mean = np.mean(samples_array, axis=0)
    std = np.std(samples_array, axis=0)

    ax.plot(times_flat, mean, '-', label='Model Mean', color='blue', linewidth=1.5)
    ax.fill_between(times_flat, mean - std, mean + std, color='blue', alpha=0.2, label='± Std')

    ax.set_ylabel('Dimension 1')
    ax.set_title('Uncertainty Quantification: Mean ± Std over Samples')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Dimension 2
    ax = axes[1]
    ax.plot(times_flat, test_obs_denorm[:, 1], 'k-', label='True', linewidth=2, alpha=0.8)

    samples_array = np.array([s[:, 1] for s in samples_denorm])
    mean = np.mean(samples_array, axis=0)
    std = np.std(samples_array, axis=0)

    ax.plot(times_flat, mean, '-', label='Model Mean', color='blue', linewidth=1.5)
    ax.fill_between(times_flat, mean - std, mean + std, color='blue', alpha=0.2, label='± Std')

    ax.set_ylabel('Dimension 2')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Control signal
    ax = axes[2]
    ax.plot(times_flat, control_denorm[:, 0], label='Control Input', color='green', linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Control Input')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("controlled_sde_uncertainty.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: controlled_sde_uncertainty.png")

    # Plot 4: 2D trajectories in state space
    fig4 = plt.figure(figsize=(8, 8))
    ax = fig4.add_subplot(111)

    # Plot true trajectory
    ax.plot(test_obs_denorm[:, 0], test_obs_denorm[:, 1], 'k-',
            label='True', linewidth=2.5, alpha=0.8, zorder=10)

    # Plot all sampled trajectories
    for i, sample in enumerate(samples_denorm):
        label = 'Samples' if i == 0 else None
        ax.plot(sample[:, 0], sample[:, 1], '--',
                color=colors_samples[i], linewidth=1.5, alpha=0.5, label=label)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('State Space Trajectories (2D)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("controlled_sde_trajectories_2d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: controlled_sde_trajectories_2d.png")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nKey Observations:")
    print("   - Model learns control-dependent drift and diffusion")
    print("   - Stochasticity causes variation between samples")
    print("   - Posterior (conditioned) has less uncertainty than prior")
    print("=" * 60)


if __name__ == "__main__":
    main()
