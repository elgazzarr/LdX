"""Basic example of using LatentSDE for stochastic time series modeling.

This example demonstrates:
- Training a latent SDE with proper variational inference
- Using posterior drift (conditioned on observations) during training
- Using prior drift (unconditional) for generation
- Generating multiple samples from the SDE to capture uncertainty
- Plotting mean and 95% confidence intervals
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
import optax

import ldx


def generate_multiple_samples(
    model,
    initial_state: jnp.ndarray,
    times: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_samples: int = 50
):
    """Generate multiple samples from the prior for uncertainty quantification.

    Args:
        model: Trained LatentSDE model
        initial_state: Initial latent state
        times: Time points
        key: Random key
        n_samples: Number of samples to generate

    Returns:
        Array of samples with shape (n_samples, seq_len, latent_dim)
    """
    keys = jax.random.split(key, n_samples)

    # Generate samples using prior (unconditional)
    def sample_fn(k):
        return model.sample_trajectory(initial_state, times, k)

    samples = jax.vmap(sample_fn)(keys)
    return samples


def plot_predictions_with_uncertainty(
    times: np.ndarray,
    true_obs: np.ndarray,
    samples: np.ndarray,
    decoder,
    norm_params,
    title: str = "SDE Predictions with Uncertainty"
):
    """Plot predictions with mean and 95% confidence intervals.

    Args:
        times: Time points
        true_obs: True observations
        samples: Latent samples (n_samples, seq_len, latent_dim)
        decoder: Decoder to map latent to observations
        norm_params: Normalization parameters for denormalization
        title: Plot title
    """
    import ldx

    # Decode all samples to observation space and denormalize
    n_samples = samples.shape[0]
    decoded_samples = []

    for i in range(n_samples):
        decoded = decoder(samples[i])
        decoded_denorm = ldx.utils.denormalize_data(np.array(decoded), norm_params)
        # Ensure correct shape
        while decoded_denorm.ndim > 2:
            decoded_denorm = decoded_denorm.squeeze()
        decoded_samples.append(decoded_denorm)

    decoded_samples = np.array(decoded_samples)  # (n_samples, seq_len, obs_dim)

    # Compute statistics
    mean_pred = np.mean(decoded_samples, axis=0)
    std_pred = np.std(decoded_samples, axis=0)
    lower_bound = np.percentile(decoded_samples, 2.5, axis=0)
    upper_bound = np.percentile(decoded_samples, 97.5, axis=0)

    obs_dim = true_obs.shape[1]

    fig, axes = plt.subplots(1, obs_dim, figsize=(6 * obs_dim, 4))
    if obs_dim == 1:
        axes = [axes]

    for dim in range(obs_dim):
        ax = axes[dim]

        # Plot individual samples (lightly)
        for i in range(min(10, n_samples)):  # Plot max 10 samples for clarity
            ax.plot(times, decoded_samples[i, :, dim],
                   alpha=0.1, color='blue', linewidth=0.5)

        # Plot mean prediction
        ax.plot(times, mean_pred[:, dim],
               color='blue', linewidth=2, label='Mean prediction')

        # Plot confidence interval
        ax.fill_between(times,
                        lower_bound[:, dim],
                        upper_bound[:, dim],
                        alpha=0.3, color='blue',
                        label='95% confidence interval')

        # Plot true observations
        ax.plot(times, true_obs[:, dim],
               'ro-', alpha=0.7, linewidth=1.5,
               markersize=4, label='True observations')

        ax.set_xlabel('Time')
        ax.set_ylabel(f'Dimension {dim}')
        ax.set_title(f'{title} - Dim {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main example function."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    print("=" * 60)
    print("Latent SDE Example: Stochastic Time Series Modeling")
    print("=" * 60)

    # Generate synthetic data - multiple spiral trajectories
    print("\n[1/6] Generating synthetic spiral dataset...")
    n_sequences = 10
    n_points = 50

    batch_times, batch_observations = ldx.datasets.generate_spiral_trajectory(
        n_points=n_points,
        n_sequences=n_sequences,
        n_dims=2,
        noise_std=0.15,  # Slightly more noise for SDE
        key=key
    )

    print(f"   ✓ Number of sequences: {n_sequences}")
    print(f"   ✓ Time points per sequence: {n_points}")
    print(f"   ✓ Observation shape: {batch_observations.shape}")

    # Normalize data
    normalized_obs, norm_params = ldx.utils.normalize_data(batch_observations, method="standard")

    # Create train/val split
    (train_obs, train_times), (val_obs, val_times) = ldx.utils.create_train_val_split(
        normalized_obs, batch_times, val_fraction=0.3
    )

    print(f"   ✓ Training sequences: {train_obs.shape[0]}")
    print(f"   ✓ Validation sequences: {val_obs.shape[0]}")

    # Model configuration
    latent_dim = 4
    obs_dim = batch_observations.shape[2]

    print(f"\n[2/6] Creating LatentSDE model...")
    print(f"   - Latent dimension: {latent_dim}")
    print(f"   - Observation dimension: {obs_dim}")

    # Create model
    key, model_key = jax.random.split(key)
    rngs = nnx.Rngs(model_key)

    model = ldx.models.create_latent_sde(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dims=(32, 32),
        posterior_hidden_dims=(32, 32),
        prior_hidden_dims=(32, 32),
        diffusion_hidden_dims=(16, 16),
        context_dim=32,
        encoder_type="mlp",
        decoder_type="mlp",
        use_linear_prior=False,
        rngs=rngs
    )

    print("   ✓ Model created with posterior/prior drift separation")
    print("   ✓ Observation encoder for context")
    print("   ✓ Augmented dynamics for KL computation")

    # Training setup
    print(f"\n[3/6] Setting up training...")

    learning_rate = 1e-3
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Create trainer
    trainer = ldx.inference.VariationalTrainer(
        model=model,
        optimizer=optimizer,
        reconstruction_loss_fn=ldx.inference.reconstruction_loss_mse,
        beta_schedule=ldx.inference.create_beta_schedule("linear", warmup_steps=50)
    )

    print(f"   ✓ Optimizer: Adam (lr={learning_rate})")
    print(f"   ✓ Beta schedule: Linear warmup (50 steps)")

    # Training loop
    n_epochs = 200
    print(f"\n[4/6] Training for {n_epochs} epochs...")

    train_losses = []
    val_losses = []
    kl_losses = []

    for epoch in range(n_epochs):
        key, train_key = jax.random.split(key)

        # Training step on a random training sequence
        train_idx = jax.random.randint(train_key, (), 0, train_obs.shape[0])
        model, train_metrics = trainer.train_step(
            model,
            train_obs[train_idx],
            train_times[train_idx],
            train_key,
            epoch
        )

        train_losses.append(float(train_metrics["total_loss"]))
        kl_losses.append(float(train_metrics["kl_loss"]))

        # Validation every 20 epochs
        if epoch % 20 == 0:
            key, val_key = jax.random.split(key)
            val_idx = jax.random.randint(val_key, (), 0, val_obs.shape[0])
            val_metrics = trainer.evaluate(
                model,
                val_obs[val_idx],
                val_times[val_idx],
                val_key
            )
            val_losses.append(float(val_metrics["total_loss"]))

            print(f"   Epoch {epoch:3d}: "
                  f"Train Loss = {train_metrics['total_loss']:.4f}, "
                  f"Val Loss = {val_metrics['total_loss']:.4f}, "
                  f"KL = {train_metrics['kl_loss']:.4f}")

    print("   ✓ Training completed!")

    # Generate predictions with uncertainty quantification
    print(f"\n[5/6] Generating predictions with uncertainty quantification...")

    key, pred_key = jax.random.split(key)

    # Use first validation sequence
    val_sequence = val_obs[0]
    val_times_sequence = val_times[0]

    # Get initial latent state from encoder
    mean, logvar = model.encode(val_sequence[None, ...], val_times_sequence)
    key, sample_key = jax.random.split(key)
    initial_state = model.reparameterize(mean[0], logvar[0], sample_key)

    # Generate multiple samples from the SDE
    n_samples = 100
    print(f"   - Generating {n_samples} samples from prior...")

    key, samples_key = jax.random.split(key)
    latent_samples = generate_multiple_samples(
        model, initial_state, val_times_sequence, samples_key, n_samples=n_samples
    )

    print(f"   ✓ Generated samples shape: {latent_samples.shape}")

    # Also generate posterior samples (conditioned on observations)
    print(f"   - Generating {n_samples} samples from posterior (conditioned)...")
    keys = jax.random.split(samples_key, n_samples)

    def sample_posterior_fn(k):
        return model.sample_posterior(
            initial_state, val_times_sequence, val_sequence, k, return_kl=False
        )

    posterior_samples = jax.vmap(sample_posterior_fn)(keys)
    print(f"   ✓ Generated posterior samples shape: {posterior_samples.shape}")

    # Denormalize true observations
    val_obs_denorm = ldx.utils.denormalize_data(val_sequence, norm_params)
    # Ensure correct shape
    if val_obs_denorm.ndim > 2:
        val_obs_denorm = val_obs_denorm[0]

    # Compute mean prediction and metrics
    decoded_samples = []
    for i in range(n_samples):
        decoded = model.decoder(latent_samples[i])
        decoded_denorm = ldx.utils.denormalize_data(np.array(decoded), norm_params)
        decoded_samples.append(decoded_denorm)

    decoded_samples = np.array(decoded_samples)
    mean_prediction = np.mean(decoded_samples, axis=0)

    metrics = ldx.utils.compute_all_metrics(mean_prediction, np.array(val_obs_denorm))

    print(f"\n   Evaluation Metrics (Mean Prediction):")
    for metric_name, value in metrics.items():
        print(f"      - {metric_name.upper()}: {value:.4f}")

    # Visualization
    print(f"\n[6/6] Creating visualizations...")

    # 1. Training curves
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(kl_losses, label='KL Divergence', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title('KL Divergence (from Augmented Dynamics)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sde_training_curves.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: sde_training_curves.png")
    plt.close()

    # 2. Prior predictions with uncertainty
    fig2 = plot_predictions_with_uncertainty(
        np.array(val_times_sequence),
        np.array(val_obs_denorm),
        np.array(latent_samples),
        model.decoder,
        norm_params,
        title="Prior (Generative) Predictions"
    )
    plt.savefig("sde_prior_predictions.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: sde_prior_predictions.png")
    plt.close()

    # 3. Posterior predictions with uncertainty
    fig3 = plot_predictions_with_uncertainty(
        np.array(val_times_sequence),
        np.array(val_obs_denorm),
        np.array(posterior_samples),
        model.decoder,
        norm_params,
        title="Posterior (Conditioned) Predictions"
    )
    plt.savefig("sde_posterior_predictions.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: sde_posterior_predictions.png")
    plt.close()

    # 4. Latent space visualization
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for dim in range(min(4, latent_dim)):
        ax = axes[dim]

        # Plot samples
        for i in range(min(20, n_samples)):
            ax.plot(val_times_sequence, latent_samples[i, :, dim],
                   alpha=0.2, color='blue', linewidth=0.5)

        # Plot mean
        mean_latent = np.mean(latent_samples, axis=0)
        ax.plot(val_times_sequence, mean_latent[:, dim],
               color='blue', linewidth=2, label='Prior mean')

        # Plot posterior mean
        mean_posterior = np.mean(posterior_samples, axis=0)
        ax.plot(val_times_sequence, mean_posterior[:, dim],
               color='red', linewidth=2, label='Posterior mean', linestyle='--')

        ax.set_xlabel('Time')
        ax.set_ylabel(f'Latent Dim {dim}')
        ax.set_title(f'Latent Dimension {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sde_latent_space.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: sde_latent_space.png")
    plt.close()

    # 5. Comparison plot showing the difference
    fig5, axes = plt.subplots(1, 2, figsize=(14, 5))

    for dim in range(obs_dim):
        ax = axes[dim]

        # Prior statistics
        decoded_prior = np.array([model.decoder(latent_samples[i])
                                  for i in range(n_samples)])
        decoded_prior_denorm = np.array([ldx.utils.denormalize_data(np.array(d), norm_params)
                                         for d in decoded_prior])
        # Remove extra dimensions if present
        while decoded_prior_denorm.ndim > 3:
            decoded_prior_denorm = decoded_prior_denorm.squeeze()

        mean_prior = np.mean(decoded_prior_denorm, axis=0).squeeze()
        if mean_prior.ndim == 2:
            mean_prior = mean_prior[:, dim]
        elif mean_prior.ndim == 1 and obs_dim == 1:
            mean_prior = mean_prior
        else:
            mean_prior = mean_prior[dim]

        std_prior = np.std(decoded_prior_denorm, axis=0).squeeze()
        if std_prior.ndim == 2:
            std_prior = std_prior[:, dim]
        elif std_prior.ndim == 1 and obs_dim == 1:
            std_prior = std_prior
        else:
            std_prior = std_prior[dim]

        # Posterior statistics
        decoded_posterior = np.array([model.decoder(posterior_samples[i])
                                      for i in range(n_samples)])
        decoded_posterior_denorm = np.array([ldx.utils.denormalize_data(np.array(d), norm_params)
                                             for d in decoded_posterior])
        # Remove extra dimensions if present
        while decoded_posterior_denorm.ndim > 3:
            decoded_posterior_denorm = decoded_posterior_denorm.squeeze()

        mean_posterior = np.mean(decoded_posterior_denorm, axis=0).squeeze()
        if mean_posterior.ndim == 2:
            mean_posterior = mean_posterior[:, dim]
        elif mean_posterior.ndim == 1 and obs_dim == 1:
            mean_posterior = mean_posterior
        else:
            mean_posterior = mean_posterior[dim]

        std_posterior = np.std(decoded_posterior_denorm, axis=0).squeeze()
        if std_posterior.ndim == 2:
            std_posterior = std_posterior[:, dim]
        elif std_posterior.ndim == 1 and obs_dim == 1:
            std_posterior = std_posterior
        else:
            std_posterior = std_posterior[dim]

        # Plot prior
        ax.plot(val_times_sequence, mean_prior,
               color='blue', linewidth=2, label='Prior (unconditional)', alpha=0.7)
        ax.fill_between(val_times_sequence,
                        mean_prior - 2*std_prior,
                        mean_prior + 2*std_prior,
                        alpha=0.2, color='blue')

        # Plot posterior
        ax.plot(val_times_sequence, mean_posterior,
               color='green', linewidth=2, label='Posterior (conditioned)', alpha=0.7)
        ax.fill_between(val_times_sequence,
                        mean_posterior - 2*std_posterior,
                        mean_posterior + 2*std_posterior,
                        alpha=0.2, color='green')

        # Plot true observations
        ax.plot(val_times_sequence, val_obs_denorm[:, dim],
               'ro-', alpha=0.7, linewidth=1.5,
               markersize=4, label='True observations')

        ax.set_xlabel('Time')
        ax.set_ylabel(f'Dimension {dim}')
        ax.set_title(f'Prior vs Posterior - Dim {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sde_prior_vs_posterior.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: sde_prior_vs_posterior.png")
    plt.close()

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - sde_training_curves.png: Training loss and KL divergence")
    print("  - sde_prior_predictions.png: Prior (generative) predictions with uncertainty")
    print("  - sde_posterior_predictions.png: Posterior (conditioned) predictions")
    print("  - sde_latent_space.png: Latent space dynamics")
    print("  - sde_prior_vs_posterior.png: Comparison of prior and posterior")
    print("\nKey insights:")
    print("  - Prior: Unconditional generation (more uncertain)")
    print("  - Posterior: Conditioned on observations (less uncertain)")
    print("  - Confidence intervals capture stochastic uncertainty")
    print("=" * 60)


if __name__ == "__main__":
    main()
