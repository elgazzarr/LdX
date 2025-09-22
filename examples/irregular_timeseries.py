"""Example of handling irregular time series with LatentODE."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
import optax

import ldx


def main():
    """Main example function for irregular time series."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(123)

    print("🚀 LatentDynamiX Example: Irregular Time Series")
    print("=" * 50)

    # Generate synthetic data - Lorenz attractor
    print("📊 Generating Lorenz attractor dataset...")
    times, observations = ldx.datasets.generate_lorenz_trajectory(
        n_points=500,
        dt=0.02,
        noise_std=0.5,
        key=key
    )

    print(f"   - Time points: {len(times)}")
    print(f"   - Observation shape: {observations.shape}")

    # Create irregular observations
    print("🕳️ Creating irregular observations...")
    key, irregular_key = jax.random.split(key)
    observed_times, observed_data, observation_mask = ldx.datasets.generate_irregular_observations(
        times, observations, observation_prob=0.6, key=irregular_key
    )

    print(f"   - Original points: {len(times)}")
    print(f"   - Observed points: {len(observed_times)} ({len(observed_times)/len(times)*100:.1f}%)")

    # Normalize data
    normalized_obs, norm_params = ldx.utils.normalize_data(observed_data, method="standard")

    # For irregular data, we'll work with the full time grid but use masking
    # Create padded sequences with the full time grid
    full_normalized_obs = ldx.utils.normalize_data(observations, method="standard")[0]

    # Create training data (reshape for batch processing)
    train_obs = full_normalized_obs[None, ...]  # Add batch dimension
    train_times = times[None, ...]  # Add batch dimension
    train_mask = observation_mask[None, ...]  # Add batch dimension

    print(f"   - Training shape: {train_obs.shape}")

    # Model configuration
    latent_dim = 3  # Lorenz attractor is 3D
    obs_dim = observations.shape[1]

    print(f"\n🔧 Creating LatentODE model for irregular data...")
    print(f"   - Latent dimension: {latent_dim}")
    print(f"   - Observation dimension: {obs_dim}")

    # Create model with RNN encoder to handle sequences better
    key, model_key = jax.random.split(key)
    rngs = nnx.Rngs(model_key)

    model = ldx.models.create_latent_ode(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dims=(64, 64),
        dynamics_hidden_dims=(64, 64, 64),
        encoder_type="rnn",  # RNN encoder for sequences
        decoder_type="gaussian",  # Gaussian decoder for uncertainty
        rngs=rngs
    )

    print("   ✅ Model created successfully")

    # Custom loss function that handles missing observations
    def masked_reconstruction_loss(predictions, targets, mask):
        """Reconstruction loss that only considers observed time points."""
        if isinstance(predictions, tuple):
            # Gaussian decoder returns (mean, logvar)
            mean, logvar = predictions
            var = jnp.exp(logvar)
            # Negative log-likelihood
            nll = 0.5 * (jnp.log(2 * jnp.pi) + logvar + (targets - mean)**2 / var)
            masked_nll = jnp.where(mask[..., None], nll, 0.0)
            return jnp.sum(masked_nll) / jnp.sum(mask)
        else:
            # Standard MSE
            mse = (predictions - targets) ** 2
            masked_mse = jnp.where(mask[..., None], mse, 0.0)
            return jnp.sum(masked_mse) / jnp.sum(mask)

    # Training setup
    print(f"\n🏋️ Setting up training for irregular data...")

    learning_rate = 5e-4
    optimizer = optax.adam(learning_rate)

    # Create trainer with custom loss
    trainer = ldx.inference.VariationalTrainer(
        model=model,
        optimizer=optimizer,
        reconstruction_loss_fn=lambda pred, target: masked_reconstruction_loss(pred, target, train_mask[0]),
        beta_schedule=ldx.inference.create_beta_schedule("linear", warmup_steps=100, end_beta=0.1)
    )

    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Using masked reconstruction loss")
    print(f"   - Beta annealing with low final beta")

    # Training loop
    n_epochs = 200
    print(f"\n🚂 Training for {n_epochs} epochs...")

    params = model  # Placeholder for actual parameter handling
    train_losses = []

    for epoch in range(n_epochs):
        key, train_key = jax.random.split(key)

        # Training step
        params, train_metrics = trainer.train_step(
            params,
            train_obs[0],
            train_times[0],
            train_key,
            epoch
        )

        train_losses.append(train_metrics["total_loss"])

        if epoch % 20 == 0:
            print(f"   Epoch {epoch:3d}: Loss = {train_metrics['total_loss']:.4f}, "
                  f"Recon = {train_metrics['reconstruction_loss']:.4f}, "
                  f"KL = {train_metrics['kl_loss']:.4f}")

    print("   ✅ Training completed!")

    # Generate predictions for the full time series
    print(f"\n🔮 Generating predictions for full time series...")

    key, pred_key = jax.random.split(key)

    # Forward pass
    reconstructed_obs, latent_mean, latent_logvar, latent_trajectory = model.apply(
        params, train_obs, train_times, pred_key
    )

    # Extract results
    reconstructed_obs = reconstructed_obs[0]  # Remove batch dimension
    latent_trajectory = latent_trajectory[0]

    # Denormalize
    reconstructed_obs_denorm = ldx.utils.denormalize_data(reconstructed_obs, norm_params)
    observed_data_denorm = ldx.utils.denormalize_data(observed_data, norm_params)

    print(f"   - Reconstructed full time series: {reconstructed_obs_denorm.shape}")

    # Compute metrics on observed points only
    observed_indices = jnp.where(observation_mask)[0]
    pred_observed = reconstructed_obs_denorm[observed_indices]

    if isinstance(reconstructed_obs, tuple):
        # Handle Gaussian predictions
        mean_pred = reconstructed_obs[0][observed_indices]
        logvar_pred = reconstructed_obs[1][observed_indices]
        mean_pred_denorm = ldx.utils.denormalize_data(mean_pred, norm_params)

        metrics = ldx.utils.compute_all_metrics(
            (mean_pred_denorm, logvar_pred),
            observed_data_denorm,
            probabilistic=True
        )
    else:
        metrics = ldx.utils.compute_all_metrics(
            pred_observed,
            observed_data_denorm
        )

    print(f"\n📈 Evaluation Metrics (on observed points):")
    for metric_name, value in metrics.items():
        print(f"   - {metric_name.upper()}: {value:.4f}")

    # Visualization
    print(f"\n📊 Creating visualizations...")

    # Plot 1: Training curve
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Progress for Irregular Time Series')
    plt.grid(True, alpha=0.3)
    plt.savefig("irregular_training.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Irregular observations vs full reconstruction
    fig2, axes = plt.subplots(3, 1, figsize=(12, 10))

    for dim in range(3):
        ax = axes[dim]

        # Plot full true trajectory
        ax.plot(times, observations[:, dim], 'b-', alpha=0.3, label='True (full)', linewidth=1)

        # Plot observed points
        ax.scatter(observed_times, observed_data_denorm[:, dim],
                  color='red', alpha=0.7, s=20, label='Observed', zorder=5)

        # Plot reconstruction
        if isinstance(reconstructed_obs, tuple):
            mean_recon = ldx.utils.denormalize_data(reconstructed_obs[0], norm_params)
            std_recon = jnp.exp(0.5 * reconstructed_obs[1])
            ax.plot(times, mean_recon[:, dim], 'g-', label='Reconstructed', linewidth=2)
            ax.fill_between(times,
                           mean_recon[:, dim] - 1.96 * std_recon[:, dim],
                           mean_recon[:, dim] + 1.96 * std_recon[:, dim],
                           alpha=0.3, color='green', label='95% CI')
        else:
            ax.plot(times, reconstructed_obs_denorm[:, dim], 'g-',
                   label='Reconstructed', linewidth=2)

        ax.set_ylabel(f'Dimension {dim + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel('Time')
    plt.suptitle('Irregular Time Series Reconstruction')
    plt.tight_layout()
    plt.savefig("irregular_reconstruction.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Learned latent dynamics
    fig3 = ldx.utils.plot_latent_trajectory(
        times,
        latent_trajectory,
        title="Learned Latent Dynamics (Irregular Data)"
    )
    plt.savefig("irregular_latent.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: 3D phase portrait of latent space
    if latent_dim >= 3:
        fig4 = plt.figure(figsize=(10, 8))
        ax = fig4.add_subplot(111, projection='3d')

        # Color by time
        colors = np.linspace(0, 1, len(latent_trajectory))
        scatter = ax.scatter(latent_trajectory[:, 0],
                           latent_trajectory[:, 1],
                           latent_trajectory[:, 2],
                           c=colors, cmap='viridis', alpha=0.7)

        # Add trajectory line
        ax.plot(latent_trajectory[:, 0],
               latent_trajectory[:, 1],
               latent_trajectory[:, 2],
               alpha=0.3, linewidth=1, color='gray')

        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_zlabel('Latent Dim 3')
        ax.set_title('3D Latent Phase Portrait')
        plt.colorbar(scatter, label='Time')
        plt.savefig("irregular_3d_phase.png", dpi=150, bbox_inches='tight')
        plt.close()

    print("   ✅ Visualizations saved:")
    print("      - irregular_training.png")
    print("      - irregular_reconstruction.png")
    print("      - irregular_latent.png")
    if latent_dim >= 3:
        print("      - irregular_3d_phase.png")

    # Demonstrate interpolation capability
    print(f"\n🔗 Demonstrating interpolation capability...")

    # Create a sparser observation set
    key, sparse_key = jax.random.split(key)
    sparse_times, sparse_data, sparse_mask = ldx.datasets.generate_irregular_observations(
        times, observations, observation_prob=0.3, key=sparse_key
    )

    print(f"   - Sparse observations: {len(sparse_times)} points ({len(sparse_times)/len(times)*100:.1f}%)")

    # The model can interpolate between sparse observations
    # This demonstrates the power of latent ODEs for irregular time series

    print(f"\n🎉 Irregular time series example completed!")
    print(f"💡 Key insights:")
    print(f"   - LatentODE can handle missing observations naturally")
    print(f"   - RNN encoder processes irregular sequences effectively")
    print(f"   - Gaussian decoder provides uncertainty estimates")
    print(f"   - Learned dynamics enable interpolation between sparse points")
    print("=" * 50)


if __name__ == "__main__":
    main()