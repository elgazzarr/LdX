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

    # Generate synthetic data - multiple Lorenz attractors
    print("📊 Generating Lorenz attractor dataset...")
    n_sequences = 50
    batch_times, batch_observations = ldx.datasets.generate_lorenz_trajectory(
        n_points=100,
        n_sequences=n_sequences,
        dt=0.02,
        noise_std=0.5,
        key=key
    )

    print(f"   - Number of sequences: {n_sequences}")
    print(f"   - Time points per sequence: {batch_times.shape[1]}")
    print(f"   - Batch observation shape: {batch_observations.shape}")

    # Create irregular observations for each sequence
    print("🕳️ Creating irregular observations...")
    key, irregular_key = jax.random.split(key)

    # Generate irregular observations for all sequences
    all_observed_times = []
    all_observed_data = []
    all_observation_masks = []

    for seq_idx in range(n_sequences):
        key, seq_key = jax.random.split(key)
        observed_times, observed_data, observation_mask = ldx.datasets.generate_irregular_observations(
            batch_times[seq_idx], batch_observations[seq_idx], observation_prob=0.6, key=seq_key
        )
        all_observed_times.append(observed_times)
        all_observed_data.append(observed_data)
        all_observation_masks.append(observation_mask)

    # For visualization and evaluation, use the first sequence
    times = batch_times[0]
    observations = batch_observations[0]
    observed_times = all_observed_times[0]
    observed_data = all_observed_data[0]
    observation_mask = all_observation_masks[0]

    print(f"   - Original points per sequence: {batch_times.shape[1]}")
    print(f"   - Example sequence - Observed points: {len(observed_times)} ({len(observed_times)/len(times)*100:.1f}%)")

    # Normalize data using the full batch dataset to get consistent normalization parameters
    # Reshape to (batch*time, obs_dim) for global normalization
    reshaped_data = batch_observations.reshape(-1, batch_observations.shape[-1])
    normalized_reshaped, norm_params = ldx.utils.normalize_data(reshaped_data, method="standard")
    batch_normalized_obs = normalized_reshaped.reshape(batch_observations.shape)

    # Apply the same normalization parameters to observed data (first sequence for evaluation)
    if norm_params["method"] == "standard":
        normalized_obs = (observed_data - norm_params["mean"]) / norm_params["std"]
    else:
        # Handle other normalization methods if needed
        normalized_obs = observed_data

    # Stack observation masks for all sequences
    batch_observation_masks = jnp.stack(all_observation_masks, axis=0)

    # Create train/validation split
    (train_obs, train_times), (val_obs, val_times) = ldx.utils.create_train_val_split(
        batch_normalized_obs, batch_times, val_fraction=0.3
    )

    # Split masks accordingly
    n_train = train_obs.shape[0]
    train_mask = batch_observation_masks[:n_train]
    val_mask = batch_observation_masks[n_train:]

    print(f"   - Training sequences: {train_obs.shape[0]}")
    print(f"   - Validation sequences: {val_obs.shape[0]}")
    print(f"   - Training shape: {train_obs.shape}")
    print(f"   - Validation shape: {val_obs.shape}")

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

    learning_rate = 1e-3
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Create trainer with custom loss for batch training
    def batch_masked_reconstruction_loss(predictions, targets):
        """Batch reconstruction loss that handles multiple sequences."""
        if isinstance(predictions, tuple):
            # Gaussian decoder returns (mean, logvar)
            mean, logvar = predictions
            var = jnp.exp(logvar)
            # Negative log-likelihood
            nll = 0.5 * (jnp.log(2 * jnp.pi) + logvar + (targets - mean)**2 / var)
            masked_nll = jnp.where(train_mask[..., None], nll, 0.0)
            return jnp.sum(masked_nll) / jnp.sum(train_mask)
        else:
            # Standard MSE
            mse = (predictions - targets) ** 2
            masked_mse = jnp.where(train_mask[..., None], mse, 0.0)
            return jnp.sum(masked_mse) / jnp.sum(train_mask)

    trainer = ldx.inference.VariationalTrainer(
        model=model,
        optimizer=optimizer,
        reconstruction_loss_fn=batch_masked_reconstruction_loss,
        beta_schedule=ldx.inference.create_beta_schedule("linear", warmup_steps=100, end_beta=0.1)
    )

    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Using masked reconstruction loss")
    print(f"   - Beta annealing with low final beta")

    # Training loop
    n_epochs = 200
    print(f"\n🚂 Training for {n_epochs} epochs...")

    train_losses = []

    val_losses = []

    for epoch in range(n_epochs):
        key, train_key = jax.random.split(key)

        # Efficient batch training on all training sequences
        model, train_metrics = trainer.train_step(
            model,
            train_obs,  # Full batch
            train_times,
            train_key,
            epoch
        )

        train_losses.append(train_metrics["total_loss"])

        # Validation every few epochs
        if epoch % 20 == 0:  # More frequent validation since it's now efficient
            # Evaluate on full validation set efficiently
            key, val_key = jax.random.split(key)
            val_metrics = trainer.evaluate(
                model,
                val_obs,  # Full validation batch
                val_times,
                val_key
            )
            val_losses.append(val_metrics["total_loss"])

            print(f"   Epoch {epoch:3d}: Train Loss = {train_metrics['total_loss']:.4f}, "
                  f"Val Loss = {val_metrics['total_loss']:.4f}, "
                  f"Train Recon = {train_metrics['reconstruction_loss']:.4f}, "
                  f"KL = {train_metrics['kl_loss']:.4f}")

    print("   ✅ Training completed!")

    # Final evaluation on validation set (already done efficiently during training)
    print(f"\n🔮 Final evaluation on validation set...")

    key, eval_key = jax.random.split(key)
    final_val_metrics = trainer.evaluate(
        model,
        val_obs,  # Full validation batch - efficient with JIT + vmap
        val_times,
        eval_key
    )

    print(f"   - Efficiently evaluated on {val_obs.shape[0]} validation sequences using JIT + vmap")
    print(f"\n📈 Final Validation Metrics:")
    for metric_name, value in final_val_metrics.items():
        print(f"   - {metric_name.upper()}: {value:.4f}")

    # For visualization, use the first validation sequence
    key, vis_key = jax.random.split(key)
    vis_obs = val_obs[0:1]
    vis_times = val_times[0:1]
    vis_mask = val_mask[0]

    reconstructed_obs, latent_mean, latent_logvar, latent_trajectory = model(
        vis_obs, vis_times, vis_key
    )

    # Extract results for visualization
    reconstructed_obs = reconstructed_obs[0]  # Remove batch dimension
    latent_trajectory = latent_trajectory[0]
    reconstructed_obs_denorm = ldx.utils.denormalize_data(reconstructed_obs, norm_params)

    # Use original data from first validation sequence for visualization
    times = val_times[0]
    observations = ldx.utils.denormalize_data(val_obs[0], norm_params)
    observation_mask = vis_mask
    observed_indices = jnp.where(observation_mask)[0]
    observed_times = times[observed_indices]
    observed_data = observations[observed_indices]

    print(f"   - Using validation sequence 0 for visualization")

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
        ax.scatter(observed_times, observed_data[:, dim],
                  color='red', alpha=0.7, s=20, label='Observed', zorder=5)

        # Plot reconstruction
        if isinstance(reconstructed_obs, tuple):
            mean_recon = ldx.utils.denormalize_data(reconstructed_obs[0], norm_params)
            std_recon = jnp.exp(0.5 * reconstructed_obs[1])
            ax.plot(times, mean_recon[0, :, dim], 'g-', label='Reconstructed', linewidth=2)
            ax.fill_between(times,
                           mean_recon[0, :, dim] - 1.96 * std_recon[0, :, dim],
                           mean_recon[0, :, dim] + 1.96 * std_recon[0, :, dim],
                           alpha=0.3, color='green', label='95% CI')
        else:
            ax.plot(times, reconstructed_obs_denorm[0, :, dim], 'g-',
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