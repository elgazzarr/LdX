"""Basic example of using LatentODE for time series modeling."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
import optax

import ldx


def main():
    """Main example function."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    print("🚀 LatentDynamiX Basic Example: Latent ODE")
    print("=" * 50)

    # Generate synthetic data - multiple spiral trajectories
    print("📊 Generating synthetic spiral dataset...")
    n_sequences = 10
    n_points = 50

    all_observations = []
    all_times = []

    for i in range(n_sequences):
        key, seq_key = jax.random.split(key)
        times, observations = ldx.datasets.generate_spiral_trajectory(
            n_points=n_points,
            n_dims=2,
            noise_std=0.1,
            key=seq_key
        )
        all_observations.append(observations)
        all_times.append(times)

    # Stack into batch format
    batch_observations = jnp.stack(all_observations, axis=0)  # (n_sequences, n_points, 2)
    batch_times = jnp.stack(all_times, axis=0)  # (n_sequences, n_points)

    print(f"   - Number of sequences: {n_sequences}")
    print(f"   - Time points per sequence: {n_points}")
    print(f"   - Batch observation shape: {batch_observations.shape}")

    # Normalize data
    normalized_obs, norm_params = ldx.utils.normalize_data(batch_observations, method="standard")

    # Create train/val split
    (train_obs, train_times), (val_obs, val_times) = ldx.utils.create_train_val_split(
        normalized_obs, batch_times, val_fraction=0.3
    )

    print(f"   - Training sequences: {train_obs.shape[0]}")
    print(f"   - Validation sequences: {val_obs.shape[0]}")

    # Model configuration
    latent_dim = 4
    obs_dim = observations.shape[1]

    print(f"\n🔧 Creating LatentODE model...")
    print(f"   - Latent dimension: {latent_dim}")
    print(f"   - Observation dimension: {obs_dim}")

    # Create model
    key, model_key = jax.random.split(key)
    rngs = nnx.Rngs(model_key)

    model = ldx.models.create_latent_ode(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dims=(32, 32),
        dynamics_hidden_dims=(32, 32),
        encoder_type="mlp",
        decoder_type="mlp",
        rngs=rngs
    )

    print("   ✅ Model created successfully")

    # Training setup
    print(f"\n🏋️ Setting up training...")

    learning_rate = 1e-3
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Create trainer
    trainer = ldx.inference.VariationalTrainer(
        model=model,
        optimizer=optimizer,
        reconstruction_loss_fn=ldx.inference.reconstruction_loss_mse,
        beta_schedule=ldx.inference.create_beta_schedule("linear", warmup_steps=50)
    )

    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Optimizer: Adam")
    print(f"   - Beta schedule: Linear warmup")

    # Training loop
    n_epochs = 50  # Fewer epochs for this basic example
    print(f"\n🚂 Training for {n_epochs} epochs...")

    train_losses = []
    val_losses = []

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

        train_losses.append(train_metrics["total_loss"])

        # Validation every 10 epochs
        if epoch % 10 == 0:
            key, val_key = jax.random.split(key)
            val_idx = jax.random.randint(val_key, (), 0, val_obs.shape[0])
            val_metrics = trainer.evaluate(
                model,
                val_obs[val_idx],
                val_times[val_idx],
                val_key
            )
            val_losses.append(val_metrics["total_loss"])

            print(f"   Epoch {epoch:3d}: Train Loss = {train_metrics['total_loss']:.4f}, "
                  f"Val Loss = {val_metrics['total_loss']:.4f}")

    print("   ✅ Training completed!")

    # Generate predictions
    print(f"\n🔮 Generating predictions...")

    key, pred_key = jax.random.split(key)

    # Use first validation sequence for prediction
    val_sequence = val_obs[0]
    val_times_sequence = val_times[0]

    # Forward pass to get reconstructions and latent trajectory (add batch dimension)
    reconstructed_obs, latent_mean, latent_logvar, latent_trajectory = model(
        val_sequence[None, ...], val_times_sequence[None, ...], pred_key
    )

    # Remove batch dimension from results
    reconstructed_obs = reconstructed_obs[0]
    latent_trajectory = latent_trajectory[0]

    # Denormalize predictions
    reconstructed_obs_denorm = ldx.utils.denormalize_data(reconstructed_obs, norm_params)
    val_obs_denorm = ldx.utils.denormalize_data(val_sequence, norm_params)

    # Fix shapes - remove batch dimensions if present
    while reconstructed_obs_denorm.ndim > 2:
        reconstructed_obs_denorm = reconstructed_obs_denorm[0]
    while val_obs_denorm.ndim > 2:
        val_obs_denorm = val_obs_denorm[0]

    print(f"   - Reconstruction shape: {reconstructed_obs_denorm.shape}")
    print(f"   - Latent trajectory shape: {latent_trajectory.shape}")

    # Compute metrics
    metrics = ldx.utils.compute_all_metrics(
        reconstructed_obs_denorm,
        val_obs_denorm
    )

    print(f"\n📈 Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"   - {metric_name.upper()}: {value:.4f}")

    # Visualization
    print(f"\n📊 Creating visualizations...")

    # Plot training curves (only training for simplicity)
    fig1 = ldx.utils.plot_training_curves(
        train_losses
    )
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot trajectory comparison
    print(f"Debug shapes: times={val_times_sequence.shape}, obs={val_obs_denorm.shape}, pred={reconstructed_obs_denorm.shape}")
    fig2 = ldx.utils.plot_trajectory(
        np.array(val_times_sequence),
        np.array(val_obs_denorm),
        np.array(reconstructed_obs_denorm),
        dims=[0, 1]  # Plot both dimensions
    )
    plt.savefig("trajectory_reconstruction.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot latent trajectory
    fig3 = ldx.utils.plot_latent_trajectory(
        val_times_sequence,
        latent_trajectory,
        title="Learned Latent Dynamics"
    )
    plt.savefig("latent_trajectory.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot phase portrait if latent_dim >= 2
    if latent_dim >= 2:
        fig4 = ldx.utils.plot_phase_portrait(
            latent_trajectory,
            dims=(0, 1),
            title="Latent Phase Portrait"
        )
        plt.savefig("phase_portrait.png", dpi=150, bbox_inches='tight')
        plt.close()

    print("   ✅ Visualizations saved:")
    print("      - training_curves.png")
    print("      - trajectory_reconstruction.png")
    print("      - latent_trajectory.png")
    if latent_dim >= 2:
        print("      - phase_portrait.png")

    print(f"\n🎉 Example completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()