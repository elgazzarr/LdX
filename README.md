# LatentDynamiX (LdX)

> **LdX** — A library for modeling **partially observed dynamical systems** with **latent ODEs and SDEs**.

LatentDynamiX (LdX) provides a unified framework for learning and simulating **latent dynamical systems**.
It focuses on **partially observed settings**, where noisy, irregular, or incomplete data must be mapped to hidden trajectories.


## Installation

```bash
# Clone the repository
git clone https://github.com/elgazzarr/ldx.git
cd ldx

# Install with uv (recommended)
uv sync

# Or install with pip in editable mode
pip install -e .
```

## Quickstart

Train a Latent SDE on synthetic data using variational inference:

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import ldx

# Generate synthetic spiral data
key = jax.random.PRNGKey(42)
times, observations = ldx.datasets.generate_spiral_trajectory(
    n_points=50, n_sequences=8, noise_std=0.1, key=key
)

# Normalize data
observations, norm_params = ldx.utils.normalize_data(observations)

# Create a Latent SDE model
model = ldx.models.create_latent_sde(
    latent_dim=4,
    obs_dim=2,
    hidden_dims=(32, 32),
    rngs=nnx.Rngs(key)
)

# Setup optimizer and trainer
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
trainer = ldx.inference.VariationalTrainer(
    model=model,
    optimizer=optimizer,
    reconstruction_loss_fn=ldx.inference.reconstruction_loss_mse,
    beta_schedule=ldx.inference.create_beta_schedule("linear", warmup_steps=50)
)

# Train the model
for epoch in range(100):
    key, train_key = jax.random.split(key)
    model, metrics = trainer.train_step(model, observations, times, train_key, epoch)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {metrics['total_loss']:.4f}")

# Sample new trajectories from the prior
key, sample_key = jax.random.split(key)
mean, logvar = model.encode(observations[0:1], times[0])
z0 = model.reparameterize(mean[0], logvar[0], sample_key)
trajectory = model.sample_trajectory(z0, times[0], sample_key)
decoded = model.decode(trajectory, times[0])
```

## Examples

- [Basic Latent ODE](examples/basic_latent_ode.py): Learn a latent ODE model on noisy observations of a 2D spiral.
- [Basic Latent SDE](examples/basic_latent_sde.py): Learn a latent SDE model on noisy observations of a 2D spiral.
- [Controlled Latent SDE](examples/controlled_latent_sde.py): Learn a controlled latent SDE model on noisy observations of a Ornstein-Uhlenbeck process with control inputs.
