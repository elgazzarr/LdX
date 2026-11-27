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

Train a Latent SDE on synthetic data via variational inference:

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

# Sample new trajectories from the prior (unconditional generation) or from the posterior (smoothing)
key, sample_key = jax.random.split(key)

# Sampling works on a single sequence with a 1-D time grid
seq_times, seq_obs = times[0], observations[0]  # (50,), (50, 2)

# Unconditional generation from the prior (sample initial state from standard normal)
initial_state = jax.random.normal(sample_key, (model.latent_dim,))
prior_trajectory = model.sample_trajectory(initial_state, seq_times, sample_key)
generated_observations = model.decoder(prior_trajectory)

# Posterior sampling (smoothing) - encode observations for initial state
mean, logvar = model.encode(seq_obs[None, ...], seq_times)
key, enc_key = jax.random.split(sample_key)
initial_state = model.reparameterize(mean[0], logvar[0], enc_key)
posterior_trajectory = model.sample_trajectory(initial_state, seq_times, enc_key)
posterior_observations = model.decoder(posterior_trajectory)
```

## Examples

- [Basic Latent ODE](examples/latent_ode.ipynb): Learn a latent ODE model on noisy observations of a 2D spiral.
- [Basic Latent SDE](examples/latent_sde.ipynb): Learn a latent SDE model on noisy observations of a 2D spiral.

## References

If you make use of the toolbox in your work, please cite the repository and/or this paper:

[ElGazzar, Ahmed, and Marcel van Gerven. Generative Modeling of Neural Dynamics via Latent Stochastic Differential Equations. December 2024. http://arxiv.org/abs/2412.12112.](http://arxiv.org/abs/2412.12112)

