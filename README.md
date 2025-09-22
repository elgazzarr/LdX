# LatentDynamiX (LdX)

> **LdX** — A library for modeling **partially observed dynamical systems** with **latent ODEs and SDEs**.

---

### ✨ Overview
LatentDynamiX (LdX) provides a unified framework for learning and simulating **latent dynamical systems**.  
It focuses on **partially observed settings**, where noisy, irregular, or incomplete data must be mapped to hidden trajectories.  

- 🔹 **Latent ODEs / SDEs** — Neural ODEs, Neural SDEs, and extensions  
- 🔹 **Flexible inference** — Variational inference, filtering, smoothing  
- 🔹 **Toolbox style** — Plug-and-play models, utilities, and benchmarks  
- 🔹 **Research-ready** — Reproducible experiments, clean API  

---

### 🚀 Getting Started
```bash
pip install ldx
```


### 🛠️ Example Usage

```python

import ldx
# Example: define a latent ODE model
model = ldx.models.LatentODE(hidden_dim=32)
trajectory = model.simulate(t_span=[0, 10])
print(trajectory)
```
