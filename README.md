# NTK $\kappa$-Heuristic Validation

Empirical validation that the Neural Tangent Kernel (NTK)-derived scalar $\kappa$ predicts pairwise token competition under exact policy gradient in Qwen-2.5 language models.

> Companion code for the BSc thesis *Inner-Optimal-Arm Dynamics in Policy Gradient Methods on Multi-armed Bandits*.

## Theory

Under NTK linearisation, the dynamics of two competing tokens $a, b$ during policy-gradient training are governed by a single scalar:

$$\kappa = \frac{\langle x_{bc}, x_{ba} \rangle}{\|x_{ba}\|^2}, \qquad x_{ba} = x_a - x_b, \quad x_{bc} = x_c - x_b$$

where $x_i = \nabla_w \theta_i$ is the gradient of token $i$'s logit with respect to trainable parameters, and $x_c$ is an effective complement gradient from all other tokens.

**Prediction rule.** Let $\rho_0 = \pi_0(a) / (\pi_0(a) + \pi_0(b))$ be the initial relative share. Then:
- $\kappa > \rho_0 \Rightarrow$ token $b$ overtakes $a$
- $\kappa < \rho_0 \Rightarrow$ token $a$ overtakes $b$

**Trajectory** (SGD only):

$$\delta_{t+1} = \delta_t + \eta V_t(1 - V_t) \|x_{ba}\|^2 (\sigma(\delta_t) - \kappa)$$

## Experimental Setup

| | |
|---|---|
| **Models** | Qwen/Qwen2.5-0.5B (L23), 1.5B (L27), 3B (L35), 0.5B |
| **Prompts** | "The movie was", "The food tasted", "The weather was" |
| **Tokens** | Top-100 single-word candidates per prompt |
| **Loss** | Exact expected policy gradient: $-\sum_i \pi(i)_{\text{sg}} \log \pi(i)$ |
| **Optimisers** | SGD (primary), Adam (comparison) |
| **LR sweep** | $10^{-4}, 5 \times 10^{-4}, 10^{-3}, 5 \times 10^{-3}, 10^{-2}$ |
| **Training** | 200 steps per pair, early stop at $\rho > 0.95$ or $\rho < 0.05$ |

Weights are reset to pretrained values after each pair. Gram matrix is computed once per (model, layer, prompt) and cached.

## Repository Structure

```
├── model.py              # Load Qwen, freeze layers, cache/reset weights
├── grads.py              # Per-token gradient extraction & Gram matrix computation
├── kappa.py              # Vectorised (ρ₀, κ) computation from Gram matrix
├── exact_pg.py           # Exact expected policy gradient loop
├── run_validation.py     # Full pipeline: Gram → pair selection → PG sweep → save
├── plots.py              # Shared plotting utilities
├── results/              # Pre-computed Gram matrices & validation results
│   ├── grads_*.pt        # Cached Gram data per (model, layer, prompt)
│   └── validation_*.pt   # Per-pair training logs with full diagnostics
└── pyproject.toml
```

## Installation

Requires Python ≥ 3.11.

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

### Reproducing all results

The full sweep was run in four parallel tmux sessions:

```bash
# Session 1: 0.5B (L23)
uv run python run_validation.py --model Qwen/Qwen2.5-0.5B --layer_id 23 --boundary_thresh 0.1 --lr 1e-4 5e-4 1e-3 5e-3 1e-2 --prompts movie food weather && \
uv run python run_validation.py --model Qwen/Qwen2.5-0.5B --layer_id 23 --boundary_thresh 0.1 --optimizer adam --lr 1e-4 5e-4 1e-3 --prompts movie food weather && \
uv run python run_validation.py --model Qwen/Qwen2.5-0.5B --layer_id 23 --pair_selection dist --n_pairs 100 --n_bins 10 --lr 1e-3 --prompts movie food weather

# Session 2: 1.5B (L27)
uv run python run_validation.py --model Qwen/Qwen2.5-1.5B --layer_id 27 --boundary_thresh 0.1 --lr 1e-4 5e-4 1e-3 5e-3 1e-2 --prompts movie food weather && \
uv run python run_validation.py --model Qwen/Qwen2.5-1.5B --layer_id 27 --boundary_thresh 0.1 --optimizer adam --lr 1e-4 5e-4 1e-3 --prompts movie food weather && \
uv run python run_validation.py --model Qwen/Qwen2.5-1.5B --layer_id 27 --pair_selection dist --n_pairs 100 --n_bins 10 --lr 1e-3 --prompts movie food weather

# Session 3: 3B (L35)
uv run python run_validation.py --model Qwen/Qwen2.5-3B --layer_id 35 --boundary_thresh 0.1 --lr 1e-4 5e-4 1e-3 5e-3 1e-2 --prompts movie food weather && \
uv run python run_validation.py --model Qwen/Qwen2.5-3B --layer_id 35 --boundary_thresh 0.1 --optimizer adam --lr 1e-4 5e-4 1e-3 --prompts movie food weather && \
uv run python run_validation.py --model Qwen/Qwen2.5-3B --layer_id 35 --pair_selection dist --n_pairs 100 --n_bins 10 --lr 1e-3 --prompts movie food weather

# Session 4: 0.5B full model (no --layer_id; all layers unfrozen, SGD only)
uv run python run_validation.py --model Qwen/Qwen2.5-0.5B --boundary_thresh 0.1 --lr 1e-4 5e-4 1e-3 5e-3 1e-2 --prompts movie food weather
```

## Results Format

Each `validation_*.pt` file contains:

```python
{
    "config": { ... },          # Full hyperparameters
    "accuracy": float,          # Overall κ-prediction accuracy
    "pairs": [{
        "a": str, "b": str,     # Token pair
        "rho": float,           # Initial ρ₀
        "kappa": float,         # NTK-predicted κ
        "rho_final": float,     # ρ after training
        "correct": bool,        # Did κ predict correctly?
        "log": {                # Per-step training log
            "step", "rho", "logit_rho", "theta_a", "theta_b",
            "a_mass", "b_mass", "loss", "grad_norm",
        },
        "gram_drift": float,    # Relative Frobenius norm change of 2×2 sub-Gram
        "param_delta": float,   # ‖w_T − w_0‖
    }, ...]
}
```

## License

MIT
