"""
Shared scoring model for theta: linear (default, matches the original f_theta exactly)
or an opt-in small MLP (one hidden layer, ReLU).

theta is always a single flat 1-D tensor/array; ModelConfig says how to interpret it.
The bilevel solver's CG/Hessian-vector-product and implicit-differentiation machinery
(draft_model/bilevel_al.py) treats theta as an opaque flat vector and needs no changes
for either model type -- only the functions that turn (theta, Xa) into logits do.
"""
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_torch(x: ArrayLike, dtype=torch.float32, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


@dataclass
class ModelConfig:
    """model_type: 'linear' (f(x,a) = thetaT[x;a]) or 'mlp' (1 hidden layer, ReLU,
    hidden_dim units)."""
    model_type: str = "linear"
    hidden_dim: int = 8

    def param_count(self, d: int) -> int:
        """Flat parameter count for input width d+1 (features + sensitive attribute)."""
        if self.model_type == "linear":
            return d + 1
        if self.model_type == "mlp":
            h = self.hidden_dim
            return h * (d + 1) + h + h + 1  # W1, b1, W2, b2
        raise ValueError(f"Unknown model_type {self.model_type!r}")


def init_params(cfg: ModelConfig, d: int, seed: int) -> np.ndarray:
    """Flat initial parameter vector of length cfg.param_count(d)."""
    if cfg.model_type == "linear":
        # Matches the original zeta = np.zeros(d_plus_1) exactly.
        return np.zeros(d + 1, dtype=np.float64)
    if cfg.model_type == "mlp":
        # Zero-init would leave all hidden units symmetric (dead gradients), so reuse
        # torch.nn.Linear's own default (Kaiming-uniform) init instead of hand-rolling it.
        h = cfg.hidden_dim
        torch.manual_seed(seed)
        lin1 = torch.nn.Linear(d + 1, h)
        lin2 = torch.nn.Linear(h, 1)
        with torch.no_grad():
            flat = torch.cat([
                lin1.weight.reshape(-1), lin1.bias.reshape(-1),
                lin2.weight.reshape(-1), lin2.bias.reshape(-1),
            ])
        return flat.numpy().astype(np.float64)
    raise ValueError(f"Unknown model_type {cfg.model_type!r}")


def score(cfg: ModelConfig, theta: ArrayLike, Xa: ArrayLike) -> torch.Tensor:
    """f(x,a) for a batch. Xa is (n, d+1) [features ++ sensitive attr]. Returns (n,) logits."""
    theta_t = _to_torch(theta)
    Xa_t = _to_torch(Xa, device=theta_t.device)
    if cfg.model_type == "linear":
        return (Xa_t @ theta_t).squeeze(-1)
    if cfg.model_type == "mlp":
        d_plus_1 = Xa_t.shape[-1]
        h = cfg.hidden_dim
        i = 0
        w1 = theta_t[i:i + h * d_plus_1].reshape(h, d_plus_1); i += h * d_plus_1
        b1 = theta_t[i:i + h]; i += h
        w2 = theta_t[i:i + h].reshape(1, h); i += h
        b2 = theta_t[i:i + 1]
        hidden = torch.relu(Xa_t @ w1.T + b1)
        logits = hidden @ w2.T + b2
        return logits.squeeze(-1)
    raise ValueError(f"Unknown model_type {cfg.model_type!r}")
