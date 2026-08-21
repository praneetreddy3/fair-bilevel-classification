"""
Loss functions for the bilevel fairness pipeline.

Updated to use the smooth TPR-gap EO surrogate from the new draft:
g_EO = |TPR_1 - TPR_0| with sigmoid smoothing around threshold tau.

Model: f_θ(x,a) = θᵀ[x; a], linear classifier with input dim d+1.
Labels y ∈ {0,1}, sensitive attribute a ∈ {0,1}.
"""
import numpy as np
import torch

from . import model as _model
from .model import ModelConfig


def _to_torch(x, dtype=torch.float32, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def pack_xa(X: np.ndarray, A: np.ndarray) -> torch.Tensor:
    """Concatenate features and sensitive attribute: [x; a] → (n, d+1)."""
    A_col = np.asarray(A).reshape(-1, 1)
    return torch.tensor(np.hstack([np.asarray(X), A_col]), dtype=torch.float32)


def f_theta(theta: torch.Tensor, Xa: torch.Tensor, cfg: ModelConfig = None) -> torch.Tensor:
    """f_θ(x,a): linear (θᵀ[x;a]) by default, or the model `cfg` selects (see model.py)."""
    return _model.score(cfg or ModelConfig(), theta, Xa)


def logistic_loss_per_sample(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Binary logistic loss: ℓ(y, f) = log(1 + exp(-(2y-1)·f))."""
    margin = (2 * y - 1) * logits
    return torch.log(1 + torch.exp(-margin.clamp(min=-50)))


def L_base(theta: torch.Tensor, X: np.ndarray, A: np.ndarray, Y: np.ndarray,
           zeta: torch.Tensor, lambda_theta: float, d_plus_1: int,
           cfg: ModelConfig = None) -> torch.Tensor:
    """Eq. 2: (1/|D|) Σ ℓ(y, f_θ(x,a)) + λ_θ/(2(d+1)²) ||θ−ζ||²."""
    Xa = pack_xa(X, A).to(theta.device)
    y = _to_torch(Y, device=theta.device)
    logits = f_theta(theta, Xa, cfg)
    loss_data = logistic_loss_per_sample(y, logits).mean()
    reg = (lambda_theta / 2.0) * torch.sum((theta - zeta) ** 2)
    return loss_data + reg


def L_universum(theta: torch.Tensor, X: np.ndarray, A: np.ndarray,
                lambda_U: float, Delta_s: float, cfg: ModelConfig = None) -> torch.Tensor:
    """Eq. 3: Universum loss — encourages θ to classify U points as positive (y=1)."""
    if Delta_s <= 0 or len(X) == 0:
        return torch.tensor(0.0, device=theta.device)
    Xa = pack_xa(X, A).to(theta.device)
    logits = f_theta(theta, Xa, cfg)
    y_one = torch.ones(Xa.shape[0], device=theta.device)
    loss_u = logistic_loss_per_sample(y_one, logits).mean()
    return lambda_U * loss_u


def compute_tpr_s(
    theta: torch.Tensor,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    s_group: int,
    tau: float = 0.0,
    alpha: float = 10.0,
    cfg: ModelConfig = None,
) -> torch.Tensor:
    """
    Smooth TPR estimate for group s on Y=1 slice:
    TPR_s ≈ mean( sigmoid(alpha * (f_theta(x,s) - tau)) | Y=1, A=s ).
    """
    y = np.asarray(Y)
    a = np.asarray(A)
    mask = (y == 1) & (a == s_group)
    if not np.any(mask):
        return torch.tensor(0.0, dtype=torch.float32, device=theta.device)
    Xa = pack_xa(np.asarray(X)[mask], a[mask]).to(theta.device)
    logits = f_theta(theta, Xa, cfg)
    smooth_pred = torch.sigmoid(alpha * (logits - tau))
    return smooth_pred.mean()


def compute_tpr_gap_surrogate(
    theta: torch.Tensor,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    tau: float = 0.0,
    alpha: float = 10.0,
    cfg: ModelConfig = None,
) -> torch.Tensor:
    """
    Section 4 surrogate: smooth absolute TPR gap.

    Uses differentiable approximation:
      g = sqrt((TPR_1 - TPR_0)^2 + eps)
    """
    tpr0 = compute_tpr_s(theta, X, A, Y, s_group=0, tau=tau, alpha=alpha, cfg=cfg)
    tpr1 = compute_tpr_s(theta, X, A, Y, s_group=1, tau=tau, alpha=alpha, cfg=cfg)
    diff = tpr1 - tpr0
    return torch.sqrt(diff * diff + 1e-12)


def g_EO(
    theta: torch.Tensor,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    tau: float = 0.0,
    alpha: float = 10.0,
    cfg: ModelConfig = None,
) -> torch.Tensor:
    """EO surrogate wrapper (TPR-gap, smooth)."""
    if len(Y) == 0 or not np.any(np.asarray(Y) == 1):
        return torch.tensor(0.0, dtype=torch.float32, device=theta.device)
    return compute_tpr_gap_surrogate(theta, X, A, Y, tau=tau, alpha=alpha, cfg=cfg)


def Lin(theta: torch.Tensor,
        Ds_X: np.ndarray, Ds_A: np.ndarray, Ds_Y: np.ndarray,
        U_X: np.ndarray, U_A: np.ndarray,
        zeta: torch.Tensor, lambda_theta_in: float, lambda_U: float,
        d_plus_1: int, Delta_s: int, cfg: ModelConfig = None) -> torch.Tensor:
    """Inner objective: Lin(θ) = L_base(θ; Dˢ, ζ) + L_universum(θ; U)."""
    L1 = L_base(theta, Ds_X, Ds_A, Ds_Y, zeta, lambda_theta_in, d_plus_1, cfg)
    L2 = L_universum(theta, U_X, U_A, lambda_U, float(Delta_s), cfg)
    return L1 + L2


def L_out(theta: torch.Tensor, B_X: np.ndarray, B_A: np.ndarray, B_Y: np.ndarray,
          zeta_out: torch.Tensor, lambda_theta_out: float, d_plus_1: int,
          cfg: ModelConfig = None) -> torch.Tensor:
    """Outer loss on real minibatch B."""
    return L_base(theta, B_X, B_A, B_Y, zeta_out, lambda_theta_out, d_plus_1, cfg)


def Phi(theta: torch.Tensor, g: torch.Tensor, Lout_val: torch.Tensor,
        lam: float, rho: float) -> torch.Tensor:
    """Augmented Lagrangian: Φ = L_out + λ·g + (ρ/2)·g²."""
    return Lout_val + lam * g + (rho / 2) * (g ** 2)
