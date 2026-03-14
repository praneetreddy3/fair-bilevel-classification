"""
Loss functions for the bilevel fairness pipeline (Eq. 2-4 from draft paper).

Model: f_θ(x,a) = θᵀ[x; a], linear classifier with input dim d+1.
Labels y ∈ {0,1}, sensitive attribute a ∈ {0,1}.
"""
import numpy as np
import torch


def _to_torch(x, dtype=torch.float32, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def pack_xa(X: np.ndarray, A: np.ndarray) -> torch.Tensor:
    """Concatenate features and sensitive attribute: [x; a] → (n, d+1)."""
    A_col = np.asarray(A).reshape(-1, 1)
    return torch.tensor(np.hstack([np.asarray(X), A_col]), dtype=torch.float32)


def f_theta(theta: torch.Tensor, Xa: torch.Tensor) -> torch.Tensor:
    """Linear model: f_θ(x,a) = θᵀ[x;a]."""
    return (Xa @ theta).squeeze(-1)


def logistic_loss_per_sample(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Binary logistic loss: ℓ(y, f) = log(1 + exp(-(2y-1)·f))."""
    margin = (2 * y - 1) * logits
    return torch.log(1 + torch.exp(-margin.clamp(min=-50)))


def L_base(theta: torch.Tensor, X: np.ndarray, A: np.ndarray, Y: np.ndarray,
           zeta: torch.Tensor, lambda_theta: float, d_plus_1: int) -> torch.Tensor:
    """Eq. 2: (1/|D|) Σ ℓ(y, f_θ(x,a)) + λ_θ/(2(d+1)²) ||θ−ζ||²."""
    Xa = pack_xa(X, A).to(theta.device)
    y = _to_torch(Y, device=theta.device)
    logits = f_theta(theta, Xa)
    loss_data = logistic_loss_per_sample(y, logits).mean()
    reg = (lambda_theta / (2 * (d_plus_1 ** 2))) * torch.sum((theta - zeta) ** 2)
    return loss_data + reg


def L_universum(theta: torch.Tensor, X: np.ndarray, A: np.ndarray,
                lambda_U: float, Delta_s: float) -> torch.Tensor:
    """Eq. 3: Universum loss — encourages θ to classify U points as positive (y=1)."""
    if Delta_s <= 0 or len(X) == 0:
        return torch.tensor(0.0, device=theta.device)
    Xa = pack_xa(X, A).to(theta.device)
    logits = f_theta(theta, Xa)
    y_one = torch.ones(Xa.shape[0], device=theta.device)
    loss_u = logistic_loss_per_sample(y_one, logits).mean()
    return lambda_U * loss_u


def g_EO(theta: torch.Tensor, X_plus: np.ndarray, A_plus: np.ndarray) -> torch.Tensor:
    """Eq. 4: EO surrogate on true positives B⁺ — g = (1/|B⁺|) Σ (s−s̄)·f_θ(x,s)."""
    if len(X_plus) == 0:
        return torch.tensor(0.0, device=theta.device)
    Xa = pack_xa(X_plus, A_plus).to(theta.device)
    s = _to_torch(A_plus, device=theta.device)
    s_bar = s.mean()
    f = f_theta(theta, Xa)
    return ((s - s_bar) * f).mean()


def Lin(theta: torch.Tensor,
        Ds_X: np.ndarray, Ds_A: np.ndarray, Ds_Y: np.ndarray,
        U_X: np.ndarray, U_A: np.ndarray,
        zeta: torch.Tensor, lambda_theta_in: float, lambda_U: float,
        d_plus_1: int, Delta_s: int) -> torch.Tensor:
    """Inner objective: Lin(θ) = L_base(θ; Dˢ, ζ) + L_universum(θ; U)."""
    L1 = L_base(theta, Ds_X, Ds_A, Ds_Y, zeta, lambda_theta_in, d_plus_1)
    L2 = L_universum(theta, U_X, U_A, lambda_U, float(Delta_s))
    return L1 + L2


def L_out(theta: torch.Tensor, B_X: np.ndarray, B_A: np.ndarray, B_Y: np.ndarray,
          zeta_out: torch.Tensor, lambda_theta_out: float, d_plus_1: int) -> torch.Tensor:
    """Outer loss on real minibatch B."""
    return L_base(theta, B_X, B_A, B_Y, zeta_out, lambda_theta_out, d_plus_1)


def Phi(theta: torch.Tensor, g: torch.Tensor, Lout_val: torch.Tensor,
        lam: float, rho: float) -> torch.Tensor:
    """Augmented Lagrangian: Φ = L_out + λ·g + (ρ/2)·g²."""
    return Lout_val + lam * g + (rho / 2) * (g ** 2)
