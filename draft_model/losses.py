"""
Loss functions for the bilevel fairness pipeline.

Updated to use the smooth TPR-gap EO surrogate from the new draft:
g_EO = |TPR_1 - TPR_0| with sigmoid smoothing around threshold tau.

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
    """Eq. 2: (1/|D|) Σ ℓ(y, f_θ(x,a)) + (λ_θ/2) ||θ−ζ||².

    Note: this (used for L_out) scales the ridge by λ_θ/2 as in the paper, whereas the inner
    loop in bilevel_al.py and the server fit in server.py use λ_θ/(2(d+1)²).
    """
    Xa = pack_xa(X, A).to(theta.device)
    y = _to_torch(Y, device=theta.device)
    logits = f_theta(theta, Xa)
    loss_data = logistic_loss_per_sample(y, logits).mean()
    reg = (lambda_theta / 2.0) * torch.sum((theta - zeta) ** 2)
    return loss_data + reg


def L_universum(theta: torch.Tensor, X: np.ndarray, A: np.ndarray,
                lambda_U: float, Delta_s: float,
                mode: str = "logistic_pseudo_positive",
                tau: float = 0.0) -> torch.Tensor:
    """Universum loss.

    mode="logistic_pseudo_positive" (default, original/shipped behavior, Eq. 3):
      encourages theta to classify U points as positive (y=1). Matches the actual
      midpoint-of-positive-and-negative construction in minibatch_design.py.

    mode="hinge_shield" (new, per the draft's description): one-sided hinge that only
      penalizes U points that land on the majority (score > tau) side, i.e.
      L_U = mean(max(f_theta(u) - tau, 0)). Does not require a y=1 target.
    """
    if Delta_s <= 0 or len(X) == 0:
        return torch.tensor(0.0, device=theta.device)
    Xa = pack_xa(X, A).to(theta.device)
    logits = f_theta(theta, Xa)
    if mode == "hinge_shield":
        loss_u = torch.clamp(logits - tau, min=0.0).mean()
        return lambda_U * loss_u
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
    logits = f_theta(theta, Xa)
    smooth_pred = torch.sigmoid(alpha * (logits - tau))
    return smooth_pred.mean()


def compute_tpr_gap_surrogate(
    theta: torch.Tensor,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    tau: float = 0.0,
    alpha: float = 10.0,
) -> torch.Tensor:
    """
    Section 4 surrogate: smooth absolute TPR gap.

    Uses differentiable approximation:
      g = sqrt((TPR_1 - TPR_0)^2 + eps)
    """
    tpr0 = compute_tpr_s(theta, X, A, Y, s_group=0, tau=tau, alpha=alpha)
    tpr1 = compute_tpr_s(theta, X, A, Y, s_group=1, tau=tau, alpha=alpha)
    diff = tpr1 - tpr0
    return torch.sqrt(diff * diff + 1e-12)


def compute_score_gap_surrogate(
    theta: torch.Tensor,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
) -> torch.Tensor:
    """
    Alternative EO surrogate proposed in the paper's revised draft: the unnormalized
    qualified mean-score gap g_EO = mu_1 - mu_0, where mu_s = mean(f_theta(x,s) | Y=1, A=s).

    Derivation note (from the draft): the Zafar-style covariance surrogate equals
    c_EO = p_bar*(1-p_bar)*(mu_1-mu_0), so it silently attenuates when one group is rare
    among the qualified population (p_bar -> 0 or 1). g_EO = mu_1-mu_0 removes that
    attenuation. Unlike compute_tpr_gap_surrogate, this is NOT passed through a sigmoid/
    threshold -- it is a raw mean-score difference, signed (not absolute value).
    """
    y = np.asarray(Y)
    a = np.asarray(A)
    mask1 = (y == 1) & (a == 1)
    mask0 = (y == 1) & (a == 0)
    if not np.any(mask1) or not np.any(mask0):
        return torch.tensor(0.0, dtype=torch.float32, device=theta.device)
    Xa1 = pack_xa(np.asarray(X)[mask1], a[mask1]).to(theta.device)
    Xa0 = pack_xa(np.asarray(X)[mask0], a[mask0]).to(theta.device)
    mu1 = f_theta(theta, Xa1).mean()
    mu0 = f_theta(theta, Xa0).mean()
    return mu1 - mu0


def compute_loss_gap_surrogate(theta, X, A, Y):
    """
    Non-saturating EO surrogate: gap between the groups' mean logistic loss on the qualified
    slice, g = sqrt((m_1 - m_0)^2 + eps), m_s = mean(softplus(-f_theta(x,s)) | Y=1, A=s).
    Unlike the sigmoid TPR gap its gradient does not vanish for misclassified positives, and
    unlike the raw score gap it grows only linearly in the score of those points.
    """
    y = np.asarray(Y); a = np.asarray(A); Xn = np.asarray(X)
    m = []
    for s_group in (0, 1):
        mask = (y == 1) & (a == s_group)
        if not np.any(mask):
            return torch.tensor(0.0, dtype=torch.float32, device=theta.device)
        logits = f_theta(theta, pack_xa(Xn[mask], a[mask]).to(theta.device))
        m.append(torch.nn.functional.softplus(-logits).mean())
    diff = m[1] - m[0]
    return torch.sqrt(diff * diff + 1e-12)


def g_EO(
    theta: torch.Tensor,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    tau: float = 0.0,
    alpha: float = 10.0,
    surrogate: str = "tpr_gap",
) -> torch.Tensor:
    """EO surrogate wrapper.

    surrogate="tpr_gap" (default, original/shipped behavior): smooth |TPR_1-TPR_0|.
    surrogate="score_gap" (new, per the revised draft): signed mu_1-mu_0, no sigmoid/threshold.
    """
    if len(Y) == 0 or not np.any(np.asarray(Y) == 1):
        return torch.tensor(0.0, dtype=torch.float32, device=theta.device)
    if surrogate == "score_gap":
        return compute_score_gap_surrogate(theta, X, A, Y)
    if surrogate == "loss_gap":
        return compute_loss_gap_surrogate(theta, X, A, Y)
    return compute_tpr_gap_surrogate(theta, X, A, Y, tau=tau, alpha=alpha)


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
