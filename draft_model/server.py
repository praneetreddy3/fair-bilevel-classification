"""
Server-side aggregation (Section 3.2): collect payloads, train global model, evaluate.

Run via: python -m draft_model.run_draft (not directly)
"""
import numpy as np
import torch
from .losses import pack_xa
from .notation import Payload


def aggregate_payloads(payloads: list) -> tuple:
    """Concatenate all client payloads into (X, A, Y). Universum points get y=1."""
    X_list, A_list, Y_list = [], [], []
    for pl in payloads:
        X_list.append(pl.synthetic.X)
        A_list.append(pl.synthetic.A)
        Y_list.append(pl.synthetic.Y)
        if pl.universum.size() > 0:
            X_list.append(pl.universum.X)
            A_list.append(pl.universum.A)
            Y_list.append(np.ones(pl.universum.size()))
    if not X_list:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
    return np.vstack(X_list), np.concatenate(A_list), np.concatenate(Y_list)


def train_global_ridge_erm(
    X: np.ndarray, A: np.ndarray, Y: np.ndarray,
    zeta: np.ndarray,
    lambda_theta: float,
    max_iter: int = 200,
    lr: float = 0.05,
    device=None,
) -> np.ndarray:
    """Train global classifier: regularized logistic loss → θ^glob."""
    if device is None:
        device = torch.device("cpu")
    if len(X) == 0:
        return zeta.copy()

    d_plus_1 = X.shape[1] + 1
    theta = torch.tensor(zeta, dtype=torch.float32, device=device, requires_grad=True)
    zeta_t = torch.tensor(zeta, dtype=torch.float32, device=device)
    opt = torch.optim.Adam([theta], lr=lr)
    Xa = pack_xa(X, A).to(device)
    y = torch.tensor(Y, dtype=torch.float32, device=device)

    for _ in range(max_iter):
        opt.zero_grad()
        logits = (Xa @ theta).squeeze(-1)
        margin = (2 * y - 1) * logits
        loss_data = torch.log(1 + torch.exp(-margin.clamp(min=-50))).mean()
        reg = (lambda_theta / (2 * (d_plus_1 ** 2))) * ((theta - zeta_t) ** 2).sum()
        (loss_data + reg).backward()
        opt.step()

    return theta.detach().cpu().numpy()


def compute_f1_score(pred: np.ndarray, Y: np.ndarray) -> float:
    tp = np.sum((pred == 1) & (Y == 1))
    fp = np.sum((pred == 1) & (Y == 0))
    fn = np.sum((pred == 0) & (Y == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def compute_eo_gap_and_accuracy(theta: np.ndarray, X: np.ndarray, A: np.ndarray, Y: np.ndarray) -> tuple:
    """Evaluate: EO gap = |TPR₁ − TPR₀|, accuracy, and F1."""
    Xa = np.hstack([X, A.reshape(-1, 1)])
    logits = Xa @ theta
    pred = (logits > 0).astype(np.float64)
    acc = np.mean(pred == Y)
    f1 = compute_f1_score(pred, Y)

    mask = Y == 1
    if not np.any(mask):
        return 0.0, 0.0, 0.0, acc, f1
    s_pos = A[mask]
    pred_pos = pred[mask]
    tpr0 = np.mean(pred_pos[s_pos == 0]) if np.any(s_pos == 0) else 0.0
    tpr1 = np.mean(pred_pos[s_pos == 1]) if np.any(s_pos == 1) else 0.0

    return float(np.abs(tpr1 - tpr0)), float(tpr0), float(tpr1), float(acc), f1
