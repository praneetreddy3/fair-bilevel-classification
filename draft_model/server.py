"""
Server-side aggregation (Section 3.2): collect payloads, train global model, evaluate.

Run via: python -m draft_model.run_draft (not directly)
"""
import numpy as np
import torch
from typing import Optional
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from .losses import pack_xa
from .dp import DPConfig, apply_post_server_dp


def aggregate_payloads(payloads: list, dp_config: Optional[DPConfig] = None,
                        rng: Optional[np.random.Generator] = None) -> tuple:
    """
    Concatenate all client payloads into (X, A, Y). Universum points get y=1.

    If dp_config enables post-server DP, apply clipping + Gaussian noise:
    clip(X, clip_min, clip_max) + N(0, sigma^2).
    """
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
    X_agg = np.vstack(X_list)
    A_agg = np.concatenate(A_list)
    Y_agg = np.concatenate(Y_list)
    if dp_config is not None and len(X_agg) > 0:
        X_agg = apply_post_server_dp(X_agg, dp_config, rng=rng)
    return X_agg, A_agg, Y_agg


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
    """Binary F1 score from precision/recall on the positive (y=1) class."""
    tp = np.sum((pred == 1) & (Y == 1))
    fp = np.sum((pred == 1) & (Y == 0))
    fn = np.sum((pred == 0) & (Y == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def compute_eo_gap_and_accuracy(theta: np.ndarray, X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                                threshold: float = 0.0) -> tuple:
    """Evaluate: EO gap = |TPR₁ − TPR₀|, accuracy, and F1.

    ``threshold`` is the decision cutoff on the logit (default 0.0 = original
    behaviour). Use ``pick_threshold`` on a validation split to calibrate it for
    imbalanced data (e.g. credit), which recovers accuracy without changing the model.
    """
    Xa = np.hstack([X, A.reshape(-1, 1)])
    logits = Xa @ theta
    pred = (logits > threshold).astype(np.float64)
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


def pick_threshold(theta: np.ndarray, X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                   metric: str = "balanced") -> float:
    """Choose the logit decision threshold that maximises balanced accuracy on (X,A,Y).

    Balanced accuracy = 0.5*(TPR + TNR), robust to class imbalance. Scans candidate
    thresholds across the observed logit range. Intended to be fit on the VALIDATION
    split and then passed to ``compute_eo_gap_and_accuracy`` for the test set.
    """
    Xa = np.hstack([X, A.reshape(-1, 1)])
    logits = Xa @ theta
    if len(np.unique(Y)) < 2:
        return 0.0
    candidates = np.quantile(logits, np.linspace(0.02, 0.98, 49))
    best_t, best_score = 0.0, -1.0
    P = Y == 1
    N = Y == 0
    for t in candidates:
        pred = logits > t
        tpr = np.mean(pred[P]) if np.any(P) else 0.0
        tnr = np.mean(~pred[N]) if np.any(N) else 0.0
        score = 0.5 * (tpr + tnr)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def compute_extended_metrics(theta: np.ndarray, X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                             threshold: float = 0.0) -> dict:
    """Imbalance/fairness metrics for paper tables: PR-AUC, ROC-AUC, macro-F1,
    balanced accuracy, demographic-parity gap (DP_gap), equalized-odds gap (EOD_gap).

    The threshold-dependent metrics (macro-F1, balanced accuracy, DP_gap, EOD_gap) use the
    logit cutoff ``threshold``. Default 0.0 is the original behaviour; pass the same tuned
    threshold as compute_eo_gap_and_accuracy so all reported metrics refer to one classifier.
    """
    Xa = np.hstack([X, A.reshape(-1, 1)])
    logits = Xa @ theta
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    pred = (logits > threshold).astype(np.float64)

    has_both_classes = len(np.unique(Y)) > 1
    pr_auc = float(average_precision_score(Y, probs)) if has_both_classes else 0.0
    roc_auc = float(roc_auc_score(Y, probs)) if has_both_classes else 0.0
    macro_f1 = float(f1_score(Y, pred, average="macro"))
    balanced_acc = float(balanced_accuracy_score(Y, pred))

    p1 = np.mean(pred[A == 1]) if np.any(A == 1) else 0.0
    p0 = np.mean(pred[A == 0]) if np.any(A == 0) else 0.0
    dp_gap = float(abs(p1 - p0))

    gaps = []
    for y_val in (0, 1):
        y_mask = Y == y_val
        if not np.any(y_mask):
            continue
        a_sub, pred_sub = A[y_mask], pred[y_mask]
        r1 = np.mean(pred_sub[a_sub == 1]) if np.any(a_sub == 1) else 0.0
        r0 = np.mean(pred_sub[a_sub == 0]) if np.any(a_sub == 0) else 0.0
        gaps.append(abs(r1 - r0))
    eod_gap = float(max(gaps)) if gaps else 0.0

    return {
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "macro_F1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "DP_gap": dp_gap,
        "EOD_gap": eod_gap,
    }


def compute_group_rates(theta: np.ndarray, X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                        threshold: float = 0.0) -> dict:
    """Per-group TPR and FPR (item: 'report per-group TPR/FPR')."""
    Xa = np.hstack([X, A.reshape(-1, 1)])
    pred = (Xa @ theta > threshold).astype(np.float64)
    out = {}
    for s in (0, 1):
        mask_s = A == s
        pos = mask_s & (Y == 1)
        neg = mask_s & (Y == 0)
        out[f"TPR_group{s}"] = float(np.mean(pred[pos])) if np.any(pos) else None
        out[f"FPR_group{s}"] = float(np.mean(pred[neg])) if np.any(neg) else None
        out[f"n_group{s}"] = int(np.sum(mask_s))
    return out


def local_vs_pooled_eo(theta: np.ndarray, X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                       num_clients: int, dirichlet_alpha: float, threshold: float = 0.0,
                       seed: int = 0) -> dict:
    """Non-IID diagnostic (checklist item: 'compare local and pooled EO under controlled
    non-IID partitions'). Partitions a held-out set into `num_clients` non-IID shards via
    label-skew Dirichlet partitioning (same recipe as training partitioning), evaluates
    the already-trained global `theta` separately on each shard (local EO/TPR/FPR) and on
    the full pooled set (pooled EO/TPR/FPR), so the two can be compared side by side.

    This is a post-hoc diagnostic on the fixed global theta -- it does not retrain
    per-client models; it asks whether one shared theta's fairness looks the same when
    viewed per-shard as when viewed on the pooled whole.
    """
    from .run_draft import dirichlet_partition_indices  # local import: avoids a cycle at module load

    rng = np.random.default_rng(seed)
    pooled_gap, tpr0, tpr1, pooled_acc, pooled_f1 = compute_eo_gap_and_accuracy(
        theta, X, A, Y, threshold=threshold
    )
    pooled_rates = compute_group_rates(theta, X, A, Y, threshold=threshold)

    client_indices = dirichlet_partition_indices(Y, num_clients, dirichlet_alpha, rng)
    per_client = []
    for k, idx in enumerate(client_indices):
        if len(idx) == 0:
            per_client.append({"client": k, "n": 0, "EO_gap": None})
            continue
        Xk, Ak, Yk = X[idx], A[idx], Y[idx]
        gap_k, tpr0_k, tpr1_k, acc_k, f1_k = compute_eo_gap_and_accuracy(
            theta, Xk, Ak, Yk, threshold=threshold
        )
        rates_k = compute_group_rates(theta, Xk, Ak, Yk, threshold=threshold)
        per_client.append({
            "client": k, "n": int(len(idx)), "EO_gap": gap_k, "accuracy": acc_k, "F1": f1_k,
            **rates_k,
        })

    valid_gaps = [c["EO_gap"] for c in per_client if c["EO_gap"] is not None]
    return {
        "num_clients": num_clients,
        "dirichlet_alpha": dirichlet_alpha,
        "pooled": {"EO_gap": pooled_gap, "accuracy": pooled_acc, "F1": pooled_f1, **pooled_rates},
        "per_client": per_client,
        "mean_local_EO_gap": float(np.mean(valid_gaps)) if valid_gaps else None,
        "max_local_EO_gap": float(np.max(valid_gaps)) if valid_gaps else None,
    }
