"""
Per-round data design (Appendix 4.2): stratified minibatch B, synthetic D^s, Universum U.

Run via: python -m draft_model.run_draft (not directly)
"""
import numpy as np
from .notation import (
    ClientOriginalData,
    OriginalMinibatch,
    SyntheticMinibatch,
    UniversumSet,
)

BMIN = 32
BMAX = 128
PTARGET = 8
CMIN = 5
D_S_SIZE = 32
C_MIN_TILDE = 2


def draw_original_minibatch(
    data: ClientOriginalData,
    rng: np.random.Generator,
    Bmin=BMIN,
    Bmax=BMAX,
    ptarget=PTARGET,
    cmin=CMIN,
) -> OriginalMinibatch:
    """
    Draw stratified minibatch B from client data.
    Size: |B| = clip(ceil(p_target / π⁺), B_min, B_max).
    Per-group positive floor c_min ensures representation.
    """
    if data.pi_plus <= 0:
        size = Bmin
    else:
        raw_B = int(np.ceil(ptarget / data.pi_plus))
        size = int(np.clip(raw_B, Bmin, Bmax))

    # Proportional quotas with positive floor enforcement
    q_a0y0 = int(round(data.pi_a0y0 * size))
    q_a0y1 = int(round(data.pi_a0y1 * size))
    q_a1y0 = int(round(data.pi_a1y0 * size))
    q_a1y1 = int(round(data.pi_a1y1 * size))
    cmin_k = min(cmin, data.N_oa0y1, data.N_oa1y1) if (data.N_oa0y1 and data.N_oa1y1) else 0
    cmin_k = max(0, cmin_k)

    if q_a0y1 < cmin_k:
        need = cmin_k - q_a0y1
        q_a0y1 = cmin_k
        q_a0y0 = max(0, q_a0y0 - need)
    if q_a1y1 < cmin_k:
        need = cmin_k - q_a1y1
        q_a1y1 = cmin_k
        q_a1y0 = max(0, q_a1y0 - need)

    total = q_a0y0 + q_a0y1 + q_a1y0 + q_a1y1
    if total != size:
        diff = size - total
        if diff > 0:
            q_a0y0 += diff
        else:
            q_a0y0 = max(0, q_a0y0 + diff)

    # Sample without replacement per stratum
    idx_a0y0 = np.where((data.A == 0) & (data.Y == 0))[0]
    idx_a0y1 = np.where((data.A == 0) & (data.Y == 1))[0]
    idx_a1y0 = np.where((data.A == 1) & (data.Y == 0))[0]
    idx_a1y1 = np.where((data.A == 1) & (data.Y == 1))[0]

    def sample_stratum(idx, q):
        if q <= 0 or len(idx) == 0:
            return np.array([], dtype=int)
        return rng.choice(idx, size=min(q, len(idx)), replace=False)

    i0 = sample_stratum(idx_a0y0, q_a0y0)
    i1 = sample_stratum(idx_a0y1, q_a0y1)
    i2 = sample_stratum(idx_a1y0, q_a1y0)
    i3 = sample_stratum(idx_a1y1, q_a1y1)
    all_idx = np.concatenate([i0, i1, i2, i3])
    rng.shuffle(all_idx)

    X, A, Y = data.X[all_idx], data.A[all_idx], data.Y[all_idx]
    n1 = int(np.sum(Y == 1))
    n0 = len(Y) - n1

    return OriginalMinibatch(
        X=X, A=A, Y=Y,
        q_a0y0=len(i0), q_a0y1=len(i1), q_a1y0=len(i2), q_a1y1=len(i3),
        n0=n0, n1=n1, Delta=max(0, n0 - n1),
    )


def build_synthetic_templates(
    B: OriginalMinibatch,
    Ds_size: int,
    cmin_tilde: int = C_MIN_TILDE,
    rng: np.random.Generator = None,
) -> SyntheticMinibatch:
    """
    Build D^s: synthetic minibatch with fixed (a,y) labels (Appendix 4.2.2).
    Features initialized from stratum-specific Gaussian noise (moment-matched to B).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    B_size = len(B.Y)
    Ds_size = min(Ds_size, B_size)
    scale = Ds_size / B_size

    q_a0y0 = int(round(B.q_a0y0 * scale))
    q_a0y1 = int(round(B.q_a0y1 * scale))
    q_a1y0 = int(round(B.q_a1y0 * scale))
    q_a1y1 = int(round(B.q_a1y1 * scale))

    # Enforce positive floor per group
    if q_a0y1 < cmin_tilde:
        need = cmin_tilde - q_a0y1
        q_a0y1 = cmin_tilde
        q_a0y0 = max(0, q_a0y0 - need)
    if q_a1y1 < cmin_tilde:
        need = cmin_tilde - q_a1y1
        q_a1y1 = cmin_tilde
        q_a1y0 = max(0, q_a1y0 - need)

    total = q_a0y0 + q_a0y1 + q_a1y0 + q_a1y1
    if total != Ds_size:
        q_a0y0 += Ds_size - total
        q_a0y0 = max(0, q_a0y0)

    Delta_s = max(0, (q_a0y0 + q_a1y0) - (q_a0y1 + q_a1y1))

    # Compute per-stratum statistics for moment-matched initialization
    d = B.X.shape[1]
    mean_B = np.mean(B.X, axis=0)
    std_B = np.std(B.X, axis=0)
    std_B = np.where(std_B > 1e-8, std_B, 1.0)

    strata = {
        (0, 0): (q_a0y0, (B.A == 0) & (B.Y == 0)),
        (0, 1): (q_a0y1, (B.A == 0) & (B.Y == 1)),
        (1, 0): (q_a1y0, (B.A == 1) & (B.Y == 0)),
        (1, 1): (q_a1y1, (B.A == 1) & (B.Y == 1)),
    }

    list_x, list_a, list_y = [], [], []
    for (a_val, y_val), (count, mask) in strata.items():
        mean_s = np.mean(B.X[mask], axis=0) if np.any(mask) else mean_B
        std_s = np.std(B.X[mask], axis=0) if np.sum(mask) > 1 else std_B
        std_s = np.where(std_s > 1e-8, std_s, 1.0)
        for _ in range(count):
            list_x.append(mean_s + rng.standard_normal(d) * 0.5 * std_s)
            list_a.append(a_val)
            list_y.append(y_val)

    return SyntheticMinibatch(
        X=np.array(list_x, dtype=np.float64),
        A=np.array(list_a, dtype=np.float64),
        Y=np.array(list_y, dtype=np.float64),
        q_a0y0=q_a0y0, q_a0y1=q_a0y1, q_a1y0=q_a1y0, q_a1y1=q_a1y1,
        Delta_s=Delta_s,
    )


def build_universum_templates(
    Delta_s: int,
    Ds_size: int,
    Delta_k: int,
    d: int,
    rng: np.random.Generator = None,
    q_a0y1: int = None,
    q_a1y1: int = None,
    B: OriginalMinibatch = None,
) -> UniversumSet:
    """
    Build Universum U: pseudo-positive points placed at midpoints between
    minority-positive and negative examples from B (Eq. 7).

    Size: |U| = min(Δˢ, |Dˢ|, Δₖ). All U points assigned to the
    minority-positive group to balance TPR across sensitive groups.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    U_size = max(0, min(Delta_s, Ds_size, Delta_k))
    if U_size == 0:
        return UniversumSet(X=np.zeros((0, d)), A=np.zeros(0))

    # Identify minority-positive group (fewer y=1 examples)
    if q_a0y1 is not None and q_a1y1 is not None:
        if q_a0y1 < q_a1y1:
            minority_a = 0
        elif q_a1y1 < q_a0y1:
            minority_a = 1
        else:
            minority_a = None
    else:
        minority_a = None

    # Assign sensitive attribute
    if minority_a is not None:
        A = np.full(U_size, minority_a, dtype=np.float64)
    else:
        half = U_size // 2
        A = np.array([0.0] * half + [1.0] * (U_size - half), dtype=np.float64)
        rng.shuffle(A)

    # Place points at midpoint of (minority-positive, negative) pairs
    X = np.zeros((U_size, d), dtype=np.float64)
    n_minority_pos = 0
    n_neg = 0
    if B is not None and minority_a is not None:
        mask_minority_pos = (B.A == minority_a) & (B.Y == 1)
        mask_neg = B.Y == 0
        X_minority_pos = B.X[mask_minority_pos]
        X_neg = B.X[mask_neg]
        n_minority_pos = X_minority_pos.shape[0]
        n_neg = X_neg.shape[0]

    if n_minority_pos > 0 and n_neg > 0:
        for i in range(U_size):
            idx_pos = rng.integers(0, n_minority_pos)
            idx_neg = rng.integers(0, n_neg)
            X[i] = (X_minority_pos[idx_pos] + X_neg[idx_neg]) / 2.0
    else:
        X = rng.standard_normal((U_size, d)).astype(np.float64) * 0.5

    return UniversumSet(X=X, A=A)
