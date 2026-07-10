"""
UCI Adult — Universum Point Visualisation (no torch / sklearn required).
Reproduces the same universum-generation logic as visualize_universum.py
but projects Adult's high-dimensional features to 2D via manual SVD-PCA
using only numpy + pandas + matplotlib.

Run from project root:
    python visualize_universum_adult.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── colour palette (same as 2D visualiser) ──────────────────────────────────
C_G0_NEG  = "#7FAADB"
C_G0_POS  = "#2C5F9B"
C_G1_NEG  = "#E8945A"
C_G1_POS  = "#A31A1A"
C_UNIV    = "#F2B8B8"
C_CONN    = "#444444"

N_RUNS = 5


# ── 1. Load & preprocess UCI Adult ──────────────────────────────────────────
ADULT_COLS = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week","native-country","income",
]

def load_adult(sensitive: str = "sex"):
    folder = os.path.join(PROJECT_ROOT, "UCIAdultdataset")
    train_path = os.path.join(folder, "adult.data")
    test_path  = os.path.join(folder, "adult.test")

    train_df = pd.read_csv(train_path, header=None, names=ADULT_COLS,
                           na_values="?", skipinitialspace=True)
    test_df  = pd.read_csv(test_path,  header=None, names=ADULT_COLS,
                           na_values="?", skipinitialspace=True, skiprows=1)
    for df in (train_df, test_df):
        df["income"] = (df["income"].astype(str).str.strip()
                                    .str.rstrip(".").str.replace(" ", "", regex=False))

    def get_ay(df):
        Y = (df["income"] == ">50K").astype(np.float64).values
        if sensitive == "sex":
            A = (df["sex"].astype(str).str.strip().str.lower() == "male"
                 ).astype(np.float64).values
        else:
            A = (df["race"].astype(str).str.strip().str.lower() == "white"
                 ).astype(np.float64).values
        return A, Y

    A_tr, Y_tr = get_ay(train_df)
    drop_cols = ["income", "sex", "race"]
    def encode(df_tr, df_te):
        Xtr = df_tr.drop(columns=[c for c in drop_cols if c in df_tr.columns])
        Xte = df_te.drop(columns=[c for c in drop_cols if c in df_te.columns])
        for col in Xtr.columns:
            if Xtr[col].dtype == object:
                fill = Xtr[col].replace("?", np.nan).mode()
                fv   = fill.iloc[0] if len(fill) > 0 else ""
                Xtr[col] = Xtr[col].replace("?", fv)
                Xte[col] = Xte[col].replace("?", fv)
        n_tr  = len(Xtr)
        combined = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
        cats = combined.select_dtypes(include="object").columns.tolist()
        if cats:
            combined = pd.get_dummies(combined, columns=cats, drop_first=True)
        combined = np.nan_to_num(combined.astype(np.float64).values,
                                 nan=0.0, posinf=0.0, neginf=0.0)
        return combined[:n_tr], combined[n_tr:]

    X_tr, _ = encode(train_df, test_df)
    return X_tr, A_tr, Y_tr


# ── 2. Helpers ───────────────────────────────────────────────────────────────
def standardise(X):
    mu  = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig < 1e-8] = 1.0
    return (X - mu) / sig, mu, sig


def pca_2d(X):
    """Manual 2-component PCA via SVD."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T          # (n, 2)


def draw_minibatch(X, A, Y, rng,
                   Bmin=32, Bmax=128, ptarget=8, cmin=5):
    pi_plus = np.mean(Y == 1)
    if pi_plus <= 0:
        size = Bmin
    else:
        size = int(np.clip(int(np.ceil(ptarget / pi_plus)), Bmin, Bmax))

    N = len(Y)
    pi = {
        (0,0): np.mean((A==0)&(Y==0)),
        (0,1): np.mean((A==0)&(Y==1)),
        (1,0): np.mean((A==1)&(Y==0)),
        (1,1): np.mean((A==1)&(Y==1)),
    }
    q = {k: int(round(v * size)) for k, v in pi.items()}

    # floor enforcement
    cmin_k = min(cmin,
                 int(np.sum((A==0)&(Y==1))),
                 int(np.sum((A==1)&(Y==1))))
    cmin_k = max(0, cmin_k)
    for a in (0, 1):
        if q[(a,1)] < cmin_k:
            need = cmin_k - q[(a,1)]
            q[(a,1)] = cmin_k
            q[(a,0)] = max(0, q[(a,0)] - need)

    total = sum(q.values())
    q[(0,0)] = max(0, q[(0,0)] + (size - total))

    def sample(mask, n):
        idx = np.where(mask)[0]
        if n <= 0 or len(idx) == 0:
            return np.array([], dtype=int)
        return rng.choice(idx, size=min(n, len(idx)), replace=False)

    idxs = np.concatenate([
        sample((A==0)&(Y==0), q[(0,0)]),
        sample((A==0)&(Y==1), q[(0,1)]),
        sample((A==1)&(Y==0), q[(1,0)]),
        sample((A==1)&(Y==1), q[(1,1)]),
    ])
    rng.shuffle(idxs)
    return X[idxs], A[idxs], Y[idxs], q


def build_universum(X_B, A_B, Y_B, q, Ds_size=32, rng=None):
    """Mirror of build_universum_templates; returns (U_X, minority_a, parents)."""
    if rng is None:
        rng = np.random.default_rng(42)

    n0 = int(np.sum(Y_B == 0))
    n1 = int(np.sum(Y_B == 1))
    Delta_k = max(0, n0 - n1)

    q01 = q.get((0,1), 0)
    q11 = q.get((1,1), 0)
    neg_count = q.get((0,0),0) + q.get((1,0),0)
    pos_count = q01 + q11
    Delta_s = max(0, neg_count - pos_count)

    U_size = max(0, min(Delta_s, Ds_size, Delta_k))
    d = X_B.shape[1]
    if U_size == 0:
        return np.zeros((0, d)), None, []

    minority_a = 0 if q01 < q11 else (1 if q11 < q01 else None)

    if minority_a is not None:
        mask_pos = (A_B == minority_a) & (Y_B == 1)
        mask_neg = (Y_B == 0)
        X_pos = X_B[mask_pos]
        X_neg = X_B[mask_neg]
    else:
        mask_pos = (Y_B == 1)
        mask_neg = (Y_B == 0)
        X_pos = X_B[mask_pos]
        X_neg = X_B[mask_neg]

    if len(X_pos) == 0 or len(X_neg) == 0:
        return rng.standard_normal((U_size, d)) * 0.5, minority_a, []

    U_X = np.zeros((U_size, d))
    parents = []
    for i in range(U_size):
        xp = X_pos[rng.integers(0, len(X_pos))]
        xn = X_neg[rng.integers(0, len(X_neg))]
        U_X[i] = (xp + xn) / 2.0
        parents.append((xp.copy(), xn.copy()))

    return U_X, minority_a, parents


# ── 3. Plotting ──────────────────────────────────────────────────────────────
def style_ax(ax, title):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.4, color="#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles():
    return [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_G0_NEG,
               alpha=0.6, markersize=7, label="Group 0 (Female), Y=0"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_G0_POS,
               alpha=0.6, markersize=7, label="Group 0 (Female), Y=1"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_G1_NEG,
               alpha=0.6, markersize=7, label="Group 1 (Male), Y=0"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_G1_POS,
               alpha=0.6, markersize=7, label="Group 1 (Male), Y=1"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_G1_POS,
               markeredgecolor="black", markeredgewidth=1.2,
               markersize=8, label="Source positive"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_G1_NEG,
               markeredgecolor="black", markeredgewidth=1.2,
               markersize=8, label="Source negative"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=C_UNIV,
               markeredgecolor=C_G1_POS, markeredgewidth=1.3,
               markersize=13, label="Universum (pseudo-pos)"),
        Line2D([0],[0], color=C_CONN, linestyle="--", linewidth=1.4,
               alpha=0.7, label="Midpoint connection"),
    ]


def plot_run(ax, Z_B, A_B, Y_B, Z_U, Z_parents, title, minority_a):
    """Plot one run on axes ax (all in PCA 2D space)."""
    groups = [
        ((A_B==0)&(Y_B==0), C_G0_NEG),
        ((A_B==0)&(Y_B==1), C_G0_POS),
        ((A_B==1)&(Y_B==0), C_G1_NEG),
        ((A_B==1)&(Y_B==1), C_G1_POS),
    ]
    for mask, col in groups:
        ax.scatter(Z_B[mask, 0], Z_B[mask, 1],
                   c=col, s=22, alpha=0.5,
                   edgecolors="white", linewidths=0.2, zorder=2)

    # Connection lines + source highlights
    for i, (xp2, xn2) in enumerate(Z_parents):
        if xp2 is None:
            continue
        u2 = Z_U[i]
        ax.plot([xn2[0], u2[0]], [xn2[1], u2[1]],
                color=C_CONN, lw=1.2, ls="--", alpha=0.65, zorder=3)
        ax.plot([xp2[0], u2[0]], [xp2[1], u2[1]],
                color=C_CONN, lw=1.2, ls="--", alpha=0.65, zorder=3)
        ax.scatter(*xp2, s=60, facecolors=C_G1_POS,
                   edgecolors="black", lw=1.2, zorder=6, alpha=0.9)
        ax.scatter(*xn2, s=60, facecolors=C_G1_NEG,
                   edgecolors="black", lw=1.2, zorder=6, alpha=0.9)

    if len(Z_U) > 0:
        ax.scatter(Z_U[:, 0], Z_U[:, 1],
                   c=C_UNIV, s=90, marker="*", alpha=0.95,
                   edgecolors=C_G1_POS, linewidths=1.4, zorder=7)
        for i in range(len(Z_U)):
            ax.annotate(
                f"U{i+1}", (Z_U[i, 0], Z_U[i, 1]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=7.5, fontweight="bold", color="#333333",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=8,
            )
    style_ax(ax, title)


# ── 4. Main ──────────────────────────────────────────────────────────────────
def main():
    print("Loading UCI Adult data …")
    X_full, A_full, Y_full = load_adult(sensitive="sex")
    print(f"  Loaded {len(Y_full)} rows, {X_full.shape[1]} features")
    print(f"  Female(0): {int(np.sum(A_full==0))}  Male(1): {int(np.sum(A_full==1))}")
    print(f"  Y=1 (>50K): {int(np.sum(Y_full==1))}  Y=0: {int(np.sum(Y_full==0))}")

    # Standardise then PCA the FULL dataset for the projection matrix
    X_std, mu, sig = standardise(X_full)
    print("  Computing PCA projection (SVD) …")
    Xc = X_std - X_std.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    PC = Vt[:2].T      # (d, 2) — projection matrix

    # Use a 2000-row stratified subsample for visualisation (minibatch drawn from it)
    rng_samp = np.random.default_rng(0)
    keep = rng_samp.choice(len(Y_full), size=min(2000, len(Y_full)), replace=False)
    X_sub, A_sub, Y_sub = X_full[keep], A_full[keep], Y_full[keep]

    runs = []
    for i in range(N_RUNS):
        rng = np.random.default_rng(42 + i * 7)
        X_B, A_B, Y_B, q = draw_minibatch(X_sub, A_sub, Y_sub, rng)
        U_X_high, minority_a, parents_high = build_universum(X_B, A_B, Y_B, q, rng=rng)

        # Project to 2D
        Z_B  = ((X_B  - mu) / sig) @ PC
        Z_U  = ((U_X_high - mu) / sig) @ PC if len(U_X_high) > 0 else np.zeros((0, 2))
        Z_par = []
        for xp, xn in parents_high:
            if xp is not None:
                Z_par.append(
                    (((xp - mu)/sig) @ PC,
                     ((xn - mu)/sig) @ PC)
                )
            else:
                Z_par.append((None, None))

        n1 = int(np.sum(Y_B == 1))
        n0 = len(Y_B) - n1
        runs.append(dict(Z_B=Z_B, A_B=A_B, Y_B=Y_B,
                         Z_U=Z_U, Z_par=Z_par,
                         minority_a=minority_a,
                         n_u=len(Z_U), Delta=max(0, n0-n1),
                         B_size=len(Y_B)))
        print(f"  Run {i+1}: |B|={len(Y_B)}, |U|={len(Z_U)}, minority_a={minority_a}")

    # ── Individual per-run plots ──────────────────────────────────────────────
    for i, run in enumerate(runs):
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("white")
        grp = (f" — minority a={run['minority_a']}"
               if run["minority_a"] is not None else "")
        title = (f"Run {i+1}: UCI Adult — Universum Construction{grp}\n"
                 f"|B|={run['B_size']}  Δ={run['Delta']}  |U|={run['n_u']}")
        plot_run(ax, run["Z_B"], run["A_B"], run["Y_B"],
                 run["Z_U"], run["Z_par"], title, run["minority_a"])
        ax.legend(handles=legend_handles(), loc="upper left",
                  fontsize=8, framealpha=0.95, edgecolor="#CCCCCC",
                  fancybox=True)
        fig.tight_layout()
        path = os.path.join(OUTPUTS_DIR, f"universum_adult_run_{i+1}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {path}")

    # ── Combined 5-panel grid ─────────────────────────────────────────────────
    cols = min(N_RUNS, 3)
    rows = (N_RUNS + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5.5*rows))
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_2d(axes).flatten()

    for i, run in enumerate(runs):
        ax = axes_flat[i]
        grp = (f"minority a={run['minority_a']}"
               if run["minority_a"] is not None else "balanced")
        title = f"Run {i+1}  |U|={run['n_u']}  ({grp})"
        plot_run(ax, run["Z_B"], run["A_B"], run["Y_B"],
                 run["Z_U"], run["Z_par"], title, run["minority_a"])
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_ylabel("PC 2", fontsize=9)

    for j in range(N_RUNS, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.legend(handles=legend_handles(), loc="lower center",
               ncol=4, fontsize=9, framealpha=0.95, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.2)
    fig.suptitle(
        "UCI Adult — Universum Pseudo-Positive Construction (PCA projection)\n"
        "Stars = Universum midpoints; dashed lines connect source positive/negative pairs",
        fontsize=13, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "universum_adult_all_runs.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")

    print("\nSummary:")
    for i, run in enumerate(runs):
        grp = f"a={run['minority_a']}" if run["minority_a"] is not None else "mixed"
        print(f"  Run {i+1}: |U|={run['n_u']}, assigned to {grp}, Δ={run['Delta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
