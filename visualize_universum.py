"""
Publication-quality visualization of Universum point generation.

For each run, produces:
  - Figure A: full dataset with Universum points and connection lines
  - Figure B: zoomed view showing only the Universum construction region

Usage: python visualize_universum.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from draft_model.notation import ClientOriginalData
from draft_model.minibatch_design import draw_original_minibatch, build_synthetic_templates

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

N_RUNS = 5

# --- Color palette ---
C_G0_NEG  = "#7FAADB"   # group 0, Y=0 — muted blue
C_G0_POS  = "#2C5F9B"   # group 0, Y=1 — dark blue
C_G1_NEG  = "#E8945A"   # group 1, Y=0 — muted orange
C_G1_POS  = "#A31A1A"   # group 1, Y=1 — dark red (minority+)
C_UNIV    = "#F2B8B8"   # universum fill — light rose (faded positive)
C_CONN    = "#444444"   # connection lines — dark gray
C_SRC_POS = "#A31A1A"   # source positive highlight edge — same red
C_SRC_NEG = "#555555"   # source negative highlight edge — dark gray


def generate_2d_client_data(rng):
    """2D dataset with group imbalance: group 1 is minority-positive."""
    n_a0y1, n_a0y0 = 50, 60
    n_a1y1, n_a1y0 = 15, 75

    X_a0y1 = rng.normal(loc=[3.5, 3.5], scale=0.7, size=(n_a0y1, 2))
    X_a0y0 = rng.normal(loc=[1.0, 1.0], scale=0.8, size=(n_a0y0, 2))
    X_a1y1 = rng.normal(loc=[2.5, 3.0], scale=0.9, size=(n_a1y1, 2))
    X_a1y0 = rng.normal(loc=[1.5, 1.5], scale=1.0, size=(n_a1y0, 2))

    X = np.vstack([X_a0y0, X_a0y1, X_a1y0, X_a1y1])
    A = np.array([0]*n_a0y0 + [0]*n_a0y1 + [1]*n_a1y0 + [1]*n_a1y1, dtype=np.float64)
    Y = np.array([0]*n_a0y0 + [1]*n_a0y1 + [0]*n_a1y0 + [1]*n_a1y1, dtype=np.float64)
    perm = rng.permutation(len(Y))
    return ClientOriginalData(X=X[perm], A=A[perm], Y=Y[perm])


def build_universum_with_parents(B, Ds, rng):
    """Build Universum points and track the (positive, negative) parent pair."""
    d = B.X.shape[1]
    U_size = max(0, min(Ds.Delta_s, Ds.size(), B.Delta))
    if U_size == 0:
        return np.zeros((0, d)), np.zeros(0), []

    if B.q_a0y1 < B.q_a1y1:
        minority_a = 0
    elif B.q_a1y1 < B.q_a0y1:
        minority_a = 1
    else:
        minority_a = None

    A = np.full(U_size, minority_a if minority_a is not None else 0, dtype=np.float64)
    if minority_a is None:
        half = U_size // 2
        A = np.array([0.0]*half + [1.0]*(U_size - half), dtype=np.float64)
        rng.shuffle(A)

    X = np.zeros((U_size, d), dtype=np.float64)
    parents = []

    if minority_a is not None:
        X_pos = B.X[(B.A == minority_a) & (B.Y == 1)]
        X_neg = B.X[B.Y == 0]
        if len(X_pos) > 0 and len(X_neg) > 0:
            for i in range(U_size):
                xp = X_pos[rng.integers(0, len(X_pos))]
                xn = X_neg[rng.integers(0, len(X_neg))]
                X[i] = (xp + xn) / 2.0
                parents.append((xp.copy(), xn.copy()))
            return X, A, parents

    X = rng.standard_normal((U_size, d)).astype(np.float64) * 0.5
    return X, A, [(None, None)] * U_size


def collect_runs(data, n_runs=N_RUNS, base_seed=42):
    results = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i * 7)
        B = draw_original_minibatch(data, rng)
        Ds = build_synthetic_templates(B, Ds_size=32, rng=rng)
        U_X, U_A, parents = build_universum_with_parents(B, Ds, rng)
        results.append({"B": B, "Ds": Ds, "U_X": U_X, "U_A": U_A, "parents": parents})
    return results


# ─────────────────────────────────────────────────────────────────────
# Reusable plotting function
# ─────────────────────────────────────────────────────────────────────

def plot_universum_generation(
    ax, data, U_X, parents,
    show_labels=True,
    real_size=30,
    real_alpha=0.55,
    univ_size=70,
    conn_lw=1.5,
    conn_alpha=0.75,
    src_size=80,
):
    """
    Plot real data (de-emphasized) + Universum points (prominent)
    + connection lines to source pairs.

    Parameters
    ----------
    ax           : matplotlib Axes
    data         : ClientOriginalData with .X, .A, .Y
    U_X          : (n_u, 2) Universum point coordinates
    parents      : list of (x_pos, x_neg) tuples per Universum point
    show_labels  : annotate each Universum point with U1, U2, ...
    real_size    : marker size for real data
    real_alpha   : alpha for real data (de-emphasize)
    univ_size    : marker size for Universum points
    conn_lw      : linewidth for connection lines
    conn_alpha   : alpha for connection lines
    src_size     : marker size for highlighted source points
    """

    # 1. Real data — de-emphasized background
    groups = [
        ((data.A == 0) & (data.Y == 0), C_G0_NEG, "Group 0, Y=0"),
        ((data.A == 0) & (data.Y == 1), C_G0_POS, "Group 0, Y=1"),
        ((data.A == 1) & (data.Y == 0), C_G1_NEG, "Group 1, Y=0"),
        ((data.A == 1) & (data.Y == 1), C_G1_POS, "Group 1, Y=1"),
    ]
    for mask, color, _ in groups:
        ax.scatter(
            data.X[mask, 0], data.X[mask, 1],
            c=color, s=real_size, alpha=real_alpha,
            edgecolors="white", linewidths=0.2, zorder=2,
        )

    if len(U_X) == 0:
        return

    # 2. Connection lines + highlighted source points
    for i in range(len(U_X)):
        xp, xn = parents[i]
        if xp is None:
            continue

        # Line from negative source to Universum point
        ax.plot(
            [xn[0], U_X[i, 0]], [xn[1], U_X[i, 1]],
            color=C_CONN, linewidth=conn_lw, linestyle="--",
            alpha=conn_alpha, zorder=3,
        )
        # Line from positive source to Universum point
        ax.plot(
            [xp[0], U_X[i, 0]], [xp[1], U_X[i, 1]],
            color=C_CONN, linewidth=conn_lw, linestyle="--",
            alpha=conn_alpha, zorder=3,
        )

        # Highlighted source: positive parent
        ax.scatter(
            xp[0], xp[1], s=src_size, facecolors=C_G1_POS,
            edgecolors="black", linewidths=1.4, zorder=6, alpha=0.9,
        )
        # Highlighted source: negative parent
        ax.scatter(
            xn[0], xn[1], s=src_size, facecolors=C_G1_NEG,
            edgecolors="black", linewidths=1.4, zorder=6, alpha=0.9,
        )

    # 3. Universum points — prominent, lighter fill, strong outline
    ax.scatter(
        U_X[:, 0], U_X[:, 1],
        c=C_UNIV, s=univ_size, alpha=0.95,
        edgecolors=C_G1_POS, linewidths=1.6, zorder=7,
    )

    # 4. Labels: U1, U2, ... near each Universum point
    if show_labels:
        for i in range(len(U_X)):
            ax.annotate(
                f"U{i+1}", (U_X[i, 0], U_X[i, 1]),
                textcoords="offset points", xytext=(7, 7),
                fontsize=8.5, fontweight="bold", color="#333333",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=8,
            )


def _build_legend():
    """Publication-style legend handles grouped logically."""
    return [
        # Real data
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_G0_NEG,
               alpha=0.6, markersize=7, label="Group 0, Y=0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_G0_POS,
               alpha=0.6, markersize=7, label="Group 0, Y=1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_G1_NEG,
               alpha=0.6, markersize=7, label="Group 1, Y=0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_G1_POS,
               alpha=0.6, markersize=7, label="Group 1, Y=1"),
        # Source / Universum
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_G1_POS,
               markeredgecolor="black", markeredgewidth=1.2,
               markersize=8, label="Source positive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_G1_NEG,
               markeredgecolor="black", markeredgewidth=1.2,
               markersize=8, label="Source negative"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_UNIV,
               markeredgecolor=C_G1_POS, markeredgewidth=1.4,
               markersize=9, label="Universum (pseudo-pos)"),
        # Connection
        Line2D([0], [0], color=C_CONN, linestyle="--", linewidth=1.5,
               alpha=0.75, label="Midpoint distance"),
    ]


def _style_axes(ax, title):
    """Apply consistent publication styling."""
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.4, color="#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_info_box(ax, run):
    """Add a neat annotation box with run parameters."""
    n_u = len(run["U_X"])
    text = (
        f"|B| = {len(run['B'].Y)}    "
        f"Delta = {run['B'].Delta}\n"
        f"|Ds| = {run['Ds'].size()}   "
        f"Delta_s = {run['Ds'].Delta_s}\n"
        f"|U|  = min({run['Ds'].Delta_s}, {run['Ds'].size()}, "
        f"{run['B'].Delta}) = {n_u}"
    )
    props = dict(
        boxstyle="round,pad=0.5", facecolor="white",
        alpha=0.92, edgecolor="#BBBBBB", linewidth=0.8,
    )
    ax.text(
        0.97, 0.04, text, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="right",
        bbox=props, family="monospace", color="#333333",
    )


# ─────────────────────────────────────────────────────────────────────
# Output generators
# ─────────────────────────────────────────────────────────────────────

def save_individual_plots(data, runs):
    """For each run: Figure A (full view) + Figure B (zoomed construction)."""
    for i, run in enumerate(runs):
        U_X = run["U_X"]
        parents = run["parents"]
        n_u = len(U_X)
        grp_str = ""
        if n_u > 0:
            if np.all(run["U_A"] == 1):
                grp_str = " for Minority Group (a=1)"
            elif np.all(run["U_A"] == 0):
                grp_str = " for Minority Group (a=0)"

        # ── Figure A: full dataset view ──
        fig_a, ax_a = plt.subplots(figsize=(9, 7))
        fig_a.patch.set_facecolor("white")
        plot_universum_generation(ax_a, data, U_X, parents, show_labels=True)
        _style_axes(ax_a, f"Run {i+1}: Universum Point Construction{grp_str}")
        ax_a.legend(
            handles=_build_legend(), loc="upper left",
            fontsize=8.5, framealpha=0.95, edgecolor="#CCCCCC",
            fancybox=True, borderpad=0.8,
        )
        _add_info_box(ax_a, run)
        fig_a.tight_layout()
        path_a = os.path.join(OUTPUTS_DIR, f"universum_run_{i+1}.png")
        fig_a.savefig(path_a, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig_a)
        print(f"Saved: {path_a}")

        # ── Figure B: zoomed construction view ──
        if n_u > 0:
            all_pts = [U_X]
            for xp, xn in parents:
                if xp is not None:
                    all_pts.append(xp.reshape(1, -1))
                    all_pts.append(xn.reshape(1, -1))
            all_pts = np.vstack(all_pts)
            pad = 1.0
            x_lo, x_hi = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
            y_lo, y_hi = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad

            fig_b, ax_b = plt.subplots(figsize=(8, 6.5))
            fig_b.patch.set_facecolor("white")
            plot_universum_generation(
                ax_b, data, U_X, parents, show_labels=True,
                real_size=25, real_alpha=0.35,
                univ_size=90, conn_lw=1.8, conn_alpha=0.85, src_size=100,
            )
            ax_b.set_xlim(x_lo, x_hi)
            ax_b.set_ylim(y_lo, y_hi)
            _style_axes(ax_b, f"Run {i+1}: Zoomed Construction View")
            ax_b.legend(
                handles=_build_legend(), loc="upper left",
                fontsize=8.5, framealpha=0.95, edgecolor="#CCCCCC",
                fancybox=True, borderpad=0.8,
            )
            fig_b.tight_layout()
            path_b = os.path.join(OUTPUTS_DIR, f"universum_run_{i+1}_zoom.png")
            fig_b.savefig(path_b, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig_b)
            print(f"Saved: {path_b}")


def save_combined_grid(data, runs):
    """5-panel grid — one subplot per run."""
    n = len(runs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.8 * rows))
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes).flatten()

    for i, run in enumerate(runs):
        ax = axes[i]
        plot_universum_generation(
            ax, data, run["U_X"], run["parents"],
            show_labels=True, real_size=18, real_alpha=0.45,
            univ_size=55, conn_lw=1.2, conn_alpha=0.7, src_size=55,
        )
        n_u = len(run["U_X"])
        grp = ""
        if n_u > 0 and np.all(run["U_A"] == 1):
            grp = " | minority: a=1"
        elif n_u > 0 and np.all(run["U_A"] == 0):
            grp = " | minority: a=0"
        _style_axes(ax, f"Run {i+1}:  |U| = {n_u}{grp}")
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)
        ax.set_title(f"Run {i+1}:  |U| = {n_u}{grp}", fontsize=11, fontweight="bold", pad=6)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.legend(
        handles=_build_legend(), loc="lower center",
        ncol=4, fontsize=9.5, framealpha=0.95, edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, -0.01), columnspacing=1.5,
    )
    fig.suptitle(
        "Universum Point Generation Across Runs\n"
        "Each Universum point (light rose) is the midpoint between a "
        "source positive (dark, black-edged) and a source negative",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "universum_all_runs.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    rng = np.random.default_rng(42)
    data = generate_2d_client_data(rng)
    runs = collect_runs(data, n_runs=N_RUNS)

    save_individual_plots(data, runs)
    save_combined_grid(data, runs)

    print("\nSummary:")
    for i, run in enumerate(runs):
        n_u = len(run["U_X"])
        grp = ("a=1" if n_u > 0 and np.all(run["U_A"] == 1) else
               "a=0" if n_u > 0 and np.all(run["U_A"] == 0) else "mixed")
        print(f"  Run {i+1}: |U|={n_u}, assigned to {grp}, "
              f"Delta={run['B'].Delta}, Delta_s={run['Ds'].Delta_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
