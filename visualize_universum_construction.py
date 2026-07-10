"""
Visualize Universum construction from 2D data:
real minibatch B -> synthetic D^s -> Universum U.

Outputs:
- outputs/universum_construction.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from draft_model.minibatch_design import (
    build_synthetic_templates,
    build_universum_templates,
    draw_original_minibatch,
)
from draft_model.notation import ClientOriginalData


PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"


def _load_client_a_raw():
    df = pd.read_csv(OUT_DIR / "2d_data.csv")
    dfa = df[df["client"] == "Client A"].copy()
    X = dfa[["x1", "x2"]].to_numpy(dtype=np.float64)
    A = dfa["s"].to_numpy(dtype=np.int64)
    Y = (dfa["y"].to_numpy(dtype=np.float64) == 1.0).astype(np.float64)  # map to {0,1}
    return X, A, Y


def main():
    X, A, Y = _load_client_a_raw()
    data = ClientOriginalData(X=X, A=A, Y=Y)

    rng = np.random.default_rng(42)
    B = draw_original_minibatch(data, rng)
    Ds = build_synthetic_templates(B, Ds_size=32, rng=rng)
    U = build_universum_templates(
        Ds.Delta_s,
        Ds.size(),
        B.Delta,
        d=2,
        rng=rng,
        q_a0y1=B.q_a0y1,
        q_a1y1=B.q_a1y1,
        B=B,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: B
    ax = axes[0]
    for y_val, marker, color in [(1, "o", "green"), (0, "^", "red")]:
        for a_val, fill in [(0, True), (1, False)]:
            m = (B.Y == y_val) & (B.A == a_val)
            if np.any(m):
                ax.scatter(
                    B.X[m, 0],
                    B.X[m, 1],
                    marker=marker,
                    s=80,
                    alpha=0.75,
                    facecolor=(color if fill else "none"),
                    edgecolor=color,
                    linewidth=1.8,
                    label=f"A={a_val},Y={y_val}",
                )
    ax.set_title("Step 1: Real minibatch B")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Plot 2: Ds
    ax = axes[1]
    for y_val, marker, color in [(1, "o", "blue"), (0, "^", "purple")]:
        for a_val, fill in [(0, True), (1, False)]:
            m = (Ds.Y == y_val) & (Ds.A == a_val)
            if np.any(m):
                ax.scatter(
                    Ds.X[m, 0],
                    Ds.X[m, 1],
                    marker=marker,
                    s=60,
                    alpha=0.65,
                    facecolor=(color if fill else "none"),
                    edgecolor=color,
                    linewidth=1.2,
                    label=f"A={a_val},Y={y_val}",
                )
    ax.set_title("Step 2: Synthetic D^s")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Plot 3: U with B context
    ax = axes[2]
    ax.scatter(B.X[:, 0], B.X[:, 1], s=30, alpha=0.15, color="gray", label="B (reference)")
    if U.size() > 0:
        for a_val, color in [(0, "orange"), (1, "pink")]:
            m = U.A == a_val
            if np.any(m):
                ax.scatter(
                    U.X[m, 0],
                    U.X[m, 1],
                    marker="*",
                    s=250,
                    alpha=0.85,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.5,
                    label=f"U: A={a_val}",
                )
    ax.set_title(f"Step 3: Universum U ({U.size()} points)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / "universum_construction.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
