"""
Create a compact dashboard for Adult ablation and dataset diagnostics.

Outputs:
- outputs/ablation_visualization.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from pipeline.load_adult import prepare_adult_for_draft


PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"


def _load_ablation_metrics():
    files = [
        ("Covariance\n(Baseline)", "ablation_test1.json"),
        ("TPR-gap\nonly", "ablation_test2.json"),
        ("TPR-gap\n+ EMA", "ablation_test3.json"),
        ("TPR-gap\n+ EMA\n+ IW", "ablation_test4.json"),
        ("All features\n(no DP)", "ablation_test5.json"),
        ("All features\n+ DP", "ablation_test6.json"),
    ]
    labels, acc, eo, f1 = [], [], [], []
    for lbl, fn in files:
        p = OUT_DIR / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        labels.append(lbl)
        acc.append(float(d["pipeline"]["accuracy"]))
        eo.append(float(d["pipeline"]["EO_gap"]))
        f1.append(float(d["pipeline"]["F1_score"]))
    return labels, acc, eo, f1


def main():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (16, 12)

    (X_train, A_train, Y_train), _ = prepare_adult_for_draft()
    tests, accuracies, eo_gaps, f1_scores = _load_ablation_metrics()

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Ablation bars
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(tests))
    width = 0.25
    ax1.bar(x - width, accuracies, width, label="Accuracy", alpha=0.85, color="#2E86AB")
    ax1.bar(x, eo_gaps, width, label="EO Gap", alpha=0.85, color="#A23B72")
    ax1.bar(x + width, f1_scores, width, label="F1 Score", alpha=0.85, color="#F18F01")
    ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax1.set_title("Ablation Study: Accuracy vs Fairness vs F1", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tests, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.75, color="red", linestyle="--", linewidth=1.8, alpha=0.7)
    ax1.axhline(y=0.20, color="green", linestyle="--", linewidth=1.8, alpha=0.7)

    # Plot 2: Train distribution by (S, Y)
    ax2 = plt.subplot(2, 3, 2)
    labels = ["S=0,Y=0", "S=0,Y=1", "S=1,Y=0", "S=1,Y=1"]
    counts = [
        int(np.sum((A_train == 0) & (Y_train == 0))),
        int(np.sum((A_train == 0) & (Y_train == 1))),
        int(np.sum((A_train == 1) & (Y_train == 0))),
        int(np.sum((A_train == 1) & (Y_train == 1))),
    ]
    bars = ax2.bar(labels, counts, color=["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9"], edgecolor="black", linewidth=1.0)
    for b, c in zip(bars, counts):
        ax2.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{c}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax2.set_title("Adult Train Set: (S, Y) Imbalance Structure", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: PCA by (S, Y)
    ax3 = plt.subplot(2, 3, 3)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)
    for s in [0, 1]:
        for y in [0, 1]:
            m = (A_train == s) & (Y_train == y)
            if not np.any(m):
                continue
            ax3.scatter(
                X_pca[m, 0],
                X_pca[m, 1],
                label=f"S={s},Y={y}",
                marker=("o" if y == 1 else "x"),
                s=20,
                alpha=(0.7 if y == 1 else 0.35),
            )
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax3.set_title("Adult PCA by (S, Y)", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Compact table
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis("off")
    table_rows = []
    for i, t in enumerate(tests):
        verdict = "OK" if eo_gaps[i] <= 0.20 else "High EO"
        table_rows.append([t.replace("\n", " "), f"{accuracies[i]:.4f}", f"{eo_gaps[i]:.4f}", verdict])
    table = ax4.table(
        cellText=table_rows,
        colLabels=["Test", "Accuracy", "EO Gap", "Verdict"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax4.set_title("Fairness Metrics Summary", fontsize=12, fontweight="bold", pad=10)

    # Plot 5: Stats block
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis("off")
    total_train = len(Y_train)
    pos_ratio = float(np.mean(Y_train == 1))
    s0_ratio = float(np.mean(A_train == 0))
    s1_ratio = float(np.mean(A_train == 1))
    s0_pos_rate = float(np.mean(Y_train[A_train == 0] == 1))
    s1_pos_rate = float(np.mean(Y_train[A_train == 1] == 1))
    stats_text = (
        "ADULT DATASET STATISTICS\n\n"
        f"Train samples: {total_train:,}\n"
        f"Positive class ratio: {pos_ratio:.1%}\n"
        f"S=0 ratio: {s0_ratio:.1%}\n"
        f"S=1 ratio: {s1_ratio:.1%}\n\n"
        f"Positive rate | S=0: {s0_pos_rate:.1%}\n"
        f"Positive rate | S=1: {s1_pos_rate:.1%}\n"
        f"Group positive-rate gap: {abs(s0_pos_rate - s1_pos_rate):.1%}\n"
    )
    ax5.text(0.02, 0.98, stats_text, va="top", fontsize=10, family="monospace")

    # Plot 6: Recommendation block
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    best_idx = int(np.argmax(np.array(accuracies) - 0.25 * np.array(eo_gaps)))
    rec_text = (
        "NEXT STEPS\n\n"
        f"Best trade-off test: {tests[best_idx].replace(chr(10), ' ')}\n"
        f"Acc={accuracies[best_idx]:.4f}, EO={eo_gaps[best_idx]:.4f}, F1={f1_scores[best_idx]:.4f}\n\n"
        "Observations:\n"
        "- Adult baseline is already relatively fair.\n"
        "- DP sigma=1.0 tends to hurt fairness.\n"
        "- Tune sigma in {0.1, 0.2, 0.5} after non-DP tuning.\n"
    )
    ax6.text(0.02, 0.98, rec_text, va="top", fontsize=10, family="monospace")

    plt.tight_layout()
    out_path = OUT_DIR / "ablation_visualization.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
