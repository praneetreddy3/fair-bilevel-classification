"""
Generate presentation-ready comparison figures from final_results_table.csv.
Run from project root:  python build_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

df = pd.read_csv(os.path.join(OUT_DIR, "final_results_table.csv"))

PAIRS = [
    ("2D (draft)",            "2D draft baseline",   "2D draft pipeline"),
    ("Stress 1\n(moderate)",  "Stress 1 baseline",   "Stress 1 pipeline"),
    ("Stress 2\n(strong)",    "Stress 2 baseline",   "Stress 2 pipeline"),
]

labels  = [p[0] for p in PAIRS]
bl_rows = [df.loc[df["Experiment"] == p[1]].iloc[0] for p in PAIRS]
pl_rows = [df.loc[df["Experiment"] == p[2]].iloc[0] for p in PAIRS]

BL_COLOR = "#5A7D9A"
PL_COLOR = "#E07B54"
W = 0.32


def grouped_bar(metric, ylabel, title, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(labels))
    v_bl = [float(r[metric]) for r in bl_rows]
    v_pl = [float(r[metric]) for r in pl_rows]

    b1 = ax.bar(x - W / 2, v_bl, W, label="Baseline",
                color=BL_COLOR, edgecolor="white")
    b2 = ax.bar(x + W / 2, v_pl, W, label="Fairness-Aware",
                color=PL_COLOR, edgecolor="white")

    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path}")


# --- Figure 1: EO gap ---
grouped_bar("EO_gap", "Equal Opportunity Gap",
            "EO Gap: Baseline vs Fairness-Aware Pipeline",
            "eo_gap_comparison.png", ylim=(0, 1.05))

# --- Figure 2: Accuracy ---
grouped_bar("Accuracy", "Accuracy",
            "Accuracy: Baseline vs Fairness-Aware Pipeline",
            "accuracy_comparison.png", ylim=(0, 1.15))

# --- Figure 3: Combined summary (4 key metrics) ---
PANELS = [
    ("Accuracy",  "Accuracy"),
    ("Recall",    "Recall"),
    ("EO_gap",    "EO Gap"),
    ("F1",        "F1 Score"),
]
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
for ax, (col, nice) in zip(axes, PANELS):
    x = np.arange(len(labels))
    v_bl = [float(r[col]) for r in bl_rows]
    v_pl = [float(r[col]) for r in pl_rows]
    b1 = ax.bar(x - W / 2, v_bl, W, label="Baseline", color=BL_COLOR,
                edgecolor="white")
    b2 = ax.bar(x + W / 2, v_pl, W, label="Pipeline", color=PL_COLOR,
                edgecolor="white")
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.018,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.018,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(nice, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Baseline vs Fairness-Aware: Key Metrics",
             fontsize=13, y=1.01)
fig.tight_layout()
path = os.path.join(OUT_DIR, "summary_comparison.png")
fig.savefig(path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {path}")

# --- Figure notes ---
notes = os.path.join(OUT_DIR, "figure_notes.txt")
with open(notes, "w") as f:
    f.write("Figure Notes\n")
    f.write("=" * 50 + "\n\n")
    f.write("eo_gap_comparison.png\n")
    f.write("  Shows EO gap (|TPR_g1 - TPR_g0|) for baseline\n")
    f.write("  vs fairness-aware pipeline across three experiment\n")
    f.write("  pairs. Lower is fairer. The pipeline reduces EO gap\n")
    f.write("  from 0.62 to 0.07 (moderate) and 0.86 to 0.14 (strong).\n\n")
    f.write("accuracy_comparison.png\n")
    f.write("  Shows overall accuracy for both methods. Accuracy\n")
    f.write("  drops under the pipeline because the baseline inflates\n")
    f.write("  accuracy by ignoring the hard-to-classify minority group.\n\n")
    f.write("summary_comparison.png\n")
    f.write("  Combined four-panel view: Accuracy, Recall, EO Gap, F1.\n")
    f.write("  Main takeaway: the pipeline trades precision/accuracy\n")
    f.write("  for much higher recall and much lower EO gap. F1 drops\n")
    f.write("  because precision falls more than recall rises in\n")
    f.write("  aggregate, but per-group fairness improves dramatically.\n")
print(f"  Saved {notes}")
