"""
Paper-ready figures: final 5-seed results only (one point per dataset, mean +/- std),
NOT the full sweep. Use this for the manuscript; use plot_results.py if you want every
sweep/ablation/dirichlet/winner run ever logged (useful for debugging, unreadable as a
paper figure once there are 50+ runs in outputs/).

Reads outputs/draft_results_{dataset}_final_seed{1..5}.json for each dataset in DATASETS
(silently skips a dataset with no final-seed files yet, e.g. before compas has been run).

Run from project root: python plot_final_results.py
Writes: outputs/final_pareto.png, outputs/final_bars.png
"""
import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

DATASETS = ["credit", "adult", "law", "compas"]
COLORS = {"credit": "#2a9d8f", "adult": "#e76f51", "law": "#264653", "compas": "#8e44ad"}
DISPLAY = {"credit": "Credit", "adult": "Adult", "law": "Law", "compas": "COMPAS"}


def load_dataset(ds):
    files = sorted(glob.glob(os.path.join(OUT, f"draft_results_{ds}_final_seed*.json")))
    runs = []
    for f in files:
        try:
            runs.append(json.load(open(f)))
        except Exception:
            continue
    if not runs:
        return None

    def agg(key_model, key_metric, nested=None):
        vals = []
        for r in runs:
            v = r[key_model]
            if nested:
                v = v[nested]
            vals.append(v[key_metric])
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std())

    return {
        "n_seeds": len(runs),
        "base_acc": agg("baseline", "accuracy"),
        "base_eo": agg("baseline", "EO_gap"),
        "pipe_acc": agg("pipeline", "accuracy"),
        "pipe_eo": agg("pipeline", "EO_gap"),
    }


def main():
    data = {ds: load_dataset(ds) for ds in DATASETS}
    data = {ds: d for ds, d in data.items() if d is not None}
    if not data:
        print(f"No draft_results_{{dataset}}_final_seed*.json found in {OUT}. Run scripts/run_final.sh first.")
        return

    # ---- Pareto: one baseline (star) + one pipeline (circle) point per dataset, with
    # error bars = std across the 5 seeds. ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for ds, d in data.items():
        c = COLORS[ds]
        ax.errorbar(d["pipe_eo"][0] * 100, d["pipe_acc"][0] * 100,
                     xerr=d["pipe_eo"][1] * 100, yerr=d["pipe_acc"][1] * 100,
                     fmt="o", ms=12, color=c, ecolor=c, elinewidth=1.5, capsize=4,
                     markeredgecolor="black", label=f"{DISPLAY[ds]} — pipeline", zorder=3)
        ax.errorbar(d["base_eo"][0] * 100, d["base_acc"][0] * 100,
                     xerr=d["base_eo"][1] * 100, yerr=d["base_acc"][1] * 100,
                     fmt="*", ms=18, color=c, ecolor=c, elinewidth=1.5, capsize=4,
                     markeredgecolor="black", label=f"{DISPLAY[ds]} — baseline", zorder=3)
    ax.set_xlabel("EO gap (%) — lower is fairer")
    ax.set_ylabel("Accuracy (%) — higher is better")
    ax.set_title("Accuracy-Fairness trade-off — final results (mean +/- std, 5 seeds)", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_pareto.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Grouped bars: accuracy + EO gap, baseline vs pipeline, one group per dataset. ----
    labels = [DISPLAY[ds] for ds in data]
    x = np.arange(len(data))
    w = 0.35
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))

    base_acc = [data[ds]["base_acc"][0] * 100 for ds in data]
    base_acc_err = [data[ds]["base_acc"][1] * 100 for ds in data]
    pipe_acc = [data[ds]["pipe_acc"][0] * 100 for ds in data]
    pipe_acc_err = [data[ds]["pipe_acc"][1] * 100 for ds in data]
    a1.bar(x - w / 2, base_acc, w, yerr=base_acc_err, capsize=3, label="Baseline", color="#9aa0a6")
    a1.bar(x + w / 2, pipe_acc, w, yerr=pipe_acc_err, capsize=3, label="Pipeline (ours)", color="#2a9d8f")
    for i, v in enumerate(base_acc):
        a1.text(i - w / 2, v + base_acc_err[i] + 1, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(pipe_acc):
        a1.text(i + w / 2, v + pipe_acc_err[i] + 1, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_title("Accuracy (%)", fontweight="bold")
    a1.set_ylim(0, 100)
    a1.legend(fontsize=9); a1.grid(axis="y", alpha=0.3)

    base_eo = [data[ds]["base_eo"][0] * 100 for ds in data]
    base_eo_err = [data[ds]["base_eo"][1] * 100 for ds in data]
    pipe_eo = [data[ds]["pipe_eo"][0] * 100 for ds in data]
    pipe_eo_err = [data[ds]["pipe_eo"][1] * 100 for ds in data]
    a2.bar(x - w / 2, base_eo, w, yerr=base_eo_err, capsize=3, label="Baseline", color="#9aa0a6")
    a2.bar(x + w / 2, pipe_eo, w, yerr=pipe_eo_err, capsize=3, label="Pipeline (ours)", color="#e76f51")
    for i, v in enumerate(base_eo):
        a2.text(i - w / 2, v + base_eo_err[i] + 1, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(pipe_eo):
        a2.text(i + w / 2, v + pipe_eo_err[i] + 1, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_title("EO gap (%) — lower is fairer", fontweight="bold")
    a2.legend(fontsize=9); a2.grid(axis="y", alpha=0.3)

    fig.suptitle("Final results (mean +/- std, 5 seeds)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "final_bars.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote outputs/final_pareto.png and outputs/final_bars.png "
          f"for datasets: {', '.join(data.keys())}")


if __name__ == "__main__":
    main()
