"""
Accuracy-vs-EO-gap Pareto sweep for our own method (draft_model/run_draft.py).

Sweeps --rho over {0.01, 0.05, 0.1, 0.5, 1.0} at 5 seeds each, for credit/adult/law, holding
every other setting at that dataset's final config from scripts/run_final.sh. Does NOT touch
draft_model/bilevel_al.py or losses.py -- just calls the existing CLI repeatedly.

Per-seed results go to outputs/pareto_sweep/ (isolated from the tracked outputs/draft_results_*.json
used by build_tables.py's T1-T5, and not meant to be committed -- see docs/RESULTS_KIT.md).
Writes the aggregated mean+/-std table to outputs/tables/pareto_sweep.csv and the figure to
outputs/pareto_tradeoff.png.

Run from project root:  "C:\\Python311\\python.exe" scripts\\pareto_sweep.py
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "pareto_sweep")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(SWEEP_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

RHOS = [0.01, 0.05, 0.1, 0.5, 1.0]
SEEDS = [1, 2, 3, 4, 5]
COMMON = ["--num_clients", "5", "--rounds", "8", "--K_inner", "100", "--deterministic", "true"]

# Final per-dataset settings, matching scripts/run_final.sh.
DATASET_CFG = {
    "credit": {"sensitive": "sex", "add_intercept": "true", "tune_threshold": "true"},
    "adult": {"sensitive": "sex", "add_intercept": "false", "tune_threshold": "false"},
    "law": {"sensitive": "race", "add_intercept": "true", "tune_threshold": "true"},
}

# Reference (FairSynData) points, from outputs/tables/COMPARISON.csv at rho_o=100 -- filled in
# by the finish script or manually if COMPARISON.csv changes. None = no reference point plotted.
REFERENCE_POINTS = {
    "adult": (0.7738, 0.1192),
    "credit": (0.7445, 0.0098),
    "law": (0.8438, 0.0849),
}


def run_one(dataset, rho, seed):
    cfg = DATASET_CFG[dataset]
    results_file = f"draft_results_{dataset}_rho{rho}_seed{seed}.json"
    out_path = os.path.join(SWEEP_DIR, results_file)
    if os.path.isfile(out_path):
        with open(out_path) as f:
            return json.load(f)
    cmd = [
        sys.executable, "-m", "draft_model.run_draft",
        "--data", dataset,
        "--sensitive", cfg["sensitive"],
        "--add_intercept", cfg["add_intercept"],
        "--tune_threshold", cfg["tune_threshold"],
        "--dp_variant", "none",
        "--rho", str(rho),
        "--epsilon_EO", "0.1",
        "--seed", str(seed),
        *COMMON,
        "--out_dir", SWEEP_DIR,
        "--results_file", results_file,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(out_path) as f:
        return json.load(f)


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def sweep():
    rows = []  # dataset, rho, acc_mean, acc_std, eo_mean, eo_std, n
    for dataset in DATASET_CFG:
        for rho in RHOS:
            accs, eos = [], []
            for seed in SEEDS:
                d = run_one(dataset, rho, seed)
                pipe = d.get("pipeline", {})
                if pipe.get("accuracy") is not None:
                    accs.append(pipe["accuracy"])
                    eos.append(pipe["EO_gap"])
            if not accs:
                print(f"WARNING: no results for {dataset} rho={rho}")
                continue
            acc_m, acc_s = mean_std(accs)
            eo_m, eo_s = mean_std(eos)
            rows.append([dataset, rho, acc_m, acc_s, eo_m, eo_s, len(accs)])
            print(f"{dataset} rho={rho}: acc={acc_m:.4f}+/-{acc_s:.4f} EO={eo_m:.4f}+/-{eo_s:.4f} (n={len(accs)})")
    return rows


def write_csv(rows):
    path = os.path.join(TABLES_DIR, "pareto_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "rho", "accuracy_mean", "accuracy_std", "EO_gap_mean", "EO_gap_std", "n_seeds"])
        w.writerows(rows)
    print(f"Wrote {path}")
    return path


def plot(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    datasets = list(DATASET_CFG.keys())
    for i, dataset in enumerate(datasets):
        c = cmap(i % 10)
        d_rows = sorted([r for r in rows if r[0] == dataset], key=lambda r: r[1])
        if not d_rows:
            continue
        eo = [r[4] * 100 for r in d_rows]
        acc = [r[2] * 100 for r in d_rows]
        ax.plot(eo, acc, marker="o", color=c, linewidth=2, markersize=7,
                label=f"{dataset} (our method, rho sweep)", zorder=3)
        for r in d_rows:
            ax.annotate(f"rho={r[1]}", (r[4] * 100, r[2] * 100), fontsize=6,
                        xytext=(5, 4), textcoords="offset points", color=c)
        ref = REFERENCE_POINTS.get(dataset)
        if ref is not None:
            ref_acc, ref_eo = ref
            ax.scatter([ref_eo * 100], [ref_acc * 100], s=260, marker="*",
                       color=c, edgecolor="black", zorder=4,
                       label=f"{dataset} (reference, FairSynData)")
    ax.set_xlabel("EO gap (%) -- lower is fairer")
    ax.set_ylabel("Accuracy (%) -- higher is better")
    ax.set_title("Accuracy-Fairness trade-off: our method (rho sweep) vs. reference", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "pareto_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    rows = sweep()
    write_csv(rows)
    plot(rows)
