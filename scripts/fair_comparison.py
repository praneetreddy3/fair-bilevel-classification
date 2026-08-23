"""
Legitimate model selection + fair reference comparison.

Extends scripts/pareto_sweep.py's rho grid with a matching --no_universum grid (same rho x
5 seeds x 3 datasets), then for EACH dataset:
  1. Selects one (rho, universum) operating point using ONLY the validation split (never test):
     maximize mean validation accuracy among configs with mean validation EO_gap <= 0.1 (the
     method's own epsilon_EO fairness target); if none qualify, minimize mean validation EO_gap.
  2. Reports that config's TEST accuracy/EO_gap as a 5-seed mean +/- std -- this is "our method"
     going forward in docs/COMPARISON.md.
  3. Computes the Pareto-efficient frontier (on TEST accuracy/EO_gap, across all 10 combos) for
     a fair matched-point comparison against the reference's single point (matched accuracy: is
     our EO lower; matched EO: is our accuracy higher).

Does not touch draft_model/bilevel_al.py or losses.py -- just calls the existing CLI and reads
the already-logged round_logs[-1] validation metrics (draft_model/run_draft.py already computes
these on a held-out validation split, disjoint from test).

Run from project root:  "C:\\Python311\\python.exe" scripts\\fair_comparison.py
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
EPSILON_EO_TARGET = 0.1  # the method's own fairness target, used throughout scripts/run_final.sh
COMMON = ["--num_clients", "5", "--rounds", "8", "--K_inner", "100", "--deterministic", "true"]

DATASET_CFG = {
    "credit": {"sensitive": "sex", "add_intercept": "true", "tune_threshold": "true"},
    "adult": {"sensitive": "sex", "add_intercept": "false", "tune_threshold": "false"},
    "law": {"sensitive": "race", "add_intercept": "true", "tune_threshold": "true"},
}

# Reference (FairSynData) points, from outputs/tables/COMPARISON.csv at rho_o=100. Not re-run here.
REFERENCE_POINTS = {
    "adult": (0.7738, 0.1192),
    "credit": (0.7445, 0.0098),
    "law": (0.8438, 0.0849),
}


def results_filename(dataset, rho, universum_off):
    tag = "_nouniversum" if universum_off else ""
    return f"draft_results_{dataset}_rho{rho}{tag}_seed{{seed}}.json"


def run_one(dataset, rho, universum_off, seed):
    cfg = DATASET_CFG[dataset]
    fname = results_filename(dataset, rho, universum_off).format(seed=seed)
    out_path = os.path.join(SWEEP_DIR, fname)
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
        "--results_file", fname,
    ]
    if universum_off:
        cmd.append("--no_universum")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(out_path) as f:
        return json.load(f)


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def build_grid():
    """dataset -> list of dicts: {universum_off, rho, val_acc_m/s, val_eo_m/s, test_acc_m/s, test_eo_m/s}"""
    grid = {}
    for dataset in DATASET_CFG:
        rows = []
        for universum_off in (False, True):
            for rho in RHOS:
                val_accs, val_eos, test_accs, test_eos = [], [], [], []
                for seed in SEEDS:
                    d = run_one(dataset, rho, universum_off, seed)
                    pipe = d.get("pipeline", {})
                    rlogs = d.get("round_logs", [])
                    if pipe.get("accuracy") is None or not rlogs:
                        continue
                    last = rlogs[-1]
                    val_accs.append(last["val_accuracy"])
                    val_eos.append(last["val_EO_gap"])
                    test_accs.append(pipe["accuracy"])
                    test_eos.append(pipe["EO_gap"])
                if not test_accs:
                    print(f"WARNING: no results for {dataset} rho={rho} universum_off={universum_off}")
                    continue
                va_m, va_s = mean_std(val_accs)
                ve_m, ve_s = mean_std(val_eos)
                ta_m, ta_s = mean_std(test_accs)
                te_m, te_s = mean_std(test_eos)
                rows.append({
                    "universum_off": universum_off, "rho": rho,
                    "val_acc_m": va_m, "val_acc_s": va_s, "val_eo_m": ve_m, "val_eo_s": ve_s,
                    "test_acc_m": ta_m, "test_acc_s": ta_s, "test_eo_m": te_m, "test_eo_s": te_s,
                    "n": len(test_accs),
                })
                uni_tag = "no-universum" if universum_off else "universum"
                print(f"{dataset} rho={rho} {uni_tag}: "
                      f"val_acc={va_m:.4f} val_EO={ve_m:.4f} | "
                      f"test_acc={ta_m:.4f}+/-{ta_s:.4f} test_EO={te_m:.4f}+/-{te_s:.4f}")
        grid[dataset] = rows
    return grid


def select_config(rows):
    """Validation-only selection: max val accuracy among val_EO<=target; else min val EO."""
    qualifying = [r for r in rows if r["val_eo_m"] <= EPSILON_EO_TARGET]
    if qualifying:
        return max(qualifying, key=lambda r: r["val_acc_m"])
    return min(rows, key=lambda r: (r["val_eo_m"], -r["val_acc_m"]))


def pareto_frontier(rows):
    """Efficient subset on (test_eo_m, test_acc_m): lower EO and/or higher accuracy is better."""
    frontier = []
    for r in rows:
        dominated = any(
            (o["test_eo_m"] <= r["test_eo_m"] and o["test_acc_m"] >= r["test_acc_m"])
            and (o["test_eo_m"] < r["test_eo_m"] or o["test_acc_m"] > r["test_acc_m"])
            for o in rows
        )
        if not dominated:
            frontier.append(r)
    return sorted(frontier, key=lambda r: r["test_eo_m"])


def matched_point_verdicts(dataset, frontier, ref_acc, ref_eo):
    """Matched-accuracy and matched-EO comparisons of our frontier vs. the reference point."""
    # Matched accuracy: frontier points with test_acc_m >= ref_acc, closest one; else best available.
    at_or_above_acc = [r for r in frontier if r["test_acc_m"] >= ref_acc]
    if at_or_above_acc:
        pt = min(at_or_above_acc, key=lambda r: r["test_acc_m"])
        matched_acc_note = (
            f"at >= reference accuracy ({ref_acc:.4f}), our nearest frontier point "
            f"(rho={pt['rho']}, {'no-universum' if pt['universum_off'] else 'universum'}) "
            f"reaches acc={pt['test_acc_m']:.4f} with EO={pt['test_eo_m']:.4f} "
            f"({'LOWER/fairer' if pt['test_eo_m'] < ref_eo else 'higher/less fair'} than reference EO={ref_eo:.4f})"
        )
    else:
        pt = max(frontier, key=lambda r: r["test_acc_m"])
        matched_acc_note = (
            f"our frontier never reaches reference accuracy ({ref_acc:.4f}); "
            f"our highest-accuracy frontier point (rho={pt['rho']}, "
            f"{'no-universum' if pt['universum_off'] else 'universum'}) is "
            f"acc={pt['test_acc_m']:.4f}, EO={pt['test_eo_m']:.4f}"
        )

    # Matched EO: frontier points with test_eo_m <= ref_eo, closest one; else best available.
    at_or_below_eo = [r for r in frontier if r["test_eo_m"] <= ref_eo]
    if at_or_below_eo:
        pt2 = max(at_or_below_eo, key=lambda r: r["test_eo_m"])
        matched_eo_note = (
            f"at <= reference EO ({ref_eo:.4f}), our nearest frontier point "
            f"(rho={pt2['rho']}, {'no-universum' if pt2['universum_off'] else 'universum'}) "
            f"reaches EO={pt2['test_eo_m']:.4f} with acc={pt2['test_acc_m']:.4f} "
            f"({'HIGHER' if pt2['test_acc_m'] > ref_acc else 'lower'} than reference acc={ref_acc:.4f})"
        )
    else:
        pt2 = min(frontier, key=lambda r: r["test_eo_m"])
        matched_eo_note = (
            f"our frontier never reaches reference EO ({ref_eo:.4f}); "
            f"our fairest frontier point (rho={pt2['rho']}, "
            f"{'no-universum' if pt2['universum_off'] else 'universum'}) is "
            f"EO={pt2['test_eo_m']:.4f}, acc={pt2['test_acc_m']:.4f}"
        )

    return matched_acc_note, matched_eo_note


def write_csv(grid, selected):
    path = os.path.join(TABLES_DIR, "fair_comparison_grid.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "universum_off", "rho", "val_acc_mean", "val_acc_std",
                     "val_EO_mean", "val_EO_std", "test_acc_mean", "test_acc_std",
                     "test_EO_mean", "test_EO_std", "n_seeds", "selected"])
        for dataset, rows in grid.items():
            sel = selected[dataset]
            for r in rows:
                is_sel = (r["universum_off"] == sel["universum_off"] and r["rho"] == sel["rho"])
                w.writerow([dataset, r["universum_off"], r["rho"], f"{r['val_acc_m']:.4f}",
                            f"{r['val_acc_s']:.4f}", f"{r['val_eo_m']:.4f}", f"{r['val_eo_s']:.4f}",
                            f"{r['test_acc_m']:.4f}", f"{r['test_acc_s']:.4f}",
                            f"{r['test_eo_m']:.4f}", f"{r['test_eo_s']:.4f}", r["n"], is_sel])
    print(f"Wrote {path}")


def plot(grid, selected):
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")
    for i, dataset in enumerate(DATASET_CFG):
        c = cmap(i % 10)
        rows = grid[dataset]
        for universum_off, ls, marker in [(False, "-", "o"), (True, "--", "s")]:
            d_rows = sorted([r for r in rows if r["universum_off"] == universum_off], key=lambda r: r["rho"])
            if not d_rows:
                continue
            eo = [r["test_eo_m"] * 100 for r in d_rows]
            acc = [r["test_acc_m"] * 100 for r in d_rows]
            tag = "no-universum" if universum_off else "universum"
            ax.plot(eo, acc, marker=marker, linestyle=ls, color=c, linewidth=1.6, markersize=6,
                     alpha=0.85, label=f"{dataset} ({tag})", zorder=3)
        sel = selected[dataset]
        ax.scatter([sel["test_eo_m"] * 100], [sel["test_acc_m"] * 100], s=280, marker="D",
                    color=c, edgecolor="black", linewidth=1.5, zorder=5,
                    label=f"{dataset} (validation-selected)")
        ref = REFERENCE_POINTS.get(dataset)
        if ref is not None:
            ref_acc, ref_eo = ref
            ax.scatter([ref_eo * 100], [ref_acc * 100], s=280, marker="*",
                        color=c, edgecolor="black", linewidth=1.2, zorder=4,
                        label=f"{dataset} (reference, FairSynData)")
    ax.set_xlabel("EO gap (%) -- lower is fairer")
    ax.set_ylabel("Accuracy (%) -- higher is better")
    ax.set_title("Accuracy-Fairness trade-off: our method (Universum on/off x rho) vs. reference",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=1)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "pareto_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    grid = build_grid()
    selected = {ds: select_config(rows) for ds, rows in grid.items()}
    write_csv(grid, selected)
    plot(grid, selected)

    print("\n" + "=" * 70)
    print("VALIDATION-SELECTED CONFIGS (selection never looked at test numbers)")
    print("=" * 70)
    frontiers = {}
    for dataset, sel in selected.items():
        uni_tag = "no-universum" if sel["universum_off"] else "universum"
        print(f"\n{dataset}: rho={sel['rho']}, {uni_tag}  "
              f"[val_acc={sel['val_acc_m']:.4f}, val_EO={sel['val_eo_m']:.4f}]")
        print(f"  TEST: acc={sel['test_acc_m']:.4f}+/-{sel['test_acc_s']:.4f}  "
              f"EO={sel['test_eo_m']:.4f}+/-{sel['test_eo_s']:.4f}")
        frontier = pareto_frontier(grid[dataset])
        frontiers[dataset] = frontier
        ref = REFERENCE_POINTS[dataset]
        acc_note, eo_note = matched_point_verdicts(dataset, frontier, ref[0], ref[1])
        print(f"  Matched-accuracy verdict: {acc_note}")
        print(f"  Matched-EO verdict: {eo_note}")
