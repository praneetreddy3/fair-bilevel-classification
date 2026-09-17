"""
Best-settings sweep + final run for COMPAS, using the exact same legitimate,
validation-only selection rule as scripts/fair_comparison.py (credit/adult/law):

  1. Grid: rho in {0.01, 0.05, 0.1, 0.5, 1.0} x Universum {on, off}, 5 seeds each (50 runs).
  2. Select ONE (rho, universum) config using ONLY the validation split (never test):
     maximise mean validation accuracy among configs with mean validation EO_gap <= 0.1
     (the method's own epsilon_EO fairness target, same rule as every other dataset);
     if none qualify, minimise mean validation EO_gap instead.
  3. Copy that winning config's 5 seed result files to the standard
     draft_results_compas_final_seed{1..5}.json names (outputs/) that
     build_tables.py / plot_final_results.py / scripts/run_final.sh already expect --
     no need to rerun those 5 seeds a second time.
  4. Print the winning CLI flags so scripts/run_final.sh's COMPAS block can be updated
     to match (keeps the script and the actual reported numbers in sync).

Does not touch draft_model/bilevel_al.py or losses.py -- calls the existing CLI only.
Resumable: skips any (rho, universum, seed) combo whose result file already exists, so
if this gets interrupted partway through, just re-run it.

Run from project root:  python scripts/compas_sweep.py
Needs: PyTorch + CompasData/compas-scores-two-years.csv (or `pip install responsibly`).
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "pareto_sweep")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(SWEEP_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

DATASET = "compas"
SENSITIVE = "race"
RHOS = [0.01, 0.05, 0.1, 0.5, 1.0]
SEEDS = [1, 2, 3, 4, 5]
EPSILON_EO_TARGET = 0.1
COMMON = ["--num_clients", "5", "--rounds", "8", "--K_inner", "100", "--deterministic", "true"]
# add_intercept + tune_threshold ON, matching credit/law (categorical-heavy, imbalance-adjacent
# datasets); leave epsilon_EO at the shared 0.1 target used for selection everywhere else.


def results_filename(rho, universum_off, seed):
    tag = "_nouniversum" if universum_off else ""
    return f"draft_results_{DATASET}_rho{rho}{tag}_seed{seed}.json"


def run_one(rho, universum_off, seed):
    fname = results_filename(rho, universum_off, seed)
    out_path = os.path.join(SWEEP_DIR, fname)
    if os.path.isfile(out_path):
        with open(out_path) as f:
            return json.load(f)
    cmd = [
        sys.executable, "-m", "draft_model.run_draft",
        "--data", DATASET,
        "--sensitive", SENSITIVE,
        "--add_intercept", "true",
        "--tune_threshold", "true",
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
    rows = []
    for universum_off in (False, True):
        for rho in RHOS:
            val_accs, val_eos, test_accs, test_eos = [], [], [], []
            for seed in SEEDS:
                d = run_one(rho, universum_off, seed)
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
                print(f"WARNING: no results for compas rho={rho} universum_off={universum_off}")
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
            print(f"compas rho={rho} {uni_tag}: "
                  f"val_acc={va_m:.4f} val_EO={ve_m:.4f} | "
                  f"test_acc={ta_m:.4f}+/-{ta_s:.4f} test_EO={te_m:.4f}+/-{te_s:.4f}")
    return rows


def select_config(rows):
    """Validation-only selection: max val accuracy among val_EO<=target; else min val EO.
    Identical rule to scripts/fair_comparison.py -- never looks at test numbers."""
    qualifying = [r for r in rows if r["val_eo_m"] <= EPSILON_EO_TARGET]
    if qualifying:
        return max(qualifying, key=lambda r: r["val_acc_m"])
    return min(rows, key=lambda r: (r["val_eo_m"], -r["val_acc_m"]))


def write_csv(rows, selected):
    path = os.path.join(TABLES_DIR, "compas_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["universum_off", "rho", "val_acc_mean", "val_acc_std", "val_EO_mean",
                     "val_EO_std", "test_acc_mean", "test_acc_std", "test_EO_mean",
                     "test_EO_std", "n_seeds", "selected"])
        for r in rows:
            is_sel = (r["universum_off"] == selected["universum_off"] and r["rho"] == selected["rho"])
            w.writerow([r["universum_off"], r["rho"], f"{r['val_acc_m']:.4f}", f"{r['val_acc_s']:.4f}",
                        f"{r['val_eo_m']:.4f}", f"{r['val_eo_s']:.4f}", f"{r['test_acc_m']:.4f}",
                        f"{r['test_acc_s']:.4f}", f"{r['test_eo_m']:.4f}", f"{r['test_eo_s']:.4f}",
                        r["n"], is_sel])
    print(f"Wrote {path}")


def promote_winner_to_final(selected):
    """Copy the winning config's 5 already-computed seed files to the standard
    draft_results_compas_final_seed{1..5}.json names build_tables.py expects --
    no need to rerun those 5 seeds."""
    rho, universum_off = selected["rho"], selected["universum_off"]
    for seed in SEEDS:
        src = os.path.join(SWEEP_DIR, results_filename(rho, universum_off, seed))
        dst = os.path.join(OUT_DIR, f"draft_results_{DATASET}_final_seed{seed}.json")
        shutil.copy2(src, dst)
    print(f"Copied winner (rho={rho}, universum_off={universum_off}) seeds 1-5 to "
          f"outputs/draft_results_{DATASET}_final_seed{{1..5}}.json")


if __name__ == "__main__":
    rows = build_grid()
    selected = select_config(rows)
    write_csv(rows, selected)
    promote_winner_to_final(selected)

    uni_tag = "no-universum" if selected["universum_off"] else "universum"
    uni_flag = " --no_universum" if selected["universum_off"] else ""
    print("\n" + "=" * 70)
    print("WINNER (validation-selected, never looked at test numbers)")
    print("=" * 70)
    print(f"compas: rho={selected['rho']}, {uni_tag}  "
          f"[val_acc={selected['val_acc_m']:.4f}, val_EO={selected['val_eo_m']:.4f}]")
    print(f"  TEST: acc={selected['test_acc_m']:.4f}+/-{selected['test_acc_s']:.4f}  "
          f"EO={selected['test_eo_m']:.4f}+/-{selected['test_eo_s']:.4f}")
    print("\nUpdate scripts/run_final.sh's COMPAS block to match, e.g.:")
    print(f"  python -m draft_model.run_draft --data compas --sensitive race \\")
    print(f"    --add_intercept true --tune_threshold true \\")
    print(f"    --dp_variant none --rho {selected['rho']} --epsilon_EO 0.1{uni_flag} "
          f"--seed $s $COMMON \\")
    print(f"    --results_file draft_results_compas_final_seed$s.json")
    print("\nThen: python build_tables.py && python plot_final_results.py && python verify.py")
