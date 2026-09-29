"""
Compare the shipped formulation (draft_results_<ds>_final_seed*.json) against the
paper's proposed formulation (draft_results_<ds>_newmath_seed*.json: eo_surrogate=
score_gap, universum_mode=hinge_shield) across all four datasets.

Run from repo root:  python scripts/compare_newmath.py
"""
import glob
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

DATASETS = ["credit", "adult", "law", "compas"]
METRICS_PERF = ["accuracy", "F1_score"]
METRICS_FAIR = ["EO_gap"]
EXT_FAIR = ["DP_gap", "EOD_gap"]


def load_runs(dataset: str, tag: str):
    pattern = os.path.join(OUT, f"draft_results_{dataset}_{tag}_seed*.json")
    runs = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def summarize(runs, key: str):
    if not runs:
        return None, None
    accs = [r["pipeline"]["accuracy"] for r in runs]
    f1s = [r["pipeline"]["F1_score"] for r in runs]
    eos = [r["pipeline"]["EO_gap"] for r in runs]
    dps = [r["pipeline"]["extended_metrics"]["DP_gap"] for r in runs]
    eods = [r["pipeline"]["extended_metrics"]["EOD_gap"] for r in runs]
    return {
        "n": len(runs),
        "acc": (np.mean(accs), np.std(accs)),
        "f1": (np.mean(f1s), np.std(f1s)),
        "eo": (np.mean(eos), np.std(eos)),
        "dp": (np.mean(dps), np.std(dps)),
        "eod": (np.mean(eods), np.std(eods)),
    }


def fmt(pair):
    if pair is None:
        return "  n/a  "
    m, s = pair
    return f"{m:.3f}±{s:.3f}"


def main():
    print(f"{'Dataset':<8} {'Variant':<10} {'n':<3} {'Acc':<14} {'F1':<14} {'EO gap':<14} {'DP gap':<14} {'EOD gap':<14}")
    print("-" * 100)
    for ds in DATASETS:
        old_runs = load_runs(ds, "final")
        new_runs = load_runs(ds, "newmath")
        old = summarize(old_runs, "old")
        new = summarize(new_runs, "new")
        for tag, s in (("shipped", old), ("proposed", new)):
            if s is None:
                print(f"{ds:<8} {tag:<10} {'--':<3} (no files found)")
                continue
            print(f"{ds:<8} {tag:<10} {s['n']:<3} {fmt(s['acc']):<14} {fmt(s['f1']):<14} "
                  f"{fmt(s['eo']):<14} {fmt(s['dp']):<14} {fmt(s['eod']):<14}")
        if old and new:
            d_acc = new["acc"][0] - old["acc"][0]
            d_eo = new["eo"][0] - old["eo"][0]
            verdict = "proposed lower EO gap" if d_eo < 0 else "shipped lower EO gap"
            print(f"{'':<8} {'delta':<10} {'':<3} {d_acc:+.3f}         {'':<14} {d_eo:+.3f}         "
                  f"{'':<14} {'':<14}   <- {verdict}")
        print()


if __name__ == "__main__":
    main()
