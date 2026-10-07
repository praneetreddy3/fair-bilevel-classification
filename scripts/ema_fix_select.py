"""
Summarise the EMA-fix runs (scripts/run_fix.py) with validation-only selection.

For each dataset: every (rho, epsilon_EO) config found with all 5 seeds is scored on the validation
split (round_logs[-1], threshold 0, exactly as scripts/fair_comparison.py); the selected config is
max mean validation accuracy among configs with mean validation EO gap <= 0.1 (else min validation
EO gap). Test numbers are printed for every config but never used to choose.

Also prints the rho = 0 control (same fix) and the previously reported numbers, so the question
"does the AL stage do anything?" can be answered by comparing selected vs control.

Run from repo root:  python scripts/ema_fix_select.py      -> outputs/ema_fix_summary.md
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DATASETS = ["credit", "adult", "law", "compas"]
EO_TARGET = 0.1


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, pattern)))]


def summ(runs):
    g = lambda fn: (float(np.mean([fn(r) for r in runs])), float(np.std([fn(r) for r in runs])))
    return {"n": len(runs),
            "val_acc": g(lambda r: r["round_logs"][-1]["val_accuracy"])[0],
            "val_eo": g(lambda r: r["round_logs"][-1]["val_EO_gap"])[0],
            "acc": g(lambda r: r["pipeline"]["accuracy"]), "f1": g(lambda r: r["pipeline"]["F1_score"]),
            "eo": g(lambda r: r["pipeline"]["EO_gap"]),
            "dp": g(lambda r: r["pipeline"]["extended_metrics"]["DP_gap"]),
            "eod": g(lambda r: r["pipeline"]["extended_metrics"]["EOD_gap"])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="fix", choices=["fix", "sfix", "smooth", "vA", "vB", "vAB", "vAC", "vABC", "vSA", "vSAB", "vSAC"])
    PFX = ap.parse_args().prefix
    f = lambda p: f"{p[0]:.3f}±{p[1]:.3f}"
    hdr = ("| Dataset | Setting | n | val Acc | val EO | test Acc | test F1 | test EO | test DemP | test EOD | selected |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [hdr]
    for ds in DATASETS:
        grid = {}
        for p in glob.glob(os.path.join(OUT, f"draft_results_{ds}_{PFX}_rho*_eps*_seed1.json")):
            m = re.search(rf"draft_results_{ds}_{PFX}_(rho[0-9.]+_eps[0-9.]+)_seed1", os.path.basename(p))
            if not m:
                continue
            tag = m.group(1)
            pat = f"draft_results_{ds}_{PFX}_{tag}_seed*.json"
            runs = load(pat)
            if len(runs) == 5:
                grid[tag] = summ(runs)
        feas = {k: v for k, v in grid.items() if v["val_eo"] <= EO_TARGET}
        sel = (max(feas, key=lambda k: feas[k]["val_acc"]) if feas
               else min(grid, key=lambda k: grid[k]["val_eo"]) if grid else None)
        extra = {}
        if (r := load(f"draft_results_{ds}_{PFX}ctl_seed*.json")) and len(r) == 5:
            extra["control rho=0 (same fix)"] = summ(r)
        if (r := load(f"draft_results_{ds}_final_seed*.json")) and len(r) == 5:
            extra["previous (EMA bug)"] = summ(r)
            base = [x["baseline"] for x in r]
            extra["Baseline (real ERM)"] = {"n": 5, "val_acc": float("nan"), "val_eo": float("nan"),
                "acc": (np.mean([b["accuracy"] for b in base]), np.std([b["accuracy"] for b in base])),
                "f1": (np.mean([b["F1_score"] for b in base]), np.std([b["F1_score"] for b in base])),
                "eo": (np.mean([b["EO_gap"] for b in base]), np.std([b["EO_gap"] for b in base])),
                "dp": (np.mean([b["extended_metrics"]["DP_gap"] for b in base]), np.std([b["extended_metrics"]["DP_gap"] for b in base])),
                "eod": (np.mean([b["extended_metrics"]["EOD_gap"] for b in base]), np.std([b["extended_metrics"]["EOD_gap"] for b in base]))}
        for name, s in {**{f"{PFX} {k}": v for k, v in grid.items()}, **extra}.items():
            mark = "**yes**" if name == f"{PFX} {sel}" else ""
            lines.append(f"| {ds} | {name} | {s['n']} | {s['val_acc']:.3f} | {s['val_eo']:.3f} | {f(s['acc'])} | "
                         f"{f(s['f1'])} | {f(s['eo'])} | {f(s['dp'])} | {f(s['eod'])} | {mark} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, f"ema_{PFX}_summary.md" if PFX != "fix" else "ema_fix_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
