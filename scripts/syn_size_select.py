"""
Option B summary: does a larger synthetic batch (m = 64, 128 vs the reported m = 32) help?

Selection uses ONLY validation data, with the same rule as scripts/fair_comparison.py:
maximize mean validation accuracy among sizes with mean validation EO gap <= 0.1; if none qualify,
minimize mean validation EO gap. Validation numbers are round_logs[-1] of each run (logged by
run_draft.py on the held-out validation split); test numbers are reported for every size, but the
size that would be "selected" is decided without them.

Run from repo root:  python scripts/syn_size_select.py
Writes outputs/syn_size_summary.md
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
EO_TARGET = 0.1
DATASETS = ["credit", "adult", "law", "compas"]
TAGS = [(32, "final"), (64, "syn64"), (128, "syn128")]


def load(ds, tag):
    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, f"draft_results_{ds}_{tag}_seed*.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def summarize(runs):
    g = lambda fn: (float(np.mean([fn(r) for r in runs])), float(np.std([fn(r) for r in runs])))
    return {
        "n": len(runs),
        "val_acc": g(lambda r: r["round_logs"][-1]["val_accuracy"])[0],
        "val_eo": g(lambda r: r["round_logs"][-1]["val_EO_gap"])[0],
        "acc": g(lambda r: r["pipeline"]["accuracy"]),
        "f1": g(lambda r: r["pipeline"]["F1_score"]),
        "eo": g(lambda r: r["pipeline"]["EO_gap"]),
        "dp": g(lambda r: r["pipeline"]["extended_metrics"]["DP_gap"]),
        "eod": g(lambda r: r["pipeline"]["extended_metrics"]["EOD_gap"]),
    }


def main():
    f = lambda p: f"{p[0]:.3f}±{p[1]:.3f}"
    lines = ["| Dataset | m | n seeds | val Acc | val EO | test Acc | test F1 | test EO | test DemP | test EOD | selected on validation |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        rows = {m: summarize(r) for m, tag in TAGS if (r := load(ds, tag))}
        complete = {m: s for m, s in rows.items() if s["n"] == 5}
        feas = {m: s for m, s in complete.items() if s["val_eo"] <= EO_TARGET}
        if feas:
            sel = max(feas, key=lambda m: feas[m]["val_acc"])
        elif complete:
            sel = min(complete, key=lambda m: complete[m]["val_eo"])
        else:
            sel = None
        for m, s in rows.items():
            lines.append(f"| {ds} | {m} | {s['n']} | {s['val_acc']:.3f} | {s['val_eo']:.3f} | {f(s['acc'])} | "
                         f"{f(s['f1'])} | {f(s['eo'])} | {f(s['dp'])} | {f(s['eod'])} | {'**yes**' if m == sel else ''} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "syn_size_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
