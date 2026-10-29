"""
Compare the reported results (outputs/, laptop) with the independent rerun on different hardware
(outputs/replication/, written by `python scripts/run_fix.py --stage 8`).

The bilevel pipeline is deterministic on one machine but not bit-identical across machines:
small floating-point differences are amplified over the outer iterations and rounds, so single
seeds differ while the data, splits and the centralized oracle are identical. This script checks
whether the ten-seed conclusions (EO gap with the penalty off vs on, paired t-test) hold on the
rerun. Each comparison is paired within one machine; runs from the two machines are never mixed.

Run from repo root:  python scripts/replication_summary.py
Writes outputs/replication/summary.md
"""
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
REP = os.path.join(OUT, "replication")
DS = [("credit", "Credit", "final2univ"), ("adult", "Adult", "final2univ"),
      ("law", "Law", "final2"), ("compas", "COMPAS", "final2")]


def load(d, tag):
    fs = sorted(glob.glob(os.path.join(d, f"draft_results_{tag}_seed*.json")),
                key=lambda p: int(p.rsplit("seed", 1)[1].split(".")[0]))
    return [json.load(open(f)) for f in fs]


def ptest(on, off):
    d = np.asarray(on) - np.asarray(off)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), len(d) - 1)
    except ImportError:
        p = float("nan")
    return d.mean(), p, int((d < 0).sum())


def row(d, ds, tag):
    on, off = load(d, f"{ds}_{tag}"), load(d, f"{ds}_{tag}ctl")
    if len(on) != 10 or len(off) != 10:
        return None, f"{len(on)}/10 runs, {len(off)}/10 controls"
    eo_on = [r["pipeline"]["EO_gap"] for r in on]
    eo_off = [r["pipeline"]["EO_gap"] for r in off]
    acc_on = [r["pipeline"]["accuracy"] for r in on]
    acc_off = [r["pipeline"]["accuracy"] for r in off]
    diff, p, better = ptest(eo_on, eo_off)
    return (np.mean(eo_off), np.mean(eo_on), diff, better, p, min(1.0, 4 * p),
            np.mean(acc_off), np.mean(acc_on)), None


def main():
    lines = ["| Dataset | Run | EO off | EO on | paired diff | seeds better | p | p (Bonferroni x4) | acc off | acc on |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, name, tag in DS:
        for lab, d in (("reported", OUT), ("replication", REP)):
            r, err = row(d, ds, tag)
            if r is None:
                lines.append(f"| {name} | {lab} | missing: {err} |||||||| ")
                continue
            lines.append(f"| {name} | {lab} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:+.3f} | {r[3]}/10 | {r[4]:.3f} | "
                         f"{r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} |")
    text = "\n".join(lines)
    os.makedirs(REP, exist_ok=True)
    open(os.path.join(REP, "summary.md"), "w", encoding="utf-8").write(
        "# Reported results vs. independent rerun on different hardware\n\n" + text + "\n")
    print(text + "\n\nWrote outputs/replication/summary.md")


if __name__ == "__main__":
    main()
