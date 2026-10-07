"""
Final tables for the corrected method (scripts/run_fix.py --stage 6), 10 seeds.

Reads outputs/draft_results_<ds>_final2[ctl|univ]_seed<s>.json and writes
  outputs/final2_summary.md   -- all numbers + paired AL-vs-control test per dataset
  outputs/final2_tables.tex   -- Tables I (performance), II (fairness), III (Universum toggle),
                                 IV (AL vs rho=0 control)
All threshold-dependent metrics use the same validation-tuned threshold
(``extended_metrics_thr``), so every column refers to one classifier.

Run from repo root:  python scripts/final2_tables.py
"""
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DS = [("credit", "Credit"), ("adult", "Adult"), ("law", "Law"), ("compas", "COMPAS")]


def load(tag):
    files = sorted(glob.glob(os.path.join(OUT, f"draft_results_{tag}_seed*.json")),
                   key=lambda p: int(p.rsplit("seed", 1)[1].split(".")[0]))
    return [json.load(open(f)) for f in files]


def get(r, who, key):
    m = r[who]
    ext = m.get("extended_metrics_thr", m["extended_metrics"])
    return {"acc": m["accuracy"], "f1": m["F1_score"], "eo": m["EO_gap"],
            "bal": ext["balanced_accuracy"], "prauc": ext["PR_AUC"],
            "dp": ext["DP_gap"], "eod": ext["EOD_gap"]}[key]


def ms(v):
    return f"\\({np.mean(v):.3f}\\pm{np.std(v):.3f}\\)"


def paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    t = d.mean() / se if se > 0 else float("nan")
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), n - 1)
    except ImportError:
        p = float("nan")
    return d.mean(), d.std(ddof=1), t, p, int((d < 0).sum()), n


def main():
    md, t1, t2, t3, t4 = [], [], [], [], []
    for ds, name in DS:
        ours, ctl, univ = load(f"{ds}_final2"), load(f"{ds}_final2ctl"), load(f"{ds}_final2univ")
        if not ours or len(ours) != len(ctl):
            print(f"!! {ds}: {len(ours)} runs, {len(ctl)} controls -- run scripts/run_fix.py --stage 6")
            continue
        n = len(ours)
        col = lambda rs, who, k: [get(r, who, k) for r in rs]
        for who, label, rs in (("baseline", "Baseline", ours), ("pipeline", "Ours", ours)):
            t1.append(f"{name} & {label} & " + " & ".join(ms(col(rs, who, k)) for k in ("acc", "f1", "bal", "prauc")) + " \\\\")
            t2.append(f"{name} & {label} & " + " & ".join(ms(col(rs, who, k)) for k in ("dp", "eo", "eod")) + " \\\\")
        t1.append("\\midrule"); t2.append("\\midrule")
        de, sde, t, p, better, _ = paired_t(col(ours, "pipeline", "eo"), col(ctl, "pipeline", "eo"))
        da = np.mean(col(ours, "pipeline", "acc")) - np.mean(col(ctl, "pipeline", "acc"))
        t4.append(f"{name} & {ms(col(ctl, 'pipeline', 'eo'))} & {ms(col(ours, 'pipeline', 'eo'))} & "
                  f"\\({de:+.3f}\\) & {better}/{n} & \\({p:.3f}\\) & \\({da:+.3f}\\) \\\\")
        md.append(f"| {name} | {n} | {np.mean(col(ctl,'pipeline','eo')):.3f} | {np.mean(col(ours,'pipeline','eo')):.3f} | "
                  f"{de:+.3f} ± {sde:.3f} | {better}/{n} | {t:.2f} | {p:.3f} | {da:+.3f} |")
        if univ and len(univ) == n:
            on_label = "Off" if ds in ("credit", "adult") else "On"
            for lab, rs in ((f"{'On' if on_label == 'Off' else 'Off'} (reported)", ours), (on_label, univ)):
                t3.append(f"{name} & {lab} & " + " & ".join(ms(col(rs, 'pipeline', k)) for k in ("acc", "f1", "eo", "dp", "eod")) + " \\\\")
            t3.append("\\midrule")
    md_text = ("| Dataset | seeds | EO control (rho=0) | EO ours | paired diff (mean ± sd) | seeds better | t | p | acc diff |\n"
               "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(md))
    tex = ("% Table I rows (Acc, F1, Bal. Acc., PR-AUC)\n" + "\n".join(t1) +
           "\n\n% Table II rows (DemP, EO, EOD)\n" + "\n".join(t2) +
           "\n\n% Table III rows: Universum toggle (Acc, F1, EO, DemP, EOD); Law forms no Universum points\n" + "\n".join(t3) +
           "\n\n% Table IV rows: AL vs rho=0 control (EO control, EO ours, paired diff, seeds better, p, acc diff)\n" + "\n".join(t4) + "\n")
    open(os.path.join(OUT, "final2_summary.md"), "w", encoding="utf-8").write(md_text + "\n")
    open(os.path.join(OUT, "final2_tables.tex"), "w", encoding="utf-8").write(tex)
    print(md_text)
    print("\nWrote outputs/final2_summary.md and outputs/final2_tables.tex")


if __name__ == "__main__":
    main()
