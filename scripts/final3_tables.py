"""
Final tables after Dr. Mousavi's review (corrected method, 10 seeds, validation-selected Universum).

Configuration per dataset (all chosen on validation data only):
  variant / rho : from stage 5 (scripts/run_fix.py FINAL2)
  Universum     : max validation accuracy s.t. validation EO <= 0.1 between the two stage-6 settings
                  -> Off for Credit, Adult and COMPAS; Law forms no Universum points.
Files: Ours = final2 (Law, COMPAS) / final2univ (Credit, Adult); control (rho = 0) = final2ctl /
final2univctl; non-IID check = final3noniid (10 seeds).

Writes outputs/final3_summary.md and outputs/final3_tables.tex (LaTeX rows; .tex is not tracked).
Run from repo root:  python scripts/final3_tables.py
"""
import glob
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DS = [("credit", "Credit"), ("adult", "Adult"), ("law", "Law"), ("compas", "COMPAS")]
UNIV_OFF_BY_TOGGLE = {"credit": True, "adult": True, "law": False, "compas": False}  # see docstring


def load(tag):
    fs = sorted(glob.glob(os.path.join(OUT, f"draft_results_{tag}_seed*.json")),
                key=lambda p: int(p.rsplit("seed", 1)[1].split(".")[0]))
    return [json.load(open(f)) for f in fs]


def get(r, who, k):
    m = r[who]
    e = m.get("extended_metrics_thr", m["extended_metrics"])
    return {"acc": m["accuracy"], "f1": m["F1_score"], "eo": m["EO_gap"], "bal": e["balanced_accuracy"],
            "prauc": e["PR_AUC"], "dp": e["DP_gap"], "eod": e["EOD_gap"],
            "vacc": r["round_logs"][-1]["val_accuracy"], "veo": r["round_logs"][-1]["val_EO_gap"]}[k]


def col(rs, who, k):
    return [get(r, who, k) for r in rs]


def ms(v, d=3):
    return f"\\({np.mean(v):.{d}f}\\pm{np.std(v):.{d}f}\\)"


def ptest(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / math.sqrt(len(d))
    t = d.mean() / se
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), len(d) - 1)
    except ImportError:
        p = float("nan")
    return d.mean(), d.std(ddof=1), t, p, int((d < 0).sum())


def main():
    T1, T2, TA, TU, T5, md = [], [], [], [], [], []
    for ds, name in DS:
        tog = UNIV_OFF_BY_TOGGLE[ds]
        ours = load(f"{ds}_final2univ" if tog else f"{ds}_final2")
        ctl = load(f"{ds}_final2univctl" if tog else f"{ds}_final2ctl")
        if len(ours) != 10 or len(ctl) != 10:
            print(f"!! {name}: {len(ours)} runs / {len(ctl)} controls -- run scripts/run_fix.py --stage 7")
            continue
        for who, lab in (("baseline", "Oracle (pooled real data)"), ("pipeline", "Ours")):
            T1.append(f"{name} & {lab} & " + " & ".join(ms(col(ours, who, k)) for k in ("acc", "f1", "bal", "prauc")) + " \\\\")
            T2.append(f"{name} & {lab} & " + " & ".join(ms(col(ours, who, k)) for k in ("dp", "eo", "eod")) + " \\\\")
        T1.append("\\midrule"); T2.append("\\midrule")
        de, sd, t, p, better = ptest(col(ours, "pipeline", "eo"), col(ctl, "pipeline", "eo"))
        TA.append(f"{name} & {ms(col(ctl,'pipeline','eo'))} & {ms(col(ours,'pipeline','eo'))} & \\({de:+.3f}\\) & "
                  f"{better}/10 & \\({p:.3f}\\) & {ms(col(ctl,'pipeline','acc'))} & {ms(col(ours,'pipeline','acc'))} \\\\")
        md.append(f"| {name} | {np.mean(col(ctl,'pipeline','eo')):.3f} | {np.mean(col(ours,'pipeline','eo')):.3f} | "
                  f"{de:+.3f} ± {sd:.3f} | {better}/10 | {p:.3f} | {np.mean(col(ctl,'pipeline','acc')):.3f} | "
                  f"{np.mean(col(ours,'pipeline','acc')):.3f} |")
        if ds != "law":
            on_runs = load(f"{ds}_final2") if tog else load(f"{ds}_final2univ")
            for lab, rs, sel in (("On", on_runs, False), ("Off (selected)", ours, True)):
                TU.append(f"{name} & {lab} & {np.mean(col(rs,'pipeline','vacc')):.3f} & {np.mean(col(rs,'pipeline','veo')):.3f} & "
                          + " & ".join(ms(col(rs, "pipeline", k)) for k in ("acc", "f1", "eo", "dp", "eod")) + " \\\\")
            TU.append("\\midrule")
        nn = load(f"{ds}_final3noniid")
        if len(nn) == 10:
            c = [r["non_iid_eo_check"] for r in nn]
            b = nn[0]["dp_privacy_budget"]
            T5.append(f"{name} & {ms([x['pooled']['EO_gap'] for x in c])} & {ms([x['mean_local_EO_gap'] for x in c])} & "
                      f"{ms([x['max_local_EO_gap'] for x in c])} & {b['epsilon_per_release']:.1f} & {b['epsilon_basic_composition']:.1f} \\\\")
    tex = ("% Table I (Acc, F1, Bal. Acc., PR-AUC): Oracle vs Ours, 10 seeds\n" + "\n".join(T1) +
           "\n\n% Table II (DemP, EO, EOD): Oracle vs Ours, 10 seeds\n" + "\n".join(T2) +
           "\n\n% Primary AL evidence: penalty off vs on (EO off, EO on, paired diff, seeds better, unadjusted p, acc off, acc on)\n" + "\n".join(TA) +
           "\n\n% Universum (val Acc, val EO, then test Acc, F1, EO, DemP, EOD); selection = max val Acc s.t. val EO <= 0.1\n" + "\n".join(TU) +
           "\n\n% Non-IID diagnostic, 10 seeds (pooled EO, mean local EO, max local EO, eps/release, eps basic comp.; sigma=1, delta=1e-5)\n" + "\n".join(T5) + "\n")
    open(os.path.join(OUT, "final3_tables.tex"), "w", encoding="utf-8").write(tex)
    text = ("| Dataset | EO penalty off | EO penalty on | paired diff | seeds better | p (unadjusted) | acc off | acc on |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(md))
    open(os.path.join(OUT, "final3_summary.md"), "w", encoding="utf-8").write(text + "\n")
    print(text + "\n\nWrote outputs/final3_summary.md and outputs/final3_tables.tex")


if __name__ == "__main__":
    main()
