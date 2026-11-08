"""
Extra evaluation metrics for the paper (recall, precision, F1, per-group TPR, balanced accuracy, PR-AUC)
for the oracle, Wu et al. (selected rho_o), and our method with the penalty off and on.

No new runs: everything is read from the existing result files (same ten seeds, same test split).
  Oracle / Ours on / Ours off : outputs/draft_results_<ds>_<tag>[ctl]_seed<s>.json  ("baseline" = oracle)
  Wu et al.                   : outputs/wu_baseline/<ds>_f64.json, rho_o selected on validation
Overall recall is recovered from the per-group TPRs weighted by the test set's qualified-group counts;
precision is recovered from F1 and recall (F1 = 2PR/(P+R)).

Run from repo root:  python scripts/extra_metrics.py
Writes outputs/extra_metrics/summary.md and outputs/extra_metrics/table_extra_metrics.tex
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fl_baselines as fb  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
DEST = os.path.join(OUT, "extra_metrics")
DS = [("credit", "Credit", "final2univ"), ("adult", "Adult", "final2univ"),
      ("law", "Law", "final2"), ("compas", "COMPAS", "final2")]
WU_RHO = {"credit": "0.0", "adult": "10.0", "law": "0.0", "compas": "10.0"}  # see outputs/wu_baseline/summary_f64.md


def load(tag):
    fs = sorted(glob.glob(os.path.join(OUT, f"draft_results_{tag}_seed*.json")),
                key=lambda p: int(p.rsplit("seed", 1)[1].split(".")[0]))
    return [json.load(open(f)) for f in fs]


def metrics(r, n0, n1):
    e = r.get("extended_metrics_thr", r["extended_metrics"])
    t0, t1, f1 = r["TPR_group0"], r["TPR_group1"], r["F1_score"]
    rec = (t0 * n0 + t1 * n1) / (n0 + n1)
    prec = f1 * rec / (2 * rec - f1) if 2 * rec - f1 > 0 else float("nan")
    return {"acc": r["accuracy"], "bal": e["balanced_accuracy"], "rec": rec, "prec": prec, "f1": f1,
            "tpr0": t0, "tpr1": t1, "eo": r["EO_gap"], "prauc": e["PR_AUC"]}


def ms(rows, k):
    v = np.array([r[k] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return np.mean(v), np.std(v)


def main():
    os.makedirs(DEST, exist_ok=True)
    keys = ["acc", "bal", "rec", "prec", "f1", "tpr0", "tpr1", "eo", "prauc"]
    head = ["Acc.", "Bal. acc.", "Recall", "Precision", "F1", "TPR S=0", "TPR S=1", "EO gap", "PR-AUC"]
    md = ["| Dataset | Method | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    tex = []
    for ds, name, tag in DS:
        cfg = fb.DATASET_CFG[ds]
        _, (_, At, Yt) = fb.load_dataset(ds, cfg["sensitive"])
        n0 = int(((Yt == 1) & (At == 0)).sum())
        n1 = int(((Yt == 1) & (At == 1)).sum())
        on, off = load(f"{ds}_{tag}"), load(f"{ds}_{tag}ctl")
        wu = json.load(open(os.path.join(OUT, "wu_baseline", f"{ds}_f64.json")))["grid"][WU_RHO[ds]]
        wu_runs = [v for v in wu.values() if isinstance(v, dict) and np.isfinite(v.get("accuracy", np.nan))]
        groups = [("Oracle", [metrics(r["baseline"], n0, n1) for r in on]),
                  ("Wu et al.", [metrics(r, n0, n1) for r in wu_runs]),
                  ("Ours, penalty off", [metrics(r["pipeline"], n0, n1) for r in off]),
                  ("Ours, penalty on", [metrics(r["pipeline"], n0, n1) for r in on])]
        for i, (meth, rows) in enumerate(groups):
            vals = [ms(rows, k) for k in keys]
            md.append(f"| {name} | {meth} | " + " | ".join(f"{m:.3f}±{s:.3f}" for m, s in vals) + " |")
            tex.append(f"{name if i == 0 else ''} & {meth} & " + " & ".join(f"{m:.3f}" for m, _ in vals) + r" \\")
        tex.append(r"\midrule")
    text = "\n".join(md)
    open(os.path.join(DEST, "summary.md"), "w", encoding="utf-8").write(
        "Extra metrics, mean±std over ten seeds (test set, validation-tuned threshold for Credit/Law/COMPAS).\n\n"
        + text + "\n")
    open(os.path.join(DEST, "table_extra_metrics.tex"), "w", encoding="utf-8").write("\n".join(tex[:-1]) + "\n")
    print(text)


if __name__ == "__main__":
    main()
