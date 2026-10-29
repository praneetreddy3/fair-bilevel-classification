"""
Group-threshold post-processing baselines (Hardt et al. 2016 style, deterministic thresholds).

For each dataset and seeds 1..10 (same splits as scripts/fl_baselines.py and our method):
  FedAvg+PP : FedAvg model (scripts/fl_baselines.run_method, 8 rounds), then one decision threshold
              per sensitive group.
  Oracle+PP : centralised ridge ERM on the pooled training data (500 Adam steps, the 'Baseline' of
              run_draft.py), then one threshold per group. Reference only: it sees all raw data.

Threshold selection uses VALIDATION data only. Candidate thresholds per group are 99 quantiles of
that group's validation logits. Among pairs (t0, t1) with validation EO gap <= EPS_EO (0.02, the
epsilon_EO of our method), pick the pair maximising the same criterion as the single-threshold
tuning of the dataset: balanced accuracy for Credit/Law/COMPAS (tune_threshold = True), accuracy
for Adult (tune_threshold = False). If no pair is feasible, the pair with the smallest validation
EO gap is used. Test metrics are then computed with the selected pair.

Numpy only (no torch). Run from repo root:  python scripts/postproc_baseline.py
Writes outputs/postproc_baseline/<ds>.json and outputs/postproc_baseline/summary.md
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fl_baselines as fb  # noqa: E402

OUT_DIR = os.path.join(ROOT, "outputs", "postproc_baseline")
SEEDS = list(range(1, 11))
EPS_EO = 0.02
QS = np.linspace(0.01, 0.99, 99)


def group_stats(logits, A, Y, s, thr):
    """For group s and each threshold in thr: (#TP, #TN, #pos, #neg) as arrays."""
    m = A == s
    lg, y = logits[m], Y[m]
    pred = lg[None, :] > thr[:, None]
    tp = (pred & (y == 1)[None, :]).sum(1)
    tn = (~pred & (y == 0)[None, :]).sum(1)
    return tp, tn, int((y == 1).sum()), int((y == 0).sum())


def pick_pair(theta, X, A, Y, balanced):
    logits = fb.xa(X, A) @ theta
    thr = {s: np.quantile(logits[A == s], QS) for s in (0, 1)}
    st = {s: group_stats(logits, A, Y, s, thr[s]) for s in (0, 1)}
    tp0, tn0, p0, n0 = st[0]
    tp1, tn1, p1, n1 = st[1]
    tpr0 = tp0 / max(p0, 1)
    tpr1 = tp1 / max(p1, 1)
    eo = np.abs(tpr1[None, :] - tpr0[:, None])                      # [i0, i1]
    if balanced:
        P, N = p0 + p1, n0 + n1
        crit = 0.5 * ((tp0[:, None] + tp1[None, :]) / P + (tn0[:, None] + tn1[None, :]) / N)
    else:
        crit = (tp0[:, None] + tn0[:, None] + tp1[None, :] + tn1[None, :]) / len(Y)
    feas = eo <= EPS_EO
    if feas.any():
        c = np.where(feas, crit, -np.inf)
        i0, i1 = np.unravel_index(np.argmax(c), c.shape)
    else:
        i0, i1 = np.unravel_index(np.argmin(eo), eo.shape)
    return float(thr[0][i0]), float(thr[1][i1])


def metrics_pair(theta, X, A, Y, t0, t1):
    logits = fb.xa(X, A) @ theta
    pred = np.where(A == 1, logits > t1, logits > t0).astype(float)
    acc = float(np.mean(pred == Y))
    pos = Y == 1
    tpr0 = float(np.mean(pred[pos & (A == 0)]))
    tpr1 = float(np.mean(pred[pos & (A == 1)]))
    tnr = [float(np.mean(1 - pred[(Y == 0) & (A == s)])) for s in (0, 1)]
    bal = 0.5 * (float(np.mean(pred[pos])) + float(np.mean(1 - pred[~pos])))
    dp = abs(float(np.mean(pred[A == 1])) - float(np.mean(pred[A == 0])))
    eod = max(abs(tpr1 - tpr0), abs(tnr[1] - tnr[0]))
    tp = float(np.sum((pred == 1) & pos)); fp = float(np.sum((pred == 1) & ~pos)); fn = float(np.sum((pred == 0) & pos))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
    return {"accuracy": acc, "F1_score": f1, "EO_gap": abs(tpr1 - tpr0), "TPR_group0": tpr0, "TPR_group1": tpr1,
            "balanced_accuracy": bal, "DP_gap": dp, "EOD_gap": eod, "t0": t0, "t1": t1}


def run_dataset(ds):
    cfg = fb.DATASET_CFG[ds]
    raw = fb.load_dataset(ds, cfg["sensitive"])
    out = {"dataset": ds, "eps_EO": EPS_EO, "criterion": "balanced_accuracy" if cfg["tune_threshold"] else "accuracy",
           "FedAvg+PP": {}, "Oracle+PP": {}, "FedAvg_plain": {}}
    for seed in SEEDS:
        data = fb.prepare_splits(raw, seed, cfg["add_intercept"])
        Xv, Av, Yv = data["val"]
        Xt, At, Yt = data["test"]
        th_fa = fb.run_method("FedAvg", {}, data)
        X, A, Y = data["train"]
        th_or = fb.local_adam(np.zeros(X.shape[1] + 1), fb.xa(X, A), Y, np.ones(len(Y)), steps=500)
        for name, th in (("FedAvg+PP", th_fa), ("Oracle+PP", th_or)):
            t0, t1 = pick_pair(th, Xv, Av, Yv, cfg["tune_threshold"])
            r = metrics_pair(th, Xt, At, Yt, t0, t1)
            r["val"] = metrics_pair(th, Xv, Av, Yv, t0, t1)
            out[name][seed] = r
        out["FedAvg_plain"][seed] = fb.evaluate(th_fa, data, cfg["tune_threshold"])  # sanity vs fl_baselines_10seeds
    return out


def ms(v):
    return f"{np.mean(v):.3f}±{np.std(v):.3f}"


def tex(v):
    return f"\\({np.mean(v):.3f}\\pm{np.std(v):.3f}\\)"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tex_rows = []
    md = ["| Dataset | Method | Acc | EO gap | Bal. acc | DemP gap | EOD gap | val EO |",
          "|---|---|---|---|---|---|---|---|"]
    for ds in ("credit", "adult", "law", "compas"):
        out = run_dataset(ds)
        with open(os.path.join(OUT_DIR, f"{ds}.json"), "w") as f:
            json.dump(out, f, indent=2)
        for name in ("FedAvg+PP", "Oracle+PP"):
            rs = list(out[name].values())
            md.append(f"| {ds} | {name} | {ms([r['accuracy'] for r in rs])} | {ms([r['EO_gap'] for r in rs])} | "
                      f"{ms([r['balanced_accuracy'] for r in rs])} | {ms([r['DP_gap'] for r in rs])} | "
                      f"{ms([r['EOD_gap'] for r in rs])} | {ms([r['val']['EO_gap'] for r in rs])} |")
            name_tex = {"credit": "Credit", "adult": "Adult", "law": "Law", "compas": "COMPAS"}[ds]
            tex_rows.append(f"{name_tex} & {name.replace('+PP', ' + post-proc.')} & {tex([r['accuracy'] for r in rs])} & "
                            f"{tex([r['F1_score'] for r in rs])} & {tex([r['EO_gap'] for r in rs])} & "
                            f"{tex([r['DP_gap'] for r in rs])} & {tex([r['EOD_gap'] for r in rs])} \\\\")
        fa = list(out["FedAvg_plain"].values())
        md.append(f"| {ds} | FedAvg (no PP, check) | {ms([r['accuracy'] for r in fa])} | "
                  f"{ms([r['EO_gap'] for r in fa])} | | | | |")
        print("\n".join(md[-3:]), flush=True)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("Group-threshold post-processing, 10 seeds, thresholds chosen on validation "
                f"(val EO <= {EPS_EO}, max balanced acc / acc). See scripts/postproc_baseline.py.\n\n")
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_DIR, "table_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex_rows) + "\n")


if __name__ == "__main__":
    main()
