"""
Option C: reuse of the released synthetic data by models other than the linear one it was built for.

Idea. Our method releases a synthetic dataset. FairFed/FedFB/FedFair release one trained model, built
for one model class. Here we take the synthetic pool the server collected (all 8 rounds x 5
clients, saved by `run_draft.py --save_synthetic true`) and train DIFFERENT model classes on it
without any further communication with the clients, then test on the real held-out test set.

Three data sources are compared for every model class:
  Real          the pooled real training data (what a non-private centralised learner would use;
                reference, not available in the federated setting)
  Synthetic w/o EO   the synthetic pool from the SAME pipeline with the fairness term off (rho = 0;
                     `--fairness_off`) -- the control that isolates what the EO-AL stage adds
  Synthetic (ours)   the synthetic pool from the reported configuration
Model classes: logistic regression, a small MLP, and gradient boosting (sklearn, fixed settings,
no tuning on any split except the decision threshold below).

Decision threshold: for Credit, Law and COMPAS (the datasets whose reported results use a tuned
threshold) the cutoff on predicted probability is chosen to maximise balanced accuracy on the
validation split; for Adult it is 0.5. This mirrors the reported setup; test data is never used
to choose anything.

Inputs : outputs/draft_results_<ds>_reuse_seed<k>_synthetic.npz   (ours)
         outputs/draft_results_<ds>_reusectl_seed<k>_synthetic.npz (control)
Outputs: outputs/reuse_experiment.json, outputs/reuse_summary.md, outputs/table_reuse.tex

Run from repo root:  python scripts/reuse_experiment.py
"""
import json
import os
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DATASETS = ["credit", "adult", "law", "compas"]
TUNE = {"credit": True, "adult": False, "law": True, "compas": True}
SEEDS = [1, 2, 3, 4, 5]


def make_model(kind, seed):
    if kind == "LogReg":
        return LogisticRegression(C=1e4, max_iter=2000)
    if kind == "MLP":
        return MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=600, random_state=seed)
    if kind == "GradBoost":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=seed)
    raise ValueError(kind)


KINDS = ["LogReg", "MLP", "GradBoost"]


def xa(X, A):
    return np.hstack([X, np.asarray(A, dtype=float).reshape(-1, 1)])


def best_balanced_threshold(p, y):
    best_t, best = 0.5, -1.0
    P, N = y == 1, y == 0
    for t in np.quantile(p, np.linspace(0.02, 0.98, 49)):
        pred = p > t
        s = 0.5 * (np.mean(pred[P]) + np.mean(~pred[N]))
        if s > best:
            best, best_t = s, float(t)
    return best_t


def metrics(pred, A, Y):
    pred = pred.astype(float)
    acc = float(np.mean(pred == Y))
    tp = np.sum((pred == 1) & (Y == 1)); fp = np.sum((pred == 1) & (Y == 0)); fn = np.sum((pred == 0) & (Y == 1))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0

    def rate(mask, s):
        m = mask & (A == s)
        return float(np.mean(pred[m])) if m.any() else 0.0
    pos, neg = Y == 1, Y == 0
    eo = abs(rate(pos, 1) - rate(pos, 0))
    fpr_gap = abs(rate(neg, 1) - rate(neg, 0))
    dp = abs(float(np.mean(pred[A == 1])) - float(np.mean(pred[A == 0])))
    return {"accuracy": acc, "F1": f1, "EO_gap": eo, "DemP_gap": dp, "EOD_gap": max(eo, fpr_gap)}


def fit_eval(kind, seed, Xtr, Atr, Ytr, d, tune):
    if len(np.unique(Ytr)) < 2:
        return None
    m = make_model(kind, seed).fit(xa(Xtr, Atr), Ytr.astype(int))
    pv = m.predict_proba(xa(d["X_val"], d["A_val"]))[:, 1]
    thr = best_balanced_threshold(pv, d["Y_val"]) if tune else 0.5
    pt = m.predict_proba(xa(d["X_test"], d["A_test"]))[:, 1]
    return metrics(pt > thr, d["A_test"], d["Y_test"])


def run_task(task):
    """One (dataset, seed): fit every model class on every data source. Returns a list of results."""
    ds, seed = task
    f_ours = os.path.join(OUT, f"draft_results_{ds}_reuse_seed{seed}_synthetic.npz")
    f_ctl = os.path.join(OUT, f"draft_results_{ds}_reusectl_seed{seed}_synthetic.npz")
    if not (os.path.isfile(f_ours) and os.path.isfile(f_ctl)):
        return []
    ours, ctl = np.load(f_ours), np.load(f_ctl)
    d = {k: ours[k] for k in ("X_val", "A_val", "Y_val", "X_test", "A_test", "Y_test")}
    sources = {
        "Real": (ours["X_train"], ours["A_train"], ours["Y_train"]),
        "Synthetic w/o EO": (ctl["X_syn"], ctl["A_syn"], ctl["Y_syn"]),
        "Synthetic (ours)": (ours["X_syn"], ours["A_syn"], ours["Y_syn"]),
    }
    out = []
    for src, (X, A, Y) in sources.items():
        for kind in KINDS:
            r = fit_eval(kind, seed, X, A, Y, d, TUNE[ds])
            if r is not None:
                out.append((ds, src, kind, seed, r))
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(ds, seed) for ds in DATASETS for seed in SEEDS]
    results = {}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        for rows in ex.map(run_task, tasks):
            for ds, src, kind, seed, r in rows:
                results.setdefault(ds, {}).setdefault(src, {}).setdefault(kind, {})[seed] = r
    with open(os.path.join(OUT, "reuse_experiment.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    ms = lambda vals: f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
    texms = lambda vals: f"\\({np.mean(vals):.3f}\\pm{np.std(vals):.3f}\\)"
    md = ["| Dataset | Model | Training data | n | Acc | F1 | EO gap | DemP gap | EOD gap |", "|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for ds in DATASETS:
        for kind in KINDS:
            for src in ("Real", "Synthetic w/o EO", "Synthetic (ours)"):
                runs = results.get(ds, {}).get(src, {}).get(kind)
                if not runs:
                    continue
                v = list(runs.values())
                col = lambda k: [r[k] for r in v]
                md.append(f"| {ds} | {kind} | {src} | {len(v)} | {ms(col('accuracy'))} | {ms(col('F1'))} | "
                          f"{ms(col('EO_gap'))} | {ms(col('DemP_gap'))} | {ms(col('EOD_gap'))} |")
                tex.append(f"{'COMPAS' if ds == 'compas' else ds.capitalize()} & {kind} & {src} & {texms(col('accuracy'))} & "
                           f"{texms(col('EO_gap'))} & {texms(col('DemP_gap'))} & {texms(col('EOD_gap'))} \\\\")
        tex.append("\\midrule")
    with open(os.path.join(OUT, "reuse_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "table_reuse.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex[:-1]) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
