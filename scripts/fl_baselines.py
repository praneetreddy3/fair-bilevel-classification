"""
Fair federated-learning baselines for the paper comparison: FedAvg, FairFed, FedFB, FedFair.

All four are re-implemented here for the SAME setting as our method (draft_model/run_draft.py):
  - same data loaders, same train/val/test split, same scaling/intercept, same IID client split
    (the numpy RNG calls below replay run_draft.py's exact order, so for a given --seed every
    method sees identical client data),
  - same linear logistic model f(x,a) = theta^T [x; a], same ridge term,
  - K=5 clients, T=8 communication rounds (matched communication budget),
  - same evaluation code (mirrors draft_model/server.py; copied here so this script needs only
    numpy/sklearn, no torch),
  - same validation-only model selection rule as scripts/fair_comparison.py: maximize mean
    validation accuracy among configs with mean validation EO_gap <= 0.1; if none qualify,
    minimize mean validation EO_gap. The test split is never used for selection.

Methods (EO fairness notion throughout; S = sensitive attribute, Y = label):
  FedAvg  McMahan et al. 2017. Clients run local Adam on the logistic loss; the server averages
          parameters weighted by client size. No fairness mechanism (reference point).
  FairFed Ezzeldin et al., AAAI 2023. Local training as FedAvg; the server re-weights clients by
          how far each client's local EO gap is from the global EO gap:
              w_k <- w_k - beta * (Delta_k - mean(Delta)),   Delta_k = |F_global - F_k|,
          with F_global estimated from aggregated client statistics (same computation as the
          reference implementation in FairSynData/mycodes/myFLAlg.py). Knob: beta.
  FedFB   Zeng et al. 2021 (FairBatch + FedAvg). Clients minimise a re-weighted loss in which the
          two qualified (Y=1) groups get coefficients lambda_0, lambda_1; clients report their
          per-group loss sums, the server aggregates them and updates
              lambda_a <- lambda_a + alpha * mu_a / ||mu||,  mu_1 = L_{1,1} - L_{0,1}, mu_0 = -mu_1,
          so the group with the higher positive-class loss gets more weight. Knob: alpha.
  FedFair Che et al., IEEE BigData 2024. Federated DGEO constraint on the positive class,
          |(1/N) sum_i (L_i^{0,1} - L_i^{1,1})| <= eps, solved as a min-max Lagrangian: clients
          descend loss + (lam_a - lam_b) * D_i; the server updates the multipliers with the
          paper's projected ascent rule (eqs. 12-13, gamma = 0.001). Because we allow only T=8
          communication rounds, the multiplier step size is part of the grid. Knobs: eps, step.

These are our re-implementations adapted to the shared linear-logistic setting, not the
authors' released code; the docstring of each function states exactly what is implemented.

Run from project root:  python scripts/fl_baselines.py            (all datasets)
                        python scripts/fl_baselines.py --data law  (one dataset)
Writes outputs/fl_baselines/<dataset>.json (full grid + selected configs) and prints a table.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "fl_baselines")

# Same per-dataset settings as scripts/run_final.sh / docs/RESULTS.md.
DATASET_CFG = {
    "credit": {"sensitive": "sex", "add_intercept": True, "tune_threshold": True},
    "adult": {"sensitive": "sex", "add_intercept": False, "tune_threshold": False},
    "law": {"sensitive": "race", "add_intercept": True, "tune_threshold": True},
    "compas": {"sensitive": "race", "add_intercept": True, "tune_threshold": True},
}
SEEDS = [1, 2, 3, 4, 5]
NUM_CLIENTS = 5
ROUNDS = 8
LOCAL_STEPS = 50          # local Adam steps per round (8 x 50 = 400 steps, ~ the 500-step baseline)
LR = 0.05                 # same learning rate as train_global_ridge_erm
LAMBDA_THETA = 1e-4       # same ridge weight as the baseline
VAL_FRAC = 0.2
EO_TARGET = 0.1           # same selection target as scripts/fair_comparison.py

GRIDS = {
    "FedAvg": [{}],
    "FairFed": [{"beta": b} for b in (0.5, 1.0, 2.0, 5.0, 10.0)],
    "FedFB": [{"alpha": a} for a in (0.005, 0.01, 0.05, 0.1, 0.2, 0.5)],
    # step 0.05 is the paper's multiplier step size; larger steps compensate for only T=8 rounds.
    "FedFair": [{"eps": e, "step": s} for e in (0.0, 0.02, 0.05) for s in (0.05, 0.5, 2.0, 5.0)],
}


# --------------------------------------------------------------------------- data
def load_dataset(name, sensitive):
    if name == "credit":
        from pipeline.load_credit import prepare_credit_for_draft
        return prepare_credit_for_draft(sensitive=sensitive)
    if name == "adult":
        from pipeline.load_adult import prepare_adult_for_draft
        return prepare_adult_for_draft(sensitive=sensitive, use_ucimlrepo=True)
    if name == "law":
        from pipeline.load_law import prepare_law_for_draft
        return prepare_law_for_draft(sensitive=sensitive)
    if name == "compas":
        from pipeline.load_compas import prepare_compas_for_draft
        return prepare_compas_for_draft(sensitive=sensitive)
    raise ValueError(name)


def prepare_splits(raw, seed, add_intercept):
    """Replays run_draft.py: val split -> scaling -> intercept -> IID client split, same RNG order."""
    (X_train, A_train, Y_train), (X_test, A_test, Y_test) = raw
    rng = np.random.default_rng(seed)
    n = len(Y_train)
    n_val = int(n * VAL_FRAC)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    val_idx, train_idx = all_idx[:n_val], all_idx[n_val:]
    X_val, A_val, Y_val = X_train[val_idx], A_train[val_idx], Y_train[val_idx]
    X_tr, A_tr, Y_tr = X_train[train_idx], A_train[train_idx], Y_train[train_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.astype(np.float64))
    X_val = scaler.transform(X_val.astype(np.float64))
    X_te = scaler.transform(X_test.astype(np.float64))
    if add_intercept:
        X_tr = np.hstack([X_tr, np.ones((len(X_tr), 1))])
        X_val = np.hstack([X_val, np.ones((len(X_val), 1))])
        X_te = np.hstack([X_te, np.ones((len(X_te), 1))])

    indices = np.arange(len(Y_tr))
    rng.shuffle(indices)
    splits = np.array_split(indices, NUM_CLIENTS)
    clients = [(X_tr[s], A_tr[s].astype(np.float64), Y_tr[s].astype(np.float64)) for s in splits]
    return {
        "clients": clients,
        "train": (X_tr, A_tr.astype(np.float64), Y_tr.astype(np.float64)),
        "val": (X_val, A_val.astype(np.float64), Y_val.astype(np.float64)),
        "test": (X_te, A_test.astype(np.float64), Y_test.astype(np.float64)),
    }


def xa(X, A):
    return np.hstack([X, A.reshape(-1, 1)])


# --------------------------------------------------------------------------- model / loss
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def per_sample_loss(Z, y, theta):
    """Logistic loss log(1+exp(-(2y-1) f)) per sample, same clamp as losses.py."""
    m = (2 * y - 1) * (Z @ theta)
    return np.log1p(np.exp(-np.clip(m, -50, None)))


def per_sample_grad_coef(Z, y, theta):
    """d loss_i / d f_i for the logistic loss."""
    m = (2 * y - 1) * (Z @ theta)
    return -(2 * y - 1) * sigmoid(-m)


def ridge_scale(d_plus_1):
    return LAMBDA_THETA / (d_plus_1 ** 2)  # gradient of lambda/(2(d+1)^2)||theta||^2


def local_adam(theta0, Z, y, sample_w, steps=LOCAL_STEPS, lr=LR):
    """Local client training: Adam (torch defaults) on the sample-weighted mean logistic loss + ridge.
    sample_w has mean-normalised weights (all ones = plain ERM)."""
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    reg = ridge_scale(len(theta))
    n = len(y)
    for t in range(1, steps + 1):
        coef = per_sample_grad_coef(Z, y, theta) * sample_w
        g = Z.T @ coef / n + reg * theta
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        theta -= lr * mh / (np.sqrt(vh) + eps)
    return theta


# --------------------------------------------------------------------------- evaluation (mirrors server.py)
def eo_gap_and_accuracy(theta, X, A, Y, threshold=0.0):
    logits = xa(X, A) @ theta
    pred = (logits > threshold).astype(np.float64)
    acc = float(np.mean(pred == Y))
    tp = np.sum((pred == 1) & (Y == 1)); fp = np.sum((pred == 1) & (Y == 0)); fn = np.sum((pred == 0) & (Y == 1))
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0
    mask = Y == 1
    s_pos, p_pos = A[mask], pred[mask]
    tpr0 = float(np.mean(p_pos[s_pos == 0])) if np.any(s_pos == 0) else 0.0
    tpr1 = float(np.mean(p_pos[s_pos == 1])) if np.any(s_pos == 1) else 0.0
    return abs(tpr1 - tpr0), tpr0, tpr1, acc, f1


def pick_threshold(theta, X, A, Y):
    logits = xa(X, A) @ theta
    if len(np.unique(Y)) < 2:
        return 0.0
    best_t, best = 0.0, -1.0
    P, N = Y == 1, Y == 0
    for t in np.quantile(logits, np.linspace(0.02, 0.98, 49)):
        pred = logits > t
        score = 0.5 * (np.mean(pred[P]) + np.mean(~pred[N]))
        if score > best:
            best, best_t = score, float(t)
    return best_t


def extended_metrics(theta, X, A, Y):
    logits = xa(X, A) @ theta
    probs = sigmoid(logits)
    pred = (logits > 0).astype(np.float64)
    out = {
        "PR_AUC": float(average_precision_score(Y, probs)),
        "ROC_AUC": float(roc_auc_score(Y, probs)),
        "macro_F1": float(f1_score(Y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(Y, pred)),
    }
    p1 = np.mean(pred[A == 1]); p0 = np.mean(pred[A == 0])
    out["DP_gap"] = float(abs(p1 - p0))
    gaps = []
    for yv in (0, 1):
        mk = Y == yv
        a, p = A[mk], pred[mk]
        gaps.append(abs(np.mean(p[a == 1]) - np.mean(p[a == 0])))
    out["EOD_gap"] = float(max(gaps))
    return out


# --------------------------------------------------------------------------- client statistics
def local_eo_stats(theta, Z, A, Y):
    """Counts / hard TPRs on a client's data under theta (FairFed statistics)."""
    pred = (Z @ theta > 0).astype(np.float64)
    out = {"n": len(Y)}
    for s in (0, 1):
        mk = (A == s) & (Y == 1)
        out[f"cnt_S{s}Y1"] = int(mk.sum())
        out[f"tpr_S{s}"] = float(pred[mk].mean()) if mk.any() else 0.0
    out["acc"] = float(np.mean(pred == Y))
    return out


def group_positive_losses(theta, Z, A, Y):
    """Sum of logistic loss and count over (S=s, Y=1) for s in {0,1} (FedFB / FedFair statistics)."""
    loss = per_sample_loss(Z, Y, theta)
    res = {}
    for s in (0, 1):
        mk = (A == s) & (Y == 1)
        res[s] = (float(loss[mk].sum()), int(mk.sum()))
    return res


# --------------------------------------------------------------------------- methods
def run_method(method, cfg, data):
    clients = data["clients"]
    d1 = clients[0][0].shape[1] + 1
    theta = np.zeros(d1)
    Zs = [xa(X, A) for X, A, _ in clients]
    sizes = np.array([len(Y) for _, _, Y in clients], dtype=float)
    size_w = sizes / sizes.sum()

    # global positive-group counts (aggregated once; needed by FedFB/FedFair weighting)
    n_pos = {s: sum(int(((A == s) & (Y == 1)).sum()) for _, A, Y in clients) for s in (0, 1)}
    n_pos_total = n_pos[0] + n_pos[1]

    fairfed_w = size_w.copy()
    fedfb_lam = np.array([n_pos[0] / n_pos_total, n_pos[1] / n_pos_total])
    lam_a = lam_b = 0.0

    for t in range(ROUNDS):
        local_thetas = []
        for k, (X, A, Y) in enumerate(clients):
            Z = Zs[k]
            w = np.ones(len(Y))
            if method == "FedFB":
                # Loss = sum_a lambda_a * (n_pos/n) * mean_{a,1} loss + (n_neg/n) * mean_{Y=0} loss,
                # written as per-sample weights (global counts so weighting is consistent across clients).
                for s in (0, 1):
                    mk = (A == s) & (Y == 1)
                    w[mk] = fedfb_lam[s] * n_pos_total / max(n_pos[s], 1)
            elif method == "FedFair":
                # d/dtheta of (lam_a - lam_b) * D_i with D_i = L_i^{0,1} - L_i^{1,1}; the constraint
                # uses the plain average over the N clients, so client k's term is scaled by
                # (1/N) / size_w[k] to sit alongside its size-weighted loss.
                c = (lam_a - lam_b) * (1.0 / NUM_CLIENTS) / size_w[k]
                for s, sign in ((0, 1.0), (1, -1.0)):
                    mk = (A == s) & (Y == 1)
                    if mk.any():
                        w[mk] += c * sign * len(Y) / mk.sum()
            local_thetas.append(local_adam(theta, Z, Y, w))

        if method == "FairFed":
            stats = [local_eo_stats(th, Zs[k], clients[k][1], clients[k][2]) for k, th in enumerate(local_thetas)]
            tot = sum(s["n"] for s in stats)
            tot0 = sum(s["cnt_S0Y1"] for s in stats) / tot
            tot1 = sum(s["cnt_S1Y1"] for s in stats) / tot
            f_glob = 0.0
            for s in stats:
                wk = s["n"] / tot
                term0 = (s["tpr_S0"] * s["cnt_S0Y1"] / s["n"]) / tot0 if tot0 > 0 else 0.0
                term1 = (s["tpr_S1"] * s["cnt_S1Y1"] / s["n"]) / tot1 if tot1 > 0 else 0.0
                f_glob += wk * (term0 - term1)
            deltas = np.array([abs(f_glob - (s["tpr_S0"] - s["tpr_S1"])) for s in stats])
            fairfed_w = fairfed_w - cfg["beta"] * (deltas - deltas.mean())
            fairfed_w = np.clip(fairfed_w, 1e-6, None)
            fairfed_w = fairfed_w / fairfed_w.sum()
            theta = sum(fairfed_w[k] * th for k, th in enumerate(local_thetas))
        else:
            theta = sum(size_w[k] * th for k, th in enumerate(local_thetas))

        if method == "FedFB":
            sums = {0: 0.0, 1: 0.0}
            for k, (X, A, Y) in enumerate(clients):
                g = group_positive_losses(theta, Zs[k], A, Y)
                sums[0] += g[0][0]; sums[1] += g[1][0]
            L0, L1 = sums[0] / max(n_pos[0], 1), sums[1] / max(n_pos[1], 1)
            mu = np.array([-(L1 - L0), (L1 - L0)])
            norm = np.linalg.norm(mu)
            if norm > 0:
                fedfb_lam = fedfb_lam + cfg["alpha"] * mu / norm
            fedfb_lam = np.clip(fedfb_lam, 0.0, 1.0)
            fedfb_lam = fedfb_lam / fedfb_lam.sum()
        elif method == "FedFair":
            Ds = []
            for k, (X, A, Y) in enumerate(clients):
                g = group_positive_losses(theta, Zs[k], A, Y)
                l0 = g[0][0] / g[0][1] if g[0][1] else 0.0
                l1 = g[1][0] / g[1][1] if g[1][1] else 0.0
                Ds.append(l0 - l1)
            D = float(np.mean(Ds))
            gamma, beta, eps = 0.001, cfg["step"], cfg["eps"]
            lam_a = max((1 - gamma * beta) * lam_a + beta * D - beta * eps, 0.0)
            lam_b = max((1 - gamma * beta) * lam_b - beta * D - beta * eps, 0.0)
    return theta


def evaluate(theta, data, tune):
    Xv, Av, Yv = data["val"]
    Xt, At, Yt = data["test"]
    val_eo, _, _, val_acc, _ = eo_gap_and_accuracy(theta, Xv, Av, Yv)  # same as round_logs[-1]
    thr = pick_threshold(theta, Xv, Av, Yv) if tune else 0.0
    eo, tpr0, tpr1, acc, f1 = eo_gap_and_accuracy(theta, Xt, At, Yt, threshold=thr)
    return {
        "val_accuracy": val_acc, "val_EO_gap": val_eo,
        "accuracy": acc, "F1_score": f1, "EO_gap": eo, "TPR_group0": tpr0, "TPR_group1": tpr1,
        "threshold": thr, "extended_metrics": extended_metrics(theta, Xt, At, Yt),
    }


def baseline_check(data, tune):
    """Centralised ridge ERM (500 Adam steps) -- reproduces run_draft.py's 'Baseline' to confirm the
    splits match (compare against outputs/draft_results_<ds>_final_seed<k>.json['baseline'])."""
    X, A, Y = data["train"]
    theta = local_adam(np.zeros(X.shape[1] + 1), xa(X, A), Y, np.ones(len(Y)), steps=500)
    return evaluate(theta, data, tune)


def select(rows):
    feas = [r for r in rows if r["mean_val_EO"] <= EO_TARGET]
    if feas:
        return max(feas, key=lambda r: r["mean_val_acc"])
    return min(rows, key=lambda r: r["mean_val_EO"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATASET_CFG) + ["all"], default="all")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = list(DATASET_CFG) if args.data == "all" else [args.data]

    for ds in datasets:
        cfg_ds = DATASET_CFG[ds]
        t0 = time.time()
        raw = load_dataset(ds, cfg_ds["sensitive"])
        per_seed = {seed: prepare_splits(raw, seed, cfg_ds["add_intercept"]) for seed in SEEDS}
        out = {"dataset": ds, "settings": {"num_clients": NUM_CLIENTS, "rounds": ROUNDS,
               "local_steps": LOCAL_STEPS, "lr": LR, "lambda_theta": LAMBDA_THETA,
               "eo_target": EO_TARGET, **cfg_ds}, "baseline_check": {}, "methods": {}}
        for seed in SEEDS:
            out["baseline_check"][seed] = baseline_check(per_seed[seed], cfg_ds["tune_threshold"])

        for method, grid in GRIDS.items():
            rows = []
            for cfg in grid:
                runs = {seed: evaluate(run_method(method, cfg, per_seed[seed]), per_seed[seed],
                                       cfg_ds["tune_threshold"]) for seed in SEEDS}
                rows.append({
                    "config": cfg, "runs": runs,
                    "mean_val_acc": float(np.mean([r["val_accuracy"] for r in runs.values()])),
                    "mean_val_EO": float(np.mean([r["val_EO_gap"] for r in runs.values()])),
                })
            chosen = select(rows)
            out["methods"][method] = {"grid": rows, "selected_config": chosen["config"]}
        with open(os.path.join(OUT_DIR, f"{ds}.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{ds}] done in {time.time() - t0:.1f}s -> outputs/fl_baselines/{ds}.json")
    print_summary(datasets)


def summarize_runs(runs):
    vals = list(runs.values())
    def ms(key, ext=False):
        xs = [v["extended_metrics"][key] if ext else v[key] for v in vals]
        return float(np.mean(xs)), float(np.std(xs))
    return {"acc": ms("accuracy"), "f1": ms("F1_score"), "eo": ms("EO_gap"),
            "dp": ms("DP_gap", True), "eod": ms("EOD_gap", True),
            "bal_acc": ms("balanced_accuracy", True), "pr_auc": ms("PR_AUC", True)}


def ours_runs(ds):
    """Our method's reported runs (outputs/draft_results_<ds>_final_seed<k>.json), same seeds."""
    runs = {}
    for seed in SEEDS:
        p = os.path.join(PROJECT_ROOT, "outputs", f"draft_results_{ds}_final_seed{seed}.json")
        if os.path.isfile(p):
            with open(p) as fh:
                runs[seed] = json.load(fh)["pipeline"]
    return runs


def print_summary(datasets):
    f = lambda p: f"{p[0]:.3f}±{p[1]:.3f}"
    tex = lambda p: f"\\({p[0]:.3f}\\pm{p[1]:.3f}\\)"
    header = f"{'Dataset':<8}{'Method':<9}{'config':<28}{'Acc':<14}{'F1':<14}{'EO':<14}{'DemP':<14}{'EOD':<14}"
    lines, md, tex_rows = [header], ["| Dataset | Method | Selected config | Acc | F1 | EO gap | DemP gap | EOD gap |",
                                     "|---|---|---|---|---|---|---|---|"], []
    for ds in datasets:
        with open(os.path.join(OUT_DIR, f"{ds}.json")) as fh:
            out = json.load(fh)
        entries = []
        for method, info in out["methods"].items():
            row = next(r for r in info["grid"] if r["config"] == info["selected_config"])
            entries.append((method, json.dumps(info["selected_config"]), summarize_runs(row["runs"])))
        ours = ours_runs(ds)
        if len(ours) == len(SEEDS):
            entries.append(("Ours", "reported config", summarize_runs(ours)))
        for method, cfg, s in entries:
            lines.append(f"{ds:<8}{method:<9}{cfg:<28}{f(s['acc']):<14}{f(s['f1']):<14}"
                         f"{f(s['eo']):<14}{f(s['dp']):<14}{f(s['eod']):<14}")
            md.append(f"| {ds} | {method} | {cfg} | {f(s['acc'])} | {f(s['f1'])} | {f(s['eo'])} | "
                      f"{f(s['dp'])} | {f(s['eod'])} |")
            tex_rows.append(f"{ds.capitalize() if ds != 'compas' else 'COMPAS'} & {method} & {tex(s['acc'])} & "
                            f"{tex(s['f1'])} & {tex(s['eo'])} & {tex(s['dp'])} & {tex(s['eod'])} \\\\")
        tex_rows.append("\\midrule")
    print("\n" + "\n".join(lines))
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_DIR, "table_fl_baselines.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex_rows[:-1]) + "\n")


if __name__ == "__main__":
    main()
