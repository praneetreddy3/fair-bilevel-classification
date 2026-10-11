"""
Matched comparison with Wu, Curtis & Robinson (synthetic data for fair collaborative learning),
using THEIR implementation (the local FairSynData/ checkout; gitignored, not our code) on OUR splits.

What is reused unchanged from FairSynData/mycodes:
  solve_inner_problem   (inner logistic fit on the synthetic data, L-BFGS)
  solve_nabla_P         (outer hypergradient wrt the synthetic features, EO covariance constraint)
  Adam                  (their optimizer for the synthetic features), AlgorithmParams defaults
Their client loop from main.py:single_client_training (stage 1, i.e. no CTGAN privacy stage, which
matches our no-DP main results) is replicated here step by step: synthetic features initialised at
the client's real features (N_s = N, "orig_x"/"as_Orig"), K = 20 outer iterations with their
gradient-norm stopping rule, rho_s = 0, fair_constr = 'EO', labels in {-1, 1}. The server fits a
logistic model on the union of the clients' synthetic data with the same inner solver, as in their
single communication round.

Matched to our experiments: the same clients, validation and test splits and the same 10 seeds
(scripts/fl_baselines.prepare_splits), our preprocessing (standardised features, intercept for
Credit/Law/COMPAS), the same evaluation (validation-tuned threshold for Credit/Law/COMPAS; accuracy,
EO, DemP, EOD at that threshold), and the same selection rule over their rho_o grid
{0, 10, 1e2, 1e3, 1e4}: max mean validation accuracy subject to mean validation EO <= 0.1.

Needs PyTorch and the FairSynData/ folder.
  python scripts/wu_baseline.py --data law --jobs 4
  python scripts/wu_baseline.py --jobs 4                    # all four datasets
  python scripts/wu_baseline.py --data adult --float64 --jobs 4                 # double precision
  python scripts/wu_baseline.py --data adult --float64 --adult_log1p --jobs 4   # + their Adult prep
  (option runs are saved to separate files, e.g. outputs/wu_baseline/adult_f64.json)
Writes outputs/wu_baseline/<ds>.json and outputs/wu_baseline/summary.md
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ["NO_DB"] = "1"          # FairSynData's built-in no-database mode
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIRSYN = os.path.join(ROOT, "FairSynData")
OUT_DIR = os.path.join(ROOT, "outputs", "wu_baseline")
SEEDS = list(range(1, 11))
RHO_GRID = [0.0, 10.0, 1e2, 1e3, 1e4]   # FairSynData's default rho_o_list


def _setup_paths():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    if FAIRSYN not in sys.path:
        sys.path.insert(0, FAIRSYN)


def run_one(args):
    """One (dataset, seed, rho_o): Wu et al. stage 1 on every client, then the server fit."""
    ds, seed, rho, opts = args
    _setup_paths()
    import torch
    import fl_baselines as fb
    from mycodes.myParams import AlgorithmParams, TrainAndPredictParams
    from mycodes.exactInnerMinOpt import solve_inner_problem
    from mycodes.solveNablaOuter import solve_nabla_P
    from mycodes.myadam import Adam
    import mycodes.myInit as myInit

    torch.set_num_threads(1)
    # Numerical options (do not change the algorithm): double precision, and the log1p transform of
    # Adult's heavy-tailed capital-gain / capital-loss columns that the FairSynData Adult setup uses
    # (reference_setup/rawdata/_prep_adult_credit.py).
    dt = torch.float64 if opts.get("float64") else torch.float32
    if opts.get("float64"):
        torch.set_default_dtype(torch.float64)
    cfg = fb.DATASET_CFG[ds]
    raw = fb.load_dataset(ds, cfg["sensitive"])
    if ds == "adult" and opts.get("adult_log1p"):
        (Xtr, Atr, Ytr), (Xte, Ate, Yte) = raw
        Xtr, Xte = Xtr.astype(np.float64).copy(), Xte.astype(np.float64).copy()
        # columns 3, 4 = capital-gain, capital-loss (numeric columns keep their order before the dummies)
        assert Xtr[:, 3].max() == 99999, "unexpected Adult column order"
        Xtr[:, 3:5], Xte[:, 3:5] = np.log1p(Xtr[:, 3:5]), np.log1p(Xte[:, 3:5])
        raw = ((Xtr, Atr, Ytr), (Xte, Ate, Yte))
    data = fb.prepare_splits(raw, seed, cfg["add_intercept"])

    def params():
        a = AlgorithmParams()
        a.dataset_name, a.fair_constr = ds, "EO"
        a.set_params(rho_o=rho, rho_s=0.0, constr_imposed_orig="Y" if rho > 0 else "N",
                     constr_imposed_syn="N", init_xhat="orig_x", init_shat_yhat="as_Orig",
                     init_theta="all_0")
        return a

    tp = TrainAndPredictParams()
    syn_X, syn_S, syn_Y, status = [], [], [], []
    for cid, (X, A, Y) in enumerate(data["clients"], start=1):
        # their set_seed(42) is called before each client's training
        torch.manual_seed(42); np.random.seed(42)
        alg = params()
        x = torch.tensor(X, dtype=dt)
        s = torch.tensor(A, dtype=dt)
        y = torch.tensor(np.where(Y == 1, 1.0, -1.0), dtype=dt)
        alg.set_params(N_s=x.shape[0])
        xhat, shat, yhat = x.clone().detach(), s.clone().detach(), y.clone().detach()
        theta = myInit.initialize_theta(xhat.shape[1] + 1, method=alg.init_theta)
        optimizer = Adam([xhat], lr=alg.alpha_xhat) if alg.is_alpha_xhat_adaptive else None
        norm0, st = None, "max_iter"
        for k in range(1, alg.K + 1):
            theta, _ = solve_inner_problem(1, 0, cid, k, xhat, shat, yhat, alg, tp, theta)
            grad, _, _ = solve_nabla_P(1, 0, cid, k, x, s, y, xhat, shat, yhat, theta, alg)
            ninf = float(torch.norm(grad.view(-1), p=float("inf")))
            if math.isnan(ninf):
                st = "nan"
                break
            if k == 1:
                norm0 = ninf
            tol = alg.outer_tol_xhat * max(1.0, norm0)
            if ninf > tol:
                if optimizer is not None:
                    optimizer.zero_grad()
                    xhat.grad = grad
                    optimizer.step()
                else:
                    xhat = xhat - alg.alpha_xhat * grad
            else:
                st = "tol"
                break
        status.append(st)
        if st == "nan":
            continue
        syn_X.append(xhat.detach().numpy()); syn_S.append(shat.numpy()); syn_Y.append(yhat.numpy())

    if not syn_X:
        return ds, seed, rho, None, status
    # server: one logistic fit on the union of the synthetic data (single communication round)
    torch.manual_seed(42)
    alg = params()
    Xs = torch.tensor(np.vstack(syn_X), dtype=dt)
    Ss = torch.tensor(np.concatenate(syn_S), dtype=dt)
    Ys = torch.tensor(np.concatenate(syn_Y), dtype=dt)
    theta0 = myInit.initialize_theta(Xs.shape[1] + 1, method="all_0")
    theta, _ = solve_inner_problem(1, 0, 0, 0, Xs, Ss, Ys, alg, tp, theta0)
    res = fb.evaluate(theta.numpy().astype(np.float64), data, cfg["tune_threshold"])
    return ds, seed, rho, res, status


def summarize(ds, tag=""):
    with open(os.path.join(OUT_DIR, f"{ds}{tag}.json")) as fh:
        out = json.load(fh)
    rows = []
    for rho, runs in out["grid"].items():
        ok = [r for r in runs.values() if r is not None]
        if len(ok) != len(SEEDS):
            continue
        nan = sum(st.count("nan") for st in out["status"].get(rho, {}).values())
        rows.append({"rho": float(rho), "runs": ok, "nan_clients": nan,
                     "val_acc": float(np.mean([r["val_accuracy"] for r in ok])),
                     "val_eo": float(np.mean([r["val_EO_gap"] for r in ok]))})
    feas = [r for r in rows if r["val_eo"] <= 0.1]
    sel = max(feas, key=lambda r: r["val_acc"]) if feas else min(rows, key=lambda r: r["val_eo"])
    m = lambda k, ext=False: (float(np.mean([(r.get("extended_metrics_thr", r["extended_metrics"])[k] if ext else r[k]) for r in sel["runs"]])),
                              float(np.std([(r.get("extended_metrics_thr", r["extended_metrics"])[k] if ext else r[k]) for r in sel["runs"]])))
    return sel["rho"], {"acc": m("accuracy"), "f1": m("F1_score"), "eo": m("EO_gap"),
                        "dp": m("DP_gap", True), "eod": m("EOD_gap", True),
                        "nan_clients": sel["nan_clients"]}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["credit", "adult", "law", "compas", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--float64", action="store_true", help="run their solver in double precision")
    ap.add_argument("--adult_log1p", action="store_true",
                    help="log1p of Adult capital-gain/loss, as in the FairSynData Adult setup")
    a = ap.parse_args()
    opts = {"float64": a.float64, "adult_log1p": a.adult_log1p}
    tag = ("_f64" if a.float64 else "") + ("_log1p" if a.adult_log1p else "")   # separate result files
    if not os.path.isdir(os.path.join(FAIRSYN, "mycodes")):
        sys.exit(f"FairSynData/ not found at {FAIRSYN}")
    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = ["credit", "adult", "law", "compas"] if a.data == "all" else [a.data]
    for ds in datasets:
        path = os.path.join(OUT_DIR, f"{ds}{tag}.json")
        out = json.load(open(path)) if os.path.isfile(path) else {"dataset": ds, "options": opts, "grid": {}, "status": {}}
        todo = [(ds, s, r, opts) for r in RHO_GRID for s in SEEDS
                if str(s) not in out["grid"].get(str(r), {})]
        print(f"[{ds}] {len(todo)} runs to do", flush=True)
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for d, s, r, res, st in ex.map(run_one, todo):
                out["grid"].setdefault(str(r), {})[str(s)] = res
                out["status"].setdefault(str(r), {})[str(s)] = st
                with open(path, "w") as fh:          # save after every run (safe to stop and resume)
                    json.dump(out, fh, indent=1)
                print(f"  done {d} seed {s} rho_o {r:g}  clients: {st}", flush=True)
    lines = [f"Options: {opts}",
             "| Dataset | selected rho_o | Acc | F1 | EO gap | DemP gap | EOD gap | diverged clients (of 50) |",
             "|---|---|---|---|---|---|---|---|"]
    tex = []
    for ds in ["credit", "adult", "law", "compas"]:
        if not os.path.isfile(os.path.join(OUT_DIR, f"{ds}{tag}.json")):
            continue
        rho, s, rows = summarize(ds, tag)
        f = lambda p: f"{p[0]:.3f}±{p[1]:.3f}"
        lines.append(f"| {ds} | {rho:g} | {f(s['acc'])} | {f(s['f1'])} | {f(s['eo'])} | {f(s['dp'])} | {f(s['eod'])} | {s['nan_clients']} |")
        name = "COMPAS" if ds == "compas" else ds.capitalize()
        t = lambda p: f"\\({p[0]:.3f}\\pm{p[1]:.3f}\\)"
        tex.append(f"{name} & Wu et al. & {t(s['acc'])} & {t(s['f1'])} & {t(s['eo'])} & {t(s['dp'])} & {t(s['eod'])} \\\\")
        for r in rows:
            lines.append(f"|   {ds} grid | rho_o={r['rho']:g} | val acc {r['val_acc']:.3f} | val EO {r['val_eo']:.3f} | | | | {r['nan_clients']} |")
    text = "\n".join(lines)
    open(os.path.join(OUT_DIR, f"summary{tag}.md"), "w", encoding="utf-8").write(text + "\n")
    open(os.path.join(OUT_DIR, f"table_rows{tag}.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print("\n" + text + f"\n\nLaTeX rows (one per dataset) in outputs/wu_baseline/table_rows{tag}.tex")


if __name__ == "__main__":
    main()
