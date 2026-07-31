"""
Main pipeline runner: data → clients → minibatch → synthetic → Universum → bilevel AL → server.
Evaluates both a baseline (ERM) and the fairness-aware pipeline on the test set.

Usage: python -m draft_model.run_draft --data {dummy,adult,2d} [options]
"""
import os
# Determinism: pin BLAS / OpenMP thread pools to a single thread BEFORE numpy or torch are
# imported. Multi-threaded floating-point reduction order is the main source of CPU
# non-determinism, and these env vars only take effect if set before the libraries load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from draft_model.notation import ClientOriginalData, Payload, SyntheticMinibatch, UniversumSet
from draft_model.minibatch_design import (
    draw_original_minibatch, build_synthetic_templates, build_universum_templates,
)
from draft_model.bilevel_al import client_round_simplified, client_round_al
from draft_model.dp import DP_VARIANTS, apply_pre_server_dp, resolve_dp_config
from draft_model.server import (
    aggregate_payloads, train_global_ridge_erm, compute_eo_gap_and_accuracy,
    compute_extended_metrics, pick_threshold,
)


def str2bool(v):
    """Parse bool-like CLI values while still supporting bare flags."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def load_dummy_law_style(project_root: str) -> tuple:
    """Load FairSynData dummy_run data as (X, A, Y) with Y ∈ {0,1}."""
    path_train = os.path.join(project_root, "FairSynData", "datasets", "dummy_run", "law_dummy_train_1.csv")
    path_test = os.path.join(project_root, "FairSynData", "datasets", "dummy_run", "law_dummy_test_1.csv")
    if not os.path.isfile(path_train):
        raise FileNotFoundError(f"Dummy data not found: {path_train}. Run make_dummy_data.py in FairSynData first.")
    import pandas as pd
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    feat_cols = [c for c in train.columns if c not in ("race", "pass_bar")]
    X_train = train[feat_cols].values
    A_train = train["race"].values.astype(np.int64)
    Y_train = (train["pass_bar"].values == 1).astype(np.float64)
    X_test = test[feat_cols].values
    A_test = test["race"].values.astype(np.int64)
    Y_test = (test["pass_bar"].values == 1).astype(np.float64)
    return (X_train, A_train, Y_train), (X_test, A_test, Y_test)


def load_adult_data(sensitive: str = "sex"):
    """Load UCI Adult: prefers ``UCIAdultdataset/adult.data`` + ``adult.test``, else ucimlrepo."""
    from pipeline.load_adult import prepare_adult_for_draft
    return prepare_adult_for_draft(sensitive=sensitive, use_ucimlrepo=True)


def load_2d_data(project_root: str):
    """Load 2D example data from outputs/2d_data.csv."""
    from pipeline.load_2d import load_2d_for_draft
    return load_2d_for_draft(project_root)


def load_credit_data(sensitive: str = "sex"):
    """Load 'Default of Credit Card Clients' (UCI id 350). Put the file in CreditData/."""
    from pipeline.load_credit import prepare_credit_for_draft
    return prepare_credit_for_draft(sensitive=sensitive)


def load_law_data(sensitive: str = "race"):
    """Load the Law School dataset (reference paper's own dataset). Sensitive = race."""
    from pipeline.load_law import prepare_law_for_draft
    return prepare_law_for_draft(sensitive=sensitive)


def dirichlet_partition_indices(Y: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator) -> list:
    """Label-skew non-IID client partition (standard FL recipe, e.g. Hsu et al. 2019).

    For each class, split its indices across clients according to proportions
    drawn from Dirichlet(alpha); smaller alpha => more label skew per client.
    """
    client_indices = [np.array([], dtype=int) for _ in range(num_clients)]
    for c in np.unique(Y):
        idx_c = np.where(Y == c)[0].copy()
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx_c, cuts)):
            client_indices[k] = np.concatenate([client_indices[k], part])
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
    return client_indices


def main():
    """End-to-end run: load data -> split train/val/test -> per-round client loop
    (minibatch -> synthetic -> Universum -> bilevel AL) -> server aggregation (+ optional DP)
    -> evaluate baseline (plain ERM) vs pipeline (fairness-constrained) on the test set ->
    write results JSON to out_dir.
    """
    parser = argparse.ArgumentParser(description="Run the fair bilevel pipeline.")
    parser.add_argument("--data", default="dummy", choices=["dummy", "adult", "2d", "credit", "law"])
    parser.add_argument("--sensitive", default="sex", choices=["sex", "race"])
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default="iid",
                         help="Client data partition: iid (random shuffle) or dirichlet (label-skew non-IID)")
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0,
                         help="Concentration for --partition dirichlet; smaller = more label skew")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--K_inner", type=int, default=50)
    parser.add_argument("--no_full_al", action="store_true", help="Use simplified inner-only path")
    parser.add_argument("--J_outer", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--epsilon_EO", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--results_file",
        default="draft_results.json",
        help="Results JSON filename under out_dir (e.g. draft_results_adult.json)",
    )
    parser.add_argument("--no_universum", action="store_true", help="Ablation: disable Universum")
    parser.add_argument("--fairness_off", action="store_true", help="Ablation: set ρ=0")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument("--add_intercept", type=str2bool, default=False,
                         help="Append a constant bias feature so the linear model has an intercept "
                              "(fixes calibration/accuracy on imbalanced data e.g. credit).")
    parser.add_argument("--tune_threshold", type=str2bool, default=False,
                         help="Calibrate the decision threshold on the validation split "
                              "(maximise balanced accuracy) and apply it to the test set.")
    parser.add_argument("--use_tpr_gap", type=str2bool, default=True)
    parser.add_argument("--tpr_alpha", type=float, default=10.0)
    parser.add_argument("--tpr_tau", type=float, default=0.0)
    parser.add_argument("--ema_beta", type=float, default=0.15)
    parser.add_argument("--use_importance_weighting", type=str2bool, default=True)
    parser.add_argument("--use_rolling_buffer", type=str2bool, default=True)
    parser.add_argument("--dp_enabled", type=str2bool, default=True)
    parser.add_argument("--dp_sigma", type=float, default=1.0)
    parser.add_argument("--dp_variant", choices=DP_VARIANTS, default="post_server")
    parser.add_argument("--stop_criterion", choices=["eo_gap", "grad_inf"], default="eo_gap")
    parser.add_argument("--outer_tol_xhat", type=float, default=1e-6)
    parser.add_argument("--deterministic", type=str2bool, default=True,
                         help="Pin single-thread + deterministic torch ops so identical seeds give "
                              "identical results (fixes pipeline non-determinism). Set false for speed.")
    args = parser.parse_args()

    # Reproducibility: seed every RNG and pin PyTorch's execution so the bilevel pipeline is
    # bit-reproducible for a given --seed (baseline was already deterministic; the outer
    # feature-update loop was not, due to multi-threaded float reduction order).
    import random
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dp_cfg = resolve_dp_config(
        dp_enabled=args.dp_enabled,
        dp_sigma=args.dp_sigma,
        dp_variant=args.dp_variant,
    )

    # Load data
    if args.data == "dummy":
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = load_dummy_law_style(PROJECT_ROOT)
    elif args.data == "adult":
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = load_adult_data(sensitive=args.sensitive)
    elif args.data == "2d":
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = load_2d_data(PROJECT_ROOT)
    elif args.data == "credit":
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = load_credit_data(sensitive=args.sensitive)
    elif args.data == "law":
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = load_law_data(sensitive=args.sensitive)
    else:
        raise ValueError(args.data)

    # Validation split
    n = len(Y_train)
    n_val = int(n * args.val_frac)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    val_idx, train_idx = all_idx[:n_val], all_idx[n_val:]
    X_val, A_val, Y_val = X_train[val_idx], A_train[val_idx], Y_train[val_idx]
    X_train, A_train, Y_train = X_train[train_idx], A_train[train_idx], Y_train[train_idx]
    n = len(Y_train)

    # Feature scaling
    if not args.no_scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.astype(np.float64))
        X_val = scaler.transform(X_val.astype(np.float64))
        X_test = scaler.transform(X_test.astype(np.float64))

    # Optional intercept: append a constant 1 feature AFTER scaling (so it is not
    # zeroed out by standardisation). Gives the linear model f(x,a)=θᵀ[x;a] a bias term.
    if args.add_intercept:
        X_train = np.hstack([X_train, np.ones((len(X_train), 1))])
        X_val = np.hstack([X_val, np.ones((len(X_val), 1))])
        X_test = np.hstack([X_test, np.ones((len(X_test), 1))])

    d = X_train.shape[1]
    d_plus_1 = d + 1
    zeta = np.zeros(d_plus_1)

    # Split data across clients
    if args.partition == "dirichlet":
        splits = dirichlet_partition_indices(Y_train, args.num_clients, args.dirichlet_alpha, rng)
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        splits = np.array_split(indices, args.num_clients)
    clients_data = [
        ClientOriginalData(X=X_train[s], A=A_train[s], Y=Y_train[s])
        for s in splits
    ]

    use_full_al = not args.no_full_al
    theta_glob = zeta.copy()
    rho = 0.0 if args.fairness_off else args.rho
    round_logs = []

    for t in range(args.rounds):
        round_payloads = []
        for k, data in enumerate(clients_data):
            B = draw_original_minibatch(data, rng)
            Ds = build_synthetic_templates(B, Ds_size=32, rng=rng)

            if args.no_universum:
                U = UniversumSet(X=np.zeros((0, d)), A=np.zeros(0))
            else:
                U = build_universum_templates(
                    Ds.Delta_s, Ds.size(), B.Delta, d, rng=rng,
                    q_a0y1=B.q_a0y1, q_a1y1=B.q_a1y1, B=B,
                )

            if use_full_al:
                theta_k, X_ds_new, X_u_new = client_round_al(
                    B, Ds, U, zeta,
                    lambda_theta_in=1e-4, lambda_theta_out=1e-4,
                    lambda_U=0.5, rho=rho, epsilon_EO=args.epsilon_EO,
                    K_inner=args.K_inner, J_outer=args.J_outer,
                    eta_theta=0.05, eta_x=0.02, R=10.0,
                    seed=args.seed + t * 1000 + k,
                    use_tpr_gap=args.use_tpr_gap,
                    tpr_alpha=args.tpr_alpha,
                    tpr_tau=args.tpr_tau,
                    ema_beta=args.ema_beta,
                    use_importance_weighting=args.use_importance_weighting,
                    use_rolling_buffer=args.use_rolling_buffer,
                    stop_criterion=args.stop_criterion,
                    outer_tol_xhat=args.outer_tol_xhat,
                )
                Ds_send = SyntheticMinibatch(
                    X=X_ds_new, A=Ds.A, Y=Ds.Y,
                    q_a0y0=Ds.q_a0y0, q_a0y1=Ds.q_a0y1,
                    q_a1y0=Ds.q_a1y0, q_a1y1=Ds.q_a1y1,
                    Delta_s=Ds.Delta_s,
                )
                U_send = UniversumSet(X=X_u_new, A=U.A)
                round_payloads.append(Payload(synthetic=Ds_send, universum=U_send))
            else:
                # Simplified path only fits theta locally to sanity-check Ds/U construction;
                # the fitted theta isn't sent anywhere (the server retrains from payloads).
                client_round_simplified(
                    B, Ds, U, zeta,
                    lambda_theta_in=1e-4, lambda_U=0.5,
                    K_inner=args.K_inner, eta_theta=0.05,
                )
                round_payloads.append(Payload(synthetic=Ds, universum=U))

        # Optional pre-server DP on payloads, then optional post-server DP in aggregation.
        payloads_for_server = apply_pre_server_dp(round_payloads, dp_cfg, rng=rng)
        X_agg, A_agg, Y_agg = aggregate_payloads(payloads_for_server, dp_config=dp_cfg, rng=rng)
        if len(X_agg) == 0:
            theta_glob = zeta.copy()
        else:
            theta_glob = train_global_ridge_erm(
                X_agg, A_agg, Y_agg, zeta=zeta,
                lambda_theta=1e-4, max_iter=500, lr=0.05,
            )
        zeta = theta_glob.copy()

        eo_val, tpr0_val, tpr1_val, acc_val, f1_val = compute_eo_gap_and_accuracy(
            theta_glob, X_val, A_val, Y_val
        )
        round_logs.append({
            "round": t, "val_accuracy": acc_val, "val_EO_gap": eo_val,
            "val_TPR_group0": tpr0_val, "val_TPR_group1": tpr1_val, "val_F1": f1_val,
        })

    # Final evaluation (optionally with a validation-calibrated decision threshold).
    thr_pipe = pick_threshold(theta_glob, X_val, A_val, Y_val) if args.tune_threshold else 0.0
    eo_gap, tpr0, tpr1, acc, f1 = compute_eo_gap_and_accuracy(
        theta_glob, X_test, A_test, Y_test, threshold=thr_pipe
    )

    # Baseline comparison (its own calibrated threshold for a fair comparison).
    theta_baseline = train_global_ridge_erm(
        X_train, A_train, Y_train,
        zeta=np.zeros(d_plus_1), lambda_theta=1e-4, max_iter=500, lr=0.05,
    )
    thr_base = pick_threshold(theta_baseline, X_val, A_val, Y_val) if args.tune_threshold else 0.0
    eo_baseline, tpr0_b, tpr1_b, acc_baseline, f1_baseline = compute_eo_gap_and_accuracy(
        theta_baseline, X_test, A_test, Y_test, threshold=thr_base
    )

    extended_pipeline = compute_extended_metrics(theta_glob, X_test, A_test, Y_test)
    extended_baseline = compute_extended_metrics(theta_baseline, X_test, A_test, Y_test)

    results = {
        "baseline": {
            "accuracy": acc_baseline, "F1_score": f1_baseline, "EO_gap": eo_baseline,
            "TPR_group0": tpr0_b, "TPR_group1": tpr1_b,
            "extended_metrics": extended_baseline,
        },
        "pipeline": {
            "accuracy": acc, "F1_score": f1, "EO_gap": eo_gap,
            "TPR_group0": tpr0, "TPR_group1": tpr1,
            "num_clients": args.num_clients, "rounds": args.rounds,
            "use_tpr_gap": args.use_tpr_gap,
            "tpr_alpha": args.tpr_alpha,
            "tpr_tau": args.tpr_tau,
            "ema_beta": args.ema_beta,
            "use_importance_weighting": args.use_importance_weighting,
            "use_rolling_buffer": args.use_rolling_buffer,
            "dp_enabled": args.dp_enabled,
            "dp_sigma": args.dp_sigma,
            "dp_variant": dp_cfg.variant,
            "stop_criterion": args.stop_criterion,
            "outer_tol_xhat": args.outer_tol_xhat,
            "partition": args.partition,
            "dirichlet_alpha": args.dirichlet_alpha,
            "add_intercept": args.add_intercept,
            "tune_threshold": args.tune_threshold,
            "threshold_pipeline": thr_pipe,
            "threshold_baseline": thr_base,
            "extended_metrics": extended_pipeline,
        },
        "dataset": args.data,
        "round_logs": round_logs,
    }

    out_path = os.path.join(out_dir, args.results_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
