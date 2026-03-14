"""
Main pipeline runner: data → clients → minibatch → synthetic → Universum → bilevel AL → server.
Evaluates both a baseline (ERM) and the fairness-aware pipeline on the test set.

Usage: python -m draft_model.run_draft --data {dummy,adult,2d} [options]
"""
import os
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
from draft_model.server import aggregate_payloads, train_global_ridge_erm, compute_eo_gap_and_accuracy


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
    """Load UCI Adult dataset."""
    from pipeline.load_adult import prepare_adult_for_draft
    return prepare_adult_for_draft(sensitive=sensitive, use_ucimlrepo=True)


def load_2d_data(project_root: str):
    """Load 2D example data from outputs/2d_data.csv."""
    from pipeline.load_2d import load_2d_for_draft
    return load_2d_for_draft(project_root)


def main():
    parser = argparse.ArgumentParser(description="Run the fair bilevel pipeline.")
    parser.add_argument("--data", default="dummy", choices=["dummy", "adult", "2d"])
    parser.add_argument("--sensitive", default="sex", choices=["sex", "race"])
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--K_inner", type=int, default=50)
    parser.add_argument("--no_full_al", action="store_true", help="Use simplified inner-only path")
    parser.add_argument("--J_outer", type=int, default=5)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--epsilon_EO", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--no_universum", action="store_true", help="Ablation: disable Universum")
    parser.add_argument("--fairness_off", action="store_true", help="Ablation: set ρ=0")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--no_scale", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load data
    if args.data == "dummy":
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = load_dummy_law_style(PROJECT_ROOT)
    elif args.data == "adult":
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = load_adult_data(sensitive=args.sensitive)
    elif args.data == "2d":
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = load_2d_data(PROJECT_ROOT)
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

    d = X_train.shape[1]
    d_plus_1 = d + 1
    zeta = np.zeros(d_plus_1)

    # Split data across clients
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
                theta_k = client_round_simplified(
                    B, Ds, U, zeta,
                    lambda_theta_in=1e-4, lambda_U=0.5,
                    K_inner=args.K_inner, eta_theta=0.05,
                )
                round_payloads.append(Payload(synthetic=Ds, universum=U))

        # Server aggregation
        X_agg, A_agg, Y_agg = aggregate_payloads(round_payloads)
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

    # Final evaluation
    eo_gap, tpr0, tpr1, acc, f1 = compute_eo_gap_and_accuracy(theta_glob, X_test, A_test, Y_test)

    # Baseline comparison
    theta_baseline = train_global_ridge_erm(
        X_train, A_train, Y_train,
        zeta=np.zeros(d_plus_1), lambda_theta=1e-4, max_iter=200, lr=0.05,
    )
    eo_baseline, tpr0_b, tpr1_b, acc_baseline, f1_baseline = compute_eo_gap_and_accuracy(
        theta_baseline, X_test, A_test, Y_test
    )

    results = {
        "baseline": {
            "accuracy": acc_baseline, "F1_score": f1_baseline, "EO_gap": eo_baseline,
            "TPR_group0": tpr0_b, "TPR_group1": tpr1_b,
        },
        "pipeline": {
            "accuracy": acc, "F1_score": f1, "EO_gap": eo_gap,
            "TPR_group0": tpr0, "TPR_group1": tpr1,
            "num_clients": args.num_clients, "rounds": args.rounds,
        },
        "dataset": args.data,
        "round_logs": round_logs,
    }

    out_path = os.path.join(out_dir, "draft_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
