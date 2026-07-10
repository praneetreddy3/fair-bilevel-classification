"""
Per-minibatch performance analysis for synthetic training flow.

This script mirrors run_draft's client loop and records metrics for each
client-round minibatch on:
1) real minibatch B
2) synthetic minibatch D^s

Outputs:
- outputs/minibatch_performance_<data>.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from draft_model.bilevel_al import client_round_al, client_round_simplified
from draft_model.minibatch_design import (
    build_synthetic_templates,
    build_universum_templates,
    draw_original_minibatch,
)
from draft_model.dp import DP_VARIANTS, apply_pre_server_dp, resolve_dp_config
from draft_model.notation import ClientOriginalData, Payload, SyntheticMinibatch, UniversumSet
from draft_model.run_draft import load_2d_data, load_adult_data, str2bool
from draft_model.server import aggregate_payloads, compute_eo_gap_and_accuracy, train_global_ridge_erm


@dataclass
class RunCfg:
    data: str
    sensitive: str
    num_clients: int
    rounds: int
    K_inner: int
    J_outer: int
    rho: float
    epsilon_EO: float
    seed: int
    val_frac: float
    no_scale: bool
    no_universum: bool
    no_full_al: bool
    use_tpr_gap: bool
    tpr_alpha: float
    tpr_tau: float
    ema_beta: float
    use_importance_weighting: bool
    use_rolling_buffer: bool
    dp_enabled: bool
    dp_sigma: float
    dp_variant: str
    stop_criterion: str
    outer_tol_xhat: float


def _build_cfg(args: argparse.Namespace) -> RunCfg:
    rho = 0.0 if args.fairness_off else args.rho
    return RunCfg(
        data=args.data,
        sensitive=args.sensitive,
        num_clients=args.num_clients,
        rounds=args.rounds,
        K_inner=args.K_inner,
        J_outer=args.J_outer,
        rho=rho,
        epsilon_EO=args.epsilon_EO,
        seed=args.seed,
        val_frac=args.val_frac,
        no_scale=args.no_scale,
        no_universum=args.no_universum,
        no_full_al=args.no_full_al,
        use_tpr_gap=args.use_tpr_gap,
        tpr_alpha=args.tpr_alpha,
        tpr_tau=args.tpr_tau,
        ema_beta=args.ema_beta,
        use_importance_weighting=args.use_importance_weighting,
        use_rolling_buffer=args.use_rolling_buffer,
        dp_enabled=args.dp_enabled,
        dp_sigma=args.dp_sigma,
        dp_variant=args.dp_variant,
        stop_criterion=args.stop_criterion,
        outer_tol_xhat=args.outer_tol_xhat,
    )


def _load_data(cfg: RunCfg):
    if cfg.data == "adult":
        return load_adult_data(sensitive=cfg.sensitive)
    if cfg.data == "2d":
        return load_2d_data(SCRIPT_DIR)
    raise ValueError("This analyzer currently supports --data {2d,adult}.")


def _metric_row(theta: np.ndarray, X: np.ndarray, A: np.ndarray, Y: np.ndarray):
    eo, tpr0, tpr1, acc, f1 = compute_eo_gap_and_accuracy(theta, X, A, Y)
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "eo_gap": float(eo),
        "tpr0": float(tpr0),
        "tpr1": float(tpr1),
    }


def main():
    p = argparse.ArgumentParser(description="Analyze per-minibatch synthetic performance.")
    p.add_argument("--data", choices=["2d", "adult"], default="adult")
    p.add_argument("--sensitive", choices=["sex", "race"], default="sex")
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--K_inner", type=int, default=50)
    p.add_argument("--J_outer", type=int, default=20)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--epsilon_EO", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--no_scale", action="store_true")
    p.add_argument("--no_universum", action="store_true")
    p.add_argument("--no_full_al", action="store_true")
    p.add_argument("--fairness_off", action="store_true")
    p.add_argument("--use_tpr_gap", type=str2bool, default=True)
    p.add_argument("--tpr_alpha", type=float, default=10.0)
    p.add_argument("--tpr_tau", type=float, default=0.0)
    p.add_argument("--ema_beta", type=float, default=0.15)
    p.add_argument("--use_importance_weighting", type=str2bool, default=True)
    p.add_argument("--use_rolling_buffer", type=str2bool, default=True)
    p.add_argument("--dp_enabled", type=str2bool, default=False)
    p.add_argument("--dp_sigma", type=float, default=1.0)
    p.add_argument("--dp_variant", choices=DP_VARIANTS, default="post_server")
    p.add_argument("--stop_criterion", choices=["eo_gap", "grad_inf"], default="eo_gap")
    p.add_argument("--outer_tol_xhat", type=float, default=1e-6)
    p.add_argument("--out_file", default=None)
    args = p.parse_args()

    cfg = _build_cfg(args)
    dp_cfg = resolve_dp_config(cfg.dp_enabled, cfg.dp_sigma, cfg.dp_variant)
    out_dir = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = args.out_file or os.path.join(out_dir, f"minibatch_performance_{cfg.data}.csv")

    rng = np.random.default_rng(cfg.seed)
    (X_train, A_train, Y_train), (X_test, A_test, Y_test) = _load_data(cfg)

    # Match run_draft split/scaling.
    n = len(Y_train)
    n_val = int(n * cfg.val_frac)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    val_idx, train_idx = all_idx[:n_val], all_idx[n_val:]
    X_val, A_val, Y_val = X_train[val_idx], A_train[val_idx], Y_train[val_idx]
    X_train, A_train, Y_train = X_train[train_idx], A_train[train_idx], Y_train[train_idx]

    if not cfg.no_scale:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.astype(np.float64))
        X_val = scaler.transform(X_val.astype(np.float64))
        X_test = scaler.transform(X_test.astype(np.float64))

    d = X_train.shape[1]
    zeta = np.zeros(d + 1, dtype=np.float64)
    theta_glob = zeta.copy()

    idx = np.arange(len(Y_train))
    rng.shuffle(idx)
    splits = np.array_split(idx, cfg.num_clients)
    clients = [ClientOriginalData(X=X_train[s], A=A_train[s], Y=Y_train[s]) for s in splits]

    rows = []
    for t in range(cfg.rounds):
        payloads = []
        for k, data in enumerate(clients):
            B = draw_original_minibatch(data, rng)
            Ds = build_synthetic_templates(B, Ds_size=32, rng=rng)
            if cfg.no_universum:
                U = UniversumSet(X=np.zeros((0, d)), A=np.zeros(0))
            else:
                U = build_universum_templates(
                    Ds.Delta_s, Ds.size(), B.Delta, d, rng=rng, q_a0y1=B.q_a0y1, q_a1y1=B.q_a1y1, B=B
                )

            if cfg.no_full_al:
                theta_k = client_round_simplified(
                    B, Ds, U, zeta,
                    lambda_theta_in=1e-4, lambda_U=0.5,
                    K_inner=cfg.K_inner, eta_theta=0.05,
                )
                Ds_send, U_send = Ds, U
            else:
                theta_k, X_ds_new, X_u_new = client_round_al(
                    B, Ds, U, zeta,
                    lambda_theta_in=1e-4, lambda_theta_out=1e-4,
                    lambda_U=0.5, rho=cfg.rho, epsilon_EO=cfg.epsilon_EO,
                    K_inner=cfg.K_inner, J_outer=cfg.J_outer,
                    eta_theta=0.05, eta_x=0.02, R=10.0,
                    seed=cfg.seed + t * 1000 + k,
                    use_tpr_gap=cfg.use_tpr_gap,
                    tpr_alpha=cfg.tpr_alpha,
                    tpr_tau=cfg.tpr_tau,
                    ema_beta=cfg.ema_beta,
                    use_importance_weighting=cfg.use_importance_weighting,
                    use_rolling_buffer=cfg.use_rolling_buffer,
                    stop_criterion=cfg.stop_criterion,
                    outer_tol_xhat=cfg.outer_tol_xhat,
                )
                Ds_send = SyntheticMinibatch(
                    X=X_ds_new, A=Ds.A, Y=Ds.Y,
                    q_a0y0=Ds.q_a0y0, q_a0y1=Ds.q_a0y1, q_a1y0=Ds.q_a1y0, q_a1y1=Ds.q_a1y1,
                    Delta_s=Ds.Delta_s,
                )
                U_send = UniversumSet(X=X_u_new, A=U.A)

            # Per-minibatch metrics on B (real) and Ds (synthetic)
            mB = _metric_row(theta_k, B.X, B.A, B.Y)
            mDs = _metric_row(theta_k, Ds_send.X, Ds_send.A, Ds_send.Y)
            rows.append(
                {
                    "round": t,
                    "client_id": k,
                    "B_size": len(B.Y),
                    "Ds_size": Ds_send.size(),
                    "U_size": U_send.size(),
                    "B_acc": mB["accuracy"],
                    "B_f1": mB["f1"],
                    "B_eo_gap": mB["eo_gap"],
                    "B_tpr0": mB["tpr0"],
                    "B_tpr1": mB["tpr1"],
                    "Ds_acc": mDs["accuracy"],
                    "Ds_f1": mDs["f1"],
                    "Ds_eo_gap": mDs["eo_gap"],
                    "Ds_tpr0": mDs["tpr0"],
                    "Ds_tpr1": mDs["tpr1"],
                }
            )

            payloads.append(Payload(synthetic=Ds_send, universum=U_send))

        payloads_for_server = apply_pre_server_dp(payloads, dp_cfg, rng=rng)
        X_agg, A_agg, Y_agg = aggregate_payloads(payloads_for_server, dp_config=dp_cfg)
        if len(X_agg) > 0:
            theta_glob = train_global_ridge_erm(
                X_agg, A_agg, Y_agg, zeta=zeta, lambda_theta=1e-4, max_iter=500, lr=0.05
            )
        zeta = theta_glob.copy()

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    print(df.head(12).to_string(index=False))

    # Final global snapshot for context.
    m_val = _metric_row(theta_glob, X_val, A_val, Y_val)
    m_test = _metric_row(theta_glob, X_test, A_test, Y_test)
    print("\nFinal global metrics:")
    print(
        f"  VAL  acc={m_val['accuracy']:.4f} f1={m_val['f1']:.4f} eo={m_val['eo_gap']:.4f} "
        f"(tpr0={m_val['tpr0']:.4f}, tpr1={m_val['tpr1']:.4f})"
    )
    print(
        f"  TEST acc={m_test['accuracy']:.4f} f1={m_test['f1']:.4f} eo={m_test['eo_gap']:.4f} "
        f"(tpr0={m_test['tpr0']:.4f}, tpr1={m_test['tpr1']:.4f})"
    )


if __name__ == "__main__":
    main()
