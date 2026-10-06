"""
Diagnostic: does the outer (feature-update) loop actually reduce the EO surrogate?

Runs ONE client round of draft_model.bilevel_al.client_round_al on the first client's data
(same data, split, minibatch and synthetic template as run_draft.py for seed 1) for a few outer
iteration caps J and penalties rho, and reports
  - g_before : EO surrogate g_EO of the model fitted on the initial synthetic batch
  - g_after  : EO surrogate g_EO of the model fitted on the updated synthetic batch
               (both evaluated on the client's original minibatch B)
  - moved    : mean absolute change of the synthetic features
If g_after << g_before and grows more negative with larger J / rho, the solver works.
If g_after ~ g_before for every J and rho, the fairness stage is not doing anything mechanically.

Needs PyTorch.   python scripts/diag_outer_loop.py --data law
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import fl_baselines as fb  # noqa: E402  (data loading / identical splits)
from draft_model.bilevel_al import client_round_al  # noqa: E402
from draft_model.losses import g_EO  # noqa: E402
from draft_model.minibatch_design import (  # noqa: E402
    build_synthetic_templates, build_universum_templates, draw_original_minibatch)
from draft_model.notation import ClientOriginalData, UniversumSet  # noqa: E402


def fit_theta(Xs, As, Ys, zeta, steps=100, lr=0.05):
    """Same inner fit (Adam on logistic loss + ridge) used for the 'before' reference."""
    Z = np.hstack([Xs, As.reshape(-1, 1)])
    return fb.local_adam(zeta.copy(), Z, Ys, np.ones(len(Ys)), steps=steps, lr=lr)


def surrogate(theta, B, alpha=10.0):
    return float(g_EO(torch.tensor(theta, dtype=torch.float32), B.X, B.A, B.Y, alpha=alpha).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["credit", "adult", "law", "compas"], default="law")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=10.0, help="sigmoid sharpness of the TPR surrogate")
    ap.add_argument("--hg_sign", choices=["orig", "correct"], default="orig")
    a = ap.parse_args()

    cfg = fb.DATASET_CFG[a.data]
    raw = fb.load_dataset(a.data, cfg["sensitive"])
    d = fb.prepare_splits(raw, a.seed, cfg["add_intercept"])
    X, A, Y = d["clients"][0]
    data = ClientOriginalData(X=X, A=A, Y=Y)
    zeta = np.zeros(X.shape[1] + 1)

    rng = np.random.default_rng(a.seed)
    B = draw_original_minibatch(data, rng)
    Ds = build_synthetic_templates(B, Ds_size=32, rng=rng)
    if a.data == "compas":  # reported COMPAS config runs without Universum
        U = UniversumSet(X=np.zeros((0, X.shape[1])), A=np.zeros(0))
    else:
        U = build_universum_templates(Ds.Delta_s, Ds.size(), B.Delta, X.shape[1], rng=rng,
                                      q_a0y1=B.q_a0y1, q_a1y1=B.q_a1y1, B=B)

    theta0 = fit_theta(Ds.X, Ds.A, Ds.Y, zeta)
    g0 = surrogate(theta0, B, a.alpha)
    print(f"dataset={a.data} hg_sign={a.hg_sign} alpha={a.alpha} original minibatch n={len(B.Y)} (positives per group: S0={B.q_a0y1}, S1={B.q_a1y1}) synthetic n={Ds.size()} U={U.size()}")
    print(f"g_EO of model fitted on INITIAL synthetics: {g0:.4f}\n")
    print(f"{'ema_init':<9}{'rho':<7}{'J':<4}{'g_after':<10}{'change':<10}{'moved':<10}")
    for ema_init in ("zero", "first"):
        for rho in (0.0, 0.1, 0.5, 2.0):
            for J in (1, 5, 20):
                if ema_init == "zero" and J != 20:
                    continue
                torch.manual_seed(a.seed)
                theta, Xn, Xu = client_round_al(
                    B, Ds, U, zeta, lambda_theta_in=1e-4, lambda_theta_out=1e-4, lambda_U=0.5,
                    rho=rho, epsilon_EO=0.0 if ema_init == "first" else 0.1,  # eps=0: never stop early
                    K_inner=100, J_outer=J, eta_theta=0.05, eta_x=0.02, R=10.0, seed=a.seed,
                    ema_init=ema_init, tpr_alpha=a.alpha, hg_sign=a.hg_sign)
                g1 = surrogate(theta, B, a.alpha)
                moved = float(np.mean(np.abs(Xn - Ds.X)))
                if ema_init == "first" and J == 20 and rho in (0.0, 2.0):
                    dg = []
                    torch.manual_seed(a.seed)
                    client_round_al(B, Ds, U, zeta, lambda_theta_in=1e-4, lambda_theta_out=1e-4, lambda_U=0.5,
                                    rho=rho, epsilon_EO=0.0, K_inner=100, J_outer=6, eta_theta=0.05,
                                    eta_x=0.02, R=10.0, seed=a.seed, ema_init="first",
                                    tpr_alpha=a.alpha, hg_sign=a.hg_sign, diag=dg)
                    for r in dg:
                        print(f"   rho={rho} j={r['j']} g={r['g']:.4f} |grad_Lout|={r['gl']:.2e} |grad_g|={r['gg']:.2e}")
                print(f"{ema_init:<9}{rho:<7}{J:<4}{g1:<10.4f}{g1 - g0:<+10.4f}{moved:<10.4f}")


if __name__ == "__main__":
    main()
