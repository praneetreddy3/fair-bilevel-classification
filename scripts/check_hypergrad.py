"""
Finite-difference check of the feature hypergradient in draft_model/bilevel_al.py.

Question: does the vector the code computes (grad_X) point along -dPhi/dx (so the original update
x -= eta*grad_X climbs Phi, a sign error) or along +dPhi/dx? And is it the right size?

Method (Law or Credit, client 1, seed 1, no Universum, lambda = 0 at the first outer step):
  Phi(x) = L_out(theta*(x)) + (rho/2) g(theta*(x))^2,  theta*(x) = the code's own inner fit on
  synthetic features x (client_round_al with J_outer = 1). For random directions u we compare
      numeric   : [Phi(x + e*u) - Phi(x - e*u)] / (2e)
      code      : <grad_X, u>
  slope = numeric / code over all directions.  slope ~ -1  -> grad_X = -dPhi/dx (original step
  ascends Phi);  slope ~ +1 -> original step descends Phi.  corr shows how well they agree.
Run with a long inner fit (K = 2000) so theta* is close to the true minimiser, which the
implicit-function formula assumes, and with/without --hg_weighted.

Needs PyTorch.   python scripts/check_hypergrad.py --data law
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import fl_baselines as fb  # noqa: E402
from draft_model.bilevel_al import client_round_al  # noqa: E402
from draft_model.losses import L_out, g_EO  # noqa: E402
from draft_model.minibatch_design import build_synthetic_templates, draw_original_minibatch  # noqa: E402
from draft_model.notation import ClientOriginalData, SyntheticMinibatch, UniversumSet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["credit", "adult", "law", "compas"], default="law")
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--dirs", type=int, default=12)
    ap.add_argument("--eps", type=float, default=1e-2)
    a = ap.parse_args()

    cfg = fb.DATASET_CFG[a.data]
    d = fb.prepare_splits(fb.load_dataset(a.data, cfg["sensitive"]), 1, cfg["add_intercept"])
    X, A, Y = d["clients"][0]
    data = ClientOriginalData(X=X, A=A, Y=Y)
    rng = np.random.default_rng(1)
    B = draw_original_minibatch(data, rng)
    Ds = build_synthetic_templates(B, Ds_size=32, rng=rng)
    U = UniversumSet(X=np.zeros((0, X.shape[1])), A=np.zeros(0))
    zeta = np.zeros(X.shape[1] + 1)
    fair = (X, A, Y)  # evaluate g on all client data (well-defined, non-flat signal)

    def run(Xs, weighted, diag=None):
        D2 = SyntheticMinibatch(X=Xs, A=Ds.A, Y=Ds.Y, q_a0y0=Ds.q_a0y0, q_a0y1=Ds.q_a0y1,
                                q_a1y0=Ds.q_a1y0, q_a1y1=Ds.q_a1y1, Delta_s=Ds.Delta_s)
        torch.manual_seed(1)
        th, _, _ = client_round_al(B, D2, U, zeta, lambda_theta_in=1e-4, lambda_theta_out=1e-4,
                                   lambda_U=0.5, rho=a.rho, epsilon_EO=0.0, K_inner=a.K, J_outer=1,
                                   eta_theta=0.05, eta_x=0.02, R=10.0, seed=1, ema_init="first",
                                   fair_data=fair, hg_weighted=weighted, diag=diag)
        return th

    def phi(th):
        t = torch.tensor(np.asarray(th, dtype=np.float32))
        lo = L_out(t, B.X, B.A, B.Y, torch.zeros_like(t), 1e-4, X.shape[1] + 1)
        g = g_EO(t, X, A, Y)
        return float(lo + 0.5 * a.rho * g * g)

    print(f"dataset={a.data} rho={a.rho} K_inner={a.K} directions={a.dirs}")
    for weighted in (False, True):
        dg = []
        run(Ds.X, weighted, diag=dg)
        gx = dg[0]["grad_X_ds"]
        num, ana = [], []
        r = np.random.default_rng(0)
        for _ in range(a.dirs):
            u = r.standard_normal(Ds.X.shape)
            u /= np.linalg.norm(u)
            fp = phi(run(Ds.X + a.eps * u, weighted))
            fm = phi(run(Ds.X - a.eps * u, weighted))
            num.append((fp - fm) / (2 * a.eps))
            ana.append(float((gx * u).sum()))
        num, ana = np.array(num), np.array(ana)
        slope = float(num @ ana / (ana @ ana + 1e-30))
        corr = float(np.corrcoef(num, ana)[0, 1])
        corrected = -ana  # dPhi/dx according to the code after the sign correction
        rel = float(np.linalg.norm(num - corrected) / (np.linalg.norm(num) + 1e-30))
        print(f"hg_weighted={weighted!s:<5}  slope(numeric/code)={slope:+.3f}  corr={corr:+.3f}  "
              f"relative error after sign fix={rel:.2f}")
    print("\nslope ~ -1: code's grad_X = -dPhi/dx, so the original step (x -= eta*grad_X) climbs Phi.")
    print("slope ~ +1: original step descends Phi.  |corr| near 1 = the hypergradient is otherwise right.")


if __name__ == "__main__":
    main()
