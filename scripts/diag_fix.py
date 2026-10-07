"""
Quick check (minutes): does any candidate fix make the AL penalty rho actually change fairness?

For a few clients of one dataset (seed 1, same splits/minibatches as run_draft.py) it runs one
client round of client_round_al with the stopping fix (--ema_init first, no early stop, J=20) for
each fix variant and rho in {0, 1, 10, 50}, and reports the HARD EO gap and accuracy of the
resulting client model on that client's full training data (averaged over clients).

A fix "works" here if EO goes down as rho goes up (compared with rho = 0 of the SAME variant)
without a large accuracy loss. Only variants that pass this check go to the 5-seed run
(scripts/run_fix.py --stage 5). This uses training data only; no validation/test data.

Variants
  base : original surrogate on the minibatch (reference)
  A    : surrogate evaluated on ALL the client's qualified examples   (--fair_set client)
  B    : non-saturating loss-gap surrogate                            (--eo_surrogate loss_gap)
  C    : fairness gradient rescaled to the outer-loss gradient size   (--fair_grad_norm true)

Needs PyTorch.   python scripts/diag_fix.py --data law
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
from draft_model.minibatch_design import (  # noqa: E402
    build_synthetic_templates, build_universum_templates, draw_original_minibatch)
from draft_model.notation import ClientOriginalData, UniversumSet  # noqa: E402

VARIANTS = {  # name: (fair_set_client, surrogate, grad_norm)
    "base": (False, "tpr_gap", False),
    "A": (True, "tpr_gap", False),
    "B": (False, "loss_gap", False),
    "A+B": (True, "loss_gap", False),
    "A+C": (True, "tpr_gap", True),
    "A+B+C": (True, "loss_gap", True),
}
RHOS = (0.0, 1.0, 10.0, 50.0)


def hard_metrics(theta, X, A, Y):
    logits = np.hstack([X, A.reshape(-1, 1)]) @ np.asarray(theta, dtype=float).ravel()
    pred = logits >= 0
    acc = float(np.mean(pred == (Y == 1)))
    tpr = [pred[(Y == 1) & (A == s)].mean() if np.any((Y == 1) & (A == s)) else np.nan for s in (0, 1)]
    return acc, float(abs(tpr[1] - tpr[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["credit", "adult", "law", "compas"], default="law")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clients", type=int, default=3)
    a = ap.parse_args()

    cfg = fb.DATASET_CFG[a.data]
    d = fb.prepare_splits(fb.load_dataset(a.data, cfg["sensitive"]), a.seed, cfg["add_intercept"])
    setups = []
    rng = np.random.default_rng(a.seed)
    for k in range(a.clients):
        X, A, Y = d["clients"][k]
        data = ClientOriginalData(X=X, A=A, Y=Y)
        B = draw_original_minibatch(data, rng)
        Ds = build_synthetic_templates(B, Ds_size=32, rng=rng)
        if a.data == "compas":
            U = UniversumSet(X=np.zeros((0, X.shape[1])), A=np.zeros(0))
        else:
            U = build_universum_templates(Ds.Delta_s, Ds.size(), B.Delta, X.shape[1], rng=rng,
                                          q_a0y1=B.q_a0y1, q_a1y1=B.q_a1y1, B=B)
        setups.append((data, B, Ds, U))

    print(f"dataset={a.data}  clients={a.clients}  (hard EO / accuracy on client training data, mean over clients)")
    print(f"{'variant':<8}" + "".join(f"{'rho=' + str(r):>18}" for r in RHOS))
    for name, (use_client, sur, gnorm) in VARIANTS.items():
        cells = []
        for rho in RHOS:
            accs, eos = [], []
            for data, B, Ds, U in setups:
                torch.manual_seed(a.seed)
                zeta = np.zeros(data.X.shape[1] + 1)
                try:
                    theta, _, _ = client_round_al(
                        B, Ds, U, zeta, lambda_theta_in=1e-4, lambda_theta_out=1e-4, lambda_U=0.5,
                        rho=rho, epsilon_EO=0.0, K_inner=100, J_outer=20, eta_theta=0.05,
                        eta_x=0.02, R=10.0, seed=a.seed, ema_init="first", eo_surrogate=sur,
                        fair_data=(data.X, data.A, data.Y) if use_client else None,
                        fair_grad_norm=gnorm)
                    acc, eo = hard_metrics(theta, data.X, data.A, data.Y)
                except Exception:
                    acc, eo = np.nan, np.nan
                accs.append(acc); eos.append(eo)
            cells.append(f"EO {np.nanmean(eos):.3f} acc {np.nanmean(accs):.2f}")
        print(f"{name:<8}" + "".join(f"{c:>18}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
