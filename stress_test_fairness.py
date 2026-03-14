"""
Controlled synthetic fairness stress tests.
Generates imbalanced scenarios where baseline has nonzero EO gap,
then runs the draft pipeline to test whether it reduces the gap.

Run from project root:  python stress_test_fairness.py
"""
import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from draft_model.notation import (
    ClientOriginalData, Payload, SyntheticMinibatch, UniversumSet,
)
from draft_model.minibatch_design import (
    draw_original_minibatch, build_synthetic_templates, build_universum_templates,
)
from draft_model.bilevel_al import client_round_al
from draft_model.server import (
    aggregate_payloads, train_global_ridge_erm, compute_eo_gap_and_accuracy,
)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def generate_scenario(
    name: str,
    n_total: int,
    frac_a0: float,
    pos_rate_a0: float,
    pos_rate_a1: float,
    d: int,
    sep_a0: float,
    sep_a1: float,
    noise_a0: float,
    noise_a1: float,
    seed: int,
) -> dict:
    """
    Build a binary classification dataset where group A=1 positives are
    harder to separate from negatives than group A=0 positives.

    sep_aX  : signed offset of positive-class mean along the discriminant axis
    noise_aX: std of features (higher = harder to classify)

    Returns dict with train/test splits and metadata.
    """
    rng = np.random.default_rng(seed)

    n_a0 = int(n_total * frac_a0)
    n_a1 = n_total - n_a0
    n_a0y1 = int(n_a0 * pos_rate_a0)
    n_a0y0 = n_a0 - n_a0y1
    n_a1y1 = int(n_a1 * pos_rate_a1)
    n_a1y0 = n_a1 - n_a1y1

    def make_block(n, mean_offset, noise, d):
        X = rng.normal(loc=0, scale=noise, size=(n, d))
        X[:, 0] += mean_offset
        return X

    X_a0y0 = make_block(n_a0y0, -sep_a0, noise_a0, d)
    X_a0y1 = make_block(n_a0y1, +sep_a0, noise_a0, d)
    X_a1y0 = make_block(n_a1y0, -sep_a1, noise_a1, d)
    X_a1y1 = make_block(n_a1y1, +sep_a1, noise_a1, d)

    X = np.vstack([X_a0y0, X_a0y1, X_a1y0, X_a1y1])
    A = np.array(
        [0]*n_a0y0 + [0]*n_a0y1 + [1]*n_a1y0 + [1]*n_a1y1, dtype=np.float64,
    )
    Y = np.array(
        [0]*n_a0y0 + [1]*n_a0y1 + [0]*n_a1y0 + [1]*n_a1y1, dtype=np.float64,
    )

    perm = rng.permutation(n_total)
    X, A, Y = X[perm], A[perm], Y[perm]

    split = int(0.8 * n_total)
    return {
        "name": name,
        "X_train": X[:split], "A_train": A[:split], "Y_train": Y[:split],
        "X_test": X[split:], "A_test": A[split:], "Y_test": Y[split:],
        "meta": {
            "n_total": n_total,
            "frac_a0": frac_a0,
            "pos_rate_a0": pos_rate_a0,
            "pos_rate_a1": pos_rate_a1,
            "sep_a0": sep_a0,
            "sep_a1": sep_a1,
            "noise_a0": noise_a0,
            "noise_a1": noise_a1,
            "d": d,
            "seed": seed,
            "n_a0y0": int(np.sum((A[:split] == 0) & (Y[:split] == 0))),
            "n_a0y1": int(np.sum((A[:split] == 0) & (Y[:split] == 1))),
            "n_a1y0": int(np.sum((A[:split] == 1) & (Y[:split] == 0))),
            "n_a1y1": int(np.sum((A[:split] == 1) & (Y[:split] == 1))),
        },
    }


SCENARIOS = [
    {
        "name": "moderate_imbalance",
        "n_total": 1200,
        "frac_a0": 0.60,
        "pos_rate_a0": 0.30,
        "pos_rate_a1": 0.10,
        "d": 5,
        "sep_a0": 1.8,
        "sep_a1": 0.5,
        "noise_a0": 1.0,
        "noise_a1": 1.5,
        "seed": 100,
    },
    {
        "name": "strong_imbalance",
        "n_total": 1500,
        "frac_a0": 0.70,
        "pos_rate_a0": 0.25,
        "pos_rate_a1": 0.05,
        "d": 5,
        "sep_a0": 2.2,
        "sep_a1": 0.3,
        "noise_a0": 1.0,
        "noise_a1": 2.0,
        "seed": 200,
    },
]


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def run_baseline(X_train, A_train, Y_train, X_test, A_test, Y_test):
    """Train on raw data, evaluate."""
    d_plus_1 = X_train.shape[1] + 1
    zeta = np.zeros(d_plus_1)
    theta = train_global_ridge_erm(
        X_train, A_train, Y_train,
        zeta=zeta, lambda_theta=1e-4, max_iter=500, lr=0.05,
    )
    eo, tpr0, tpr1, acc, f1 = compute_eo_gap_and_accuracy(
        theta, X_test, A_test, Y_test,
    )
    return {"accuracy": acc, "F1": f1, "TPR_group0": tpr0,
            "TPR_group1": tpr1, "EO_gap": eo}


def run_draft_pipeline(
    X_train, A_train, Y_train,
    X_test, A_test, Y_test,
    num_clients: int,
    rounds: int,
    K_inner: int,
    J_outer: int,
    rho: float,
    seed: int,
):
    """Full draft pipeline: stratified minibatch + Ds + U + AL + server."""
    rng = np.random.default_rng(seed)
    d = X_train.shape[1]
    d_plus_1 = d + 1
    zeta = np.zeros(d_plus_1)

    indices = np.arange(len(Y_train))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    clients = [
        ClientOriginalData(X=X_train[s], A=A_train[s], Y=Y_train[s])
        for s in splits
    ]

    theta_glob = zeta.copy()
    round_logs = []

    for t in range(rounds):
        payloads = []
        for k, data in enumerate(clients):
            B = draw_original_minibatch(data, rng)
            Ds = build_synthetic_templates(B, Ds_size=32, rng=rng)
            U = build_universum_templates(
                Ds.Delta_s, Ds.size(), B.Delta, d, rng=rng,
                q_a0y1=B.q_a0y1, q_a1y1=B.q_a1y1, B=B,
            )
            theta_k, X_ds_new, X_u_new = client_round_al(
                B, Ds, U, zeta,
                lambda_theta_in=1e-4, lambda_theta_out=1e-4,
                lambda_U=0.5, rho=rho, epsilon_EO=0.05,
                K_inner=K_inner, J_outer=J_outer,
                eta_theta=0.05, eta_x=0.02, R=10.0,
                seed=seed + t * 1000 + k,
            )
            Ds_send = SyntheticMinibatch(
                X=X_ds_new, A=Ds.A, Y=Ds.Y,
                q_a0y0=Ds.q_a0y0, q_a0y1=Ds.q_a0y1,
                q_a1y0=Ds.q_a1y0, q_a1y1=Ds.q_a1y1,
                Delta_s=Ds.Delta_s,
            )
            U_send = UniversumSet(X=X_u_new, A=U.A)
            payloads.append(Payload(synthetic=Ds_send, universum=U_send))

        X_agg, A_agg, Y_agg = aggregate_payloads(payloads)
        if len(X_agg) == 0:
            theta_glob = zeta.copy()
        else:
            theta_glob = train_global_ridge_erm(
                X_agg, A_agg, Y_agg,
                zeta=zeta, lambda_theta=1e-4, max_iter=500, lr=0.05,
            )
        zeta = theta_glob.copy()

        eo_v, t0_v, t1_v, acc_v, f1_v = compute_eo_gap_and_accuracy(
            theta_glob, X_test, A_test, Y_test,
        )
        round_logs.append({
            "round": t, "acc": acc_v, "EO": eo_v,
            "TPR0": t0_v, "TPR1": t1_v, "F1": f1_v,
        })

    eo, tpr0, tpr1, acc, f1 = compute_eo_gap_and_accuracy(
        theta_glob, X_test, A_test, Y_test,
    )
    return {
        "accuracy": acc, "F1": f1, "TPR_group0": tpr0,
        "TPR_group1": tpr1, "EO_gap": eo, "round_logs": round_logs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for spec in SCENARIOS:
        name = spec["name"]
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"{'='*60}")

        sc = generate_scenario(**spec)
        meta = sc["meta"]
        X_tr, A_tr, Y_tr = sc["X_train"], sc["A_train"], sc["Y_train"]
        X_te, A_te, Y_te = sc["X_test"], sc["A_test"], sc["Y_test"]

        print(f"  Train: {len(Y_tr)} samples")
        print(f"    (a=0,y=0)={meta['n_a0y0']}  (a=0,y=1)={meta['n_a0y1']}")
        print(f"    (a=1,y=0)={meta['n_a1y0']}  (a=1,y=1)={meta['n_a1y1']}")
        print(f"  Test:  {len(Y_te)} samples")

        # --- Baseline ---
        print("\n  [Baseline] training on raw data ...")
        bl = run_baseline(X_tr, A_tr, Y_tr, X_te, A_te, Y_te)
        print(f"    Acc={bl['accuracy']:.4f}  F1={bl['F1']:.4f}")
        print(f"    TPR0={bl['TPR_group0']:.4f}  TPR1={bl['TPR_group1']:.4f}  "
              f"EO_gap={bl['EO_gap']:.4f}")

        # --- Draft pipeline ---
        num_clients = 4
        rounds = 5
        print(f"\n  [Pipeline] {num_clients} clients, {rounds} rounds, full AL ...")
        pl = run_draft_pipeline(
            X_tr, A_tr, Y_tr, X_te, A_te, Y_te,
            num_clients=num_clients, rounds=rounds,
            K_inner=50, J_outer=5, rho=2.0, seed=spec["seed"],
        )
        print(f"    Acc={pl['accuracy']:.4f}  F1={pl['F1']:.4f}")
        print(f"    TPR0={pl['TPR_group0']:.4f}  TPR1={pl['TPR_group1']:.4f}  "
              f"EO_gap={pl['EO_gap']:.4f}")

        eo_delta = bl["EO_gap"] - pl["EO_gap"]
        print(f"\n  EO gap change: {bl['EO_gap']:.4f} -> {pl['EO_gap']:.4f}  "
              f"(delta={eo_delta:+.4f})")

        all_results[name] = {
            "scenario": meta,
            "baseline": bl,
            "pipeline": {k: v for k, v in pl.items() if k != "round_logs"},
            "round_logs": pl["round_logs"],
            "eo_gap_delta": eo_delta,
            "pipeline_config": {
                "num_clients": num_clients,
                "rounds": rounds,
                "K_inner": 50,
                "J_outer": 5,
                "rho": 2.0,
            },
        }

    # --- Save JSON ---
    json_path = os.path.join(out_dir, "stress_test_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # --- Save summary ---
    summary_path = os.path.join(out_dir, "stress_test_summary.txt")
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("FAIRNESS STRESS TEST SUMMARY\n")
        f.write("="*70 + "\n\n")

        for name, res in all_results.items():
            sc = res["scenario"]
            bl = res["baseline"]
            pl = res["pipeline"]
            delta = res["eo_gap_delta"]

            f.write(f"--- {name} ---\n")
            f.write(f"Construction:\n")
            f.write(f"  N={sc['n_total']}, d={sc['d']}, "
                    f"frac(A=0)={sc.get('frac_a0', 'N/A')}\n")
            f.write(f"  Positive rate A=0: {sc['pos_rate_a0']}, "
                    f"A=1: {sc['pos_rate_a1']}\n")
            f.write(f"  Separation A=0: {sc['sep_a0']}, "
                    f"A=1: {sc['sep_a1']}\n")
            f.write(f"  Noise A=0: {sc['noise_a0']}, "
                    f"A=1: {sc['noise_a1']}\n")
            f.write(f"  Train strata: (a0y0={sc['n_a0y0']}, a0y1={sc['n_a0y1']}, "
                    f"a1y0={sc['n_a1y0']}, a1y1={sc['n_a1y1']})\n")
            f.write(f"  Design: Group A=1 has lower positive rate, "
                    f"less separation, more noise.\n")
            f.write(f"          This makes A=1 positives harder to classify, "
                    f"producing lower TPR for A=1.\n\n")

            f.write(f"Baseline (raw ERM):\n")
            f.write(f"  Acc={bl['accuracy']:.4f}  F1={bl['F1']:.4f}\n")
            f.write(f"  TPR_0={bl['TPR_group0']:.4f}  "
                    f"TPR_1={bl['TPR_group1']:.4f}  "
                    f"EO_gap={bl['EO_gap']:.4f}\n")
            baseline_eo_nonzero = bl["EO_gap"] > 0.01
            f.write(f"  Baseline EO gap nonzero (>0.01): "
                    f"{'YES' if baseline_eo_nonzero else 'NO'}\n\n")

            f.write(f"Pipeline (Synthetic + Universum + AL, rho={res['pipeline_config']['rho']}):\n")
            f.write(f"  Acc={pl['accuracy']:.4f}  F1={pl['F1']:.4f}\n")
            f.write(f"  TPR_0={pl['TPR_group0']:.4f}  "
                    f"TPR_1={pl['TPR_group1']:.4f}  "
                    f"EO_gap={pl['EO_gap']:.4f}\n")
            improved = delta > 0.005
            f.write(f"  EO gap reduced: {'YES' if improved else 'NO'} "
                    f"(delta={delta:+.4f})\n\n")

            f.write(f"Per-round progression:\n")
            for rl in res["round_logs"]:
                f.write(f"  Round {rl['round']}: "
                        f"Acc={rl['acc']:.4f} EO={rl['EO']:.4f} "
                        f"TPR0={rl['TPR0']:.4f} TPR1={rl['TPR1']:.4f}\n")
            f.write("\n")

        f.write("="*70 + "\n")
        f.write("LIMITATIONS AND ASSUMPTIONS\n")
        f.write("="*70 + "\n")
        f.write("1. Synthetic Gaussian data: real-world distributions are more complex.\n")
        f.write("2. Linear model (f = theta^T [x; a]): fairness gains may differ with\n")
        f.write("   nonlinear models.\n")
        f.write("3. Small number of rounds/clients; more rounds may yield larger effects.\n")
        f.write("4. The pipeline's fairness improvement depends on rho and other\n")
        f.write("   hyperparameters; these were not tuned here.\n")
        f.write("5. EO gap is measured on held-out test data, but test set is also\n")
        f.write("   synthetic, so it shares the same distributional assumptions.\n")

    print(f"Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
