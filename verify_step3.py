"""
STEP 3 verification: correctness checks for the credit/adult fair-bilevel runs.
Run from project root: python verify_step3.py
Exits non-zero if any check fails.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f"  -- {detail}"
    print(msg)
    RESULTS.append(passed)
    return passed


def check1_py_compile():
    files = glob.glob(os.path.join(PROJECT_ROOT, "draft_model", "*.py")) + \
            glob.glob(os.path.join(PROJECT_ROOT, "pipeline", "*.py"))
    r = subprocess.run([sys.executable, "-m", "py_compile", *files], capture_output=True, text=True)
    check("py_compile draft_model/*.py pipeline/*.py", r.returncode == 0, r.stderr.strip()[:300])


def _iter_result_jsons():
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "draft_results_*.json"))):
        name = os.path.basename(p)
        if name in ("draft_results.json",):
            continue
        try:
            yield name, json.load(open(p))
        except Exception:
            continue


def _has_nan(obj):
    if isinstance(obj, float):
        return obj != obj  # NaN check without importing math
    if isinstance(obj, dict):
        return any(_has_nan(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_nan(v) for v in obj)
    return False


def check2_no_nans():
    bad = []
    for name, d in _iter_result_jsons():
        if _has_nan(d):
            bad.append(f"{name}: NaN present")
            continue
        if not d.get("baseline") or not d.get("pipeline"):
            bad.append(f"{name}: missing baseline/pipeline block")
    check("No NaNs; baseline+pipeline populated in every result JSON", len(bad) == 0, "; ".join(bad[:5]))


def _prep_like_run_draft(X_train, A_train, Y_train, X_test, A_test, Y_test, seed=42, val_frac=0.2, add_intercept=True):
    rng = np.random.default_rng(seed)
    n = len(Y_train)
    n_val = int(n * val_frac)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_val, A_val, Y_val = X_train[val_idx], A_train[val_idx], Y_train[val_idx]
    X_tr, A_tr, Y_tr = X_train[train_idx], A_train[train_idx], Y_train[train_idx]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.astype(np.float64))
    X_te = scaler.transform(X_test.astype(np.float64))

    if add_intercept:
        X_tr = np.hstack([X_tr, np.ones((len(X_tr), 1))])
        X_te = np.hstack([X_te, np.ones((len(X_te), 1))])

    return (X_tr, A_tr, Y_tr), (X_te, A_test, Y_test)


def check3_sklearn_baseline_gap():
    from sklearn.linear_model import LogisticRegression
    from draft_model.server import train_global_ridge_erm, compute_eo_gap_and_accuracy
    from pipeline.load_credit import prepare_credit_for_draft
    from pipeline.load_adult import prepare_adult_for_draft

    expected = {"credit": 0.81, "adult": 0.84}
    loaders = {
        "credit": lambda: prepare_credit_for_draft(sensitive="sex"),
        "adult": lambda: prepare_adult_for_draft(sensitive="sex", use_ucimlrepo=True),
    }
    all_ok = True
    details = []
    for ds, loader in loaders.items():
        (X_train, A_train, Y_train), (X_test, A_test, Y_test) = loader()
        (X_tr, A_tr, Y_tr), (X_te, A_te, Y_te) = _prep_like_run_draft(
            X_train, A_train, Y_train, X_test, A_test, Y_test
        )

        # Project's own ERM training, raw (untuned) threshold -- apples-to-apples with sklearn @0.5.
        zeta = np.zeros(X_tr.shape[1] + 1)
        theta = train_global_ridge_erm(X_tr, A_tr, Y_tr, zeta=zeta, lambda_theta=1e-4, max_iter=500, lr=0.05)
        _, _, _, pipeline_acc, _ = compute_eo_gap_and_accuracy(theta, X_te, A_te, Y_te, threshold=0.0)

        Xa_tr = np.hstack([X_tr, A_tr.reshape(-1, 1)])
        Xa_te = np.hstack([X_te, A_te.reshape(-1, 1)])
        clf = LogisticRegression(max_iter=2000).fit(Xa_tr, Y_tr)
        sklearn_acc = clf.score(Xa_te, Y_te)

        gap = abs(pipeline_acc - sklearn_acc)
        near_expected = abs(sklearn_acc - expected[ds]) < 0.03
        ok = gap <= 0.02
        all_ok = all_ok and ok
        details.append(f"{ds}: pipeline={pipeline_acc:.4f} sklearn={sklearn_acc:.4f} gap={gap:.4f} (sklearn~expected={near_expected})")
    check("Pipeline baseline within ~2% of sklearn LogisticRegression", all_ok, "; ".join(details))


def _load(name):
    p = os.path.join(OUT_DIR, name)
    if not os.path.isfile(p):
        return None
    return json.load(open(p))


def check4_ablation_sanity():
    pairs = [
        ("credit", "draft_results_credit_none_rho0.05_eps0.1.json"),
        ("adult", "draft_results_adult_notune_seed42.json"),  # same seed/config as the ablation runs
    ]
    all_ok = True
    details = []
    for ds, winner_file in pairs:
        winner = _load(winner_file)
        fairness_off = _load(f"draft_results_{ds}_ablation_fairnessoff.json")
        no_universum = _load(f"draft_results_{ds}_ablation_nouniversum.json")
        if not (winner and fairness_off and no_universum):
            all_ok = False
            details.append(f"{ds}: missing one of winner/fairness_off/no_universum JSON")
            continue
        w_eo = winner["pipeline"]["EO_gap"]
        fo_eo = fairness_off["pipeline"]["EO_gap"]
        nu_eo = no_universum["pipeline"]["EO_gap"]
        ok = fo_eo > w_eo and nu_eo > w_eo
        all_ok = all_ok and ok
        details.append(f"{ds}: winner_EO={w_eo:.4f} fairness_off_EO={fo_eo:.4f} no_universum_EO={nu_eo:.4f}")
    check("fairness_off / no_universum ablations both worsen EO gap vs winner", all_ok, "; ".join(details))


def check5_reproducibility():
    import subprocess as sp
    tmp_file = "draft_results_credit_repro_check.json"
    cmd = [
        sys.executable, "-m", "draft_model.run_draft", "--data", "credit",
        "--num_clients", "5", "--rounds", "8", "--K_inner", "100", "--sensitive", "sex",
        "--add_intercept", "true", "--tune_threshold", "true",
        "--dp_enabled", "false", "--dp_variant", "none", "--rho", "0.05", "--epsilon_EO", "0.1",
        "--seed", "42", "--results_file", tmp_file,
    ]
    r = sp.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    ok = r.returncode == 0
    detail = ""
    if ok:
        original = _load("draft_results_credit_none_rho0.05_eps0.1.json")
        rerun = _load(tmp_file)
        if original is None or rerun is None:
            ok = False
            detail = "missing original or rerun file"
        else:
            same = (
                original["pipeline"]["accuracy"] == rerun["pipeline"]["accuracy"]
                and original["pipeline"]["EO_gap"] == rerun["pipeline"]["EO_gap"]
                and original["baseline"]["accuracy"] == rerun["baseline"]["accuracy"]
            )
            ok = same
            detail = f"orig acc={original['pipeline']['accuracy']:.6f} rerun acc={rerun['pipeline']['accuracy']:.6f}"
        os.remove(os.path.join(OUT_DIR, tmp_file))
    else:
        detail = r.stderr.strip()[:300]
    check("Same seed (42) reproduces identical credit-winner numbers", ok, detail)


def main():
    print("=" * 70)
    print("STEP 3 VERIFICATION")
    print("=" * 70)
    check1_py_compile()
    check2_no_nans()
    check3_sklearn_baseline_gap()
    check4_ablation_sanity()
    check5_reproducibility()
    print("=" * 70)
    n_pass = sum(RESULTS)
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
