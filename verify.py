"""
Verification harness (docs/NEXT_STEPS.md, Step 0): self-contained correctness checks
that don't depend on any pre-existing outputs/*.json -- every check runs its own fresh
subprocess(es) and cleans up after itself, so this stays valid as the model/loaders change.

Run from project root: python verify.py
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
_TMP_FILES = []


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f"  -- {detail}"
    print(msg)
    RESULTS.append(passed)
    return passed


def _run(args, timeout=300):
    """Run `python -m draft_model.run_draft <args>`, return (returncode, stderr)."""
    cmd = [sys.executable, "-m", "draft_model.run_draft", *args]
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stderr


def _tmp_name(label):
    name = f"_verify_tmp_{label}.json"
    _TMP_FILES.append(name)
    return name


def _load_tmp(name):
    return json.load(open(os.path.join(OUT_DIR, name)))


def _cleanup():
    for name in _TMP_FILES:
        p = os.path.join(OUT_DIR, name)
        if os.path.isfile(p):
            os.remove(p)


def _has_nan(obj):
    if isinstance(obj, float):
        return obj != obj  # NaN check without importing math
    if isinstance(obj, dict):
        return any(_has_nan(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_nan(v) for v in obj)
    return False


# ---------------------------------------------------------------------------
# 1. Compiles: py_compile on every module in the repo.
# ---------------------------------------------------------------------------
def check1_py_compile():
    files = (
        glob.glob(os.path.join(PROJECT_ROOT, "draft_model", "*.py"))
        + glob.glob(os.path.join(PROJECT_ROOT, "pipeline", "*.py"))
        + glob.glob(os.path.join(PROJECT_ROOT, "*.py"))
    )
    r = subprocess.run([sys.executable, "-m", "py_compile", *files], capture_output=True, text=True)
    check("Compiles: py_compile on every module", r.returncode == 0, r.stderr.strip()[:300])


# ---------------------------------------------------------------------------
# 2. Smoke run finishes with no NaN; baseline + pipeline populated.
# ---------------------------------------------------------------------------
def check2_smoke_run():
    fname = _tmp_name("smoke")
    rc, err = _run([
        "--data", "credit", "--sensitive", "sex",
        "--num_clients", "3", "--rounds", "3", "--K_inner", "30",
        "--results_file", fname,
    ])
    if rc != 0:
        check("Smoke run finishes with no NaN; baseline+pipeline populated", False, err.strip()[:300])
        return
    d = _load_tmp(fname)
    ok = bool(d.get("baseline")) and bool(d.get("pipeline")) and not _has_nan(d)
    check("Smoke run finishes with no NaN; baseline+pipeline populated", ok)


# ---------------------------------------------------------------------------
# 3. Baseline ~= sklearn.LogisticRegression on the same features (+/-2%).
# ---------------------------------------------------------------------------
def _prep_like_run_draft(X_train, A_train, Y_train, X_test, A_test, Y_test, seed=42, val_frac=0.2, add_intercept=True):
    rng = np.random.default_rng(seed)
    n = len(Y_train)
    n_val = int(n * val_frac)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx = idx[n_val:]  # first n_val indices are the (unused here) validation split
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
    check("Baseline ~= sklearn LogisticRegression (+/-2%)", all_ok, "; ".join(details))


# ---------------------------------------------------------------------------
# 4. Regression: --data adult with default flags reproduces ~=0.846 acc / 0.012 EO.
# ---------------------------------------------------------------------------
def check4_adult_default_regression(adult_default_result=None):
    expected_acc, expected_eo, tol = 0.846, 0.012, 0.02
    if adult_default_result is None:
        fname = _tmp_name("adult_default_1")
        rc, err = _run(["--data", "adult", "--results_file", fname])
        if rc != 0:
            check("Regression: adult default flags ~= 0.846 acc / 0.012 EO", False, err.strip()[:300])
            return None
        adult_default_result = _load_tmp(fname)
    b = adult_default_result["baseline"]
    ok = abs(b["accuracy"] - expected_acc) <= tol and abs(b["EO_gap"] - expected_eo) <= tol
    check(
        "Regression: adult default flags ~= 0.846 acc / 0.012 EO",
        ok, f"got acc={b['accuracy']:.4f} EO={b['EO_gap']:.4f}",
    )
    return adult_default_result


# ---------------------------------------------------------------------------
# 5. Ablation sanity: --no_universum worsens EO gap.
# ---------------------------------------------------------------------------
def check5_ablation_sanity():
    winner_args = [
        "--data", "credit", "--sensitive", "sex",
        "--num_clients", "5", "--rounds", "8", "--K_inner", "100",
        "--add_intercept", "true", "--tune_threshold", "true",
        "--dp_enabled", "false", "--dp_variant", "none", "--rho", "0.05", "--epsilon_EO", "0.1",
        "--seed", "42",
    ]
    full_file = _tmp_name("ablation_full")
    nouni_file = _tmp_name("ablation_nouniversum")
    rc1, err1 = _run([*winner_args, "--results_file", full_file])
    rc2, err2 = _run([*winner_args, "--no_universum", "--results_file", nouni_file])
    if rc1 != 0 or rc2 != 0:
        check("Ablation sanity: --no_universum worsens EO gap", False, (err1 + err2).strip()[:300])
        return
    full_eo = _load_tmp(full_file)["pipeline"]["EO_gap"]
    nouni_eo = _load_tmp(nouni_file)["pipeline"]["EO_gap"]
    ok = nouni_eo > full_eo
    check("Ablation sanity: --no_universum worsens EO gap", ok, f"full_EO={full_eo:.4f} no_universum_EO={nouni_eo:.4f}")


# ---------------------------------------------------------------------------
# 6. Determinism: same --seed -> identical numbers.
# ---------------------------------------------------------------------------
def check6_determinism(adult_default_result=None):
    if adult_default_result is None:
        fname = _tmp_name("adult_default_a")
        rc, err = _run(["--data", "adult", "--results_file", fname])
        if rc != 0:
            check("Determinism: same seed reproduces identical numbers", False, err.strip()[:300])
            return
        adult_default_result = _load_tmp(fname)

    fname2 = _tmp_name("adult_default_b")
    rc, err = _run(["--data", "adult", "--results_file", fname2])
    if rc != 0:
        check("Determinism: same seed reproduces identical numbers", False, err.strip()[:300])
        return
    rerun = _load_tmp(fname2)
    same = (
        adult_default_result["pipeline"]["accuracy"] == rerun["pipeline"]["accuracy"]
        and adult_default_result["pipeline"]["EO_gap"] == rerun["pipeline"]["EO_gap"]
        and adult_default_result["baseline"]["accuracy"] == rerun["baseline"]["accuracy"]
    )
    check(
        "Determinism: same seed reproduces identical numbers", same,
        f"run1 acc={adult_default_result['pipeline']['accuracy']:.6f} run2 acc={rerun['pipeline']['accuracy']:.6f}",
    )


def main():
    print("=" * 70)
    print("VERIFICATION HARNESS")
    print("=" * 70)
    check1_py_compile()
    check2_smoke_run()
    check3_sklearn_baseline_gap()
    # Share one adult-default run between checks 4 and 6 so determinism reuses it as "run1".
    adult_default_result = check4_adult_default_regression()
    check5_ablation_sanity()
    check6_determinism(adult_default_result)
    _cleanup()
    print("=" * 70)
    n_pass = sum(RESULTS)
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
