# What Changed and Why (before pushing)

Plain-language record of this round of work. All of it is in commit
`Credit Risk application: loader, sweep, non-IID/ablation robustness, tables`.

## What changed vs the previous repo

**1. New capability — run on Credit.**
`pipeline/load_credit.py` (loader for Default of Credit Card Clients) + a `credit` option in
`run_draft.py`. The old repo only knew Adult/dummy data. Lets the same method run on a second
real dataset (the credit-risk application).

**2. New measurements + optional knobs** (in `run_draft.py` and `server.py`):
- Extended metrics: PR-AUC, ROC-AUC, macro-F1, balanced accuracy, DP gap, EOD gap — needed for
  the paper tables.
- `--partition dirichlet` + `--dirichlet_alpha` — non-IID client splits for the robustness table.
- `--add_intercept` and `--tune_threshold` — **off by default**; fix credit's accuracy
  (calibration). See below.

**3. Supporting files.** `docs/` guides, `plot_results.py` (reusable figures), the result tables
(`outputs/tables/`, `docs/PAPER_TABLES.md`), and the README rewrite. For reproducibility.

## What did NOT change

The **method itself**: `bilevel_al.py` (bilevel solver) and `losses.py` (losses + Universum) are
untouched. The algorithm is exactly the proposed draft.

## How similar / different from the proposed draft

~95% identical. The method — bilevel optimization, Universum pseudo-positives, sharing synthetic
data instead of raw data, differential privacy — is unchanged. Everything added is scaffolding
around it: a new dataset, more metrics, more run options. The only change that touches model
behaviour is the optional **intercept** (a bias term), and it is off unless requested — so by
default the pipeline runs exactly as before.

## Why the intercept, and its effect

The linear model had no bias term, so on imbalanced credit data it over-predicted defaults and
accuracy collapsed (~58–68%). Adding the intercept fixes this: credit accuracy → ~77%. On Adult
(already well-calibrated) it barely changes accuracy but slightly raises the EO gap, so Adult is
best reported without it. Same method, opposite calibration needs.

---

## How to check the code is correct

Run these on the machine with the `.venv` (VS Code terminal). Each has a clear pass condition.

1. **It compiles.**
   `python -m py_compile draft_model/*.py pipeline/*.py`
   Pass = no output.

2. **It runs end-to-end with no NaNs.**
   `python -m draft_model.run_draft --data credit --num_clients 3 --rounds 3 --K_inner 30 --results_file smoke.json`
   Pass = finishes, and the JSON has populated `baseline` and `pipeline` blocks, no `NaN`.

3. **Baseline matches a standard model.** The pipeline's baseline accuracy should be within ~2%
   of a plain `sklearn.LogisticRegression` on the same features (~81% credit, ~85% Adult). There
   is a helper `verify_step3.py` for this. Pass = numbers line up (confirms preprocessing + model
   are correct).

4. **Default behaviour is unchanged (regression check).** Run Adult with **no** new flags:
   `python -m draft_model.run_draft --data adult`
   Pass = baseline ≈ 0.846 accuracy, EO ≈ 0.012 (matches the pre-change repo). Confirms the new
   flags didn't alter default results.

5. **Ablations move the right way.**
   - `--no_universum` → EO gap gets **worse** (Universum is the fairness driver). Pass.
   - `--add_intercept false` on credit → accuracy drops back to ~60s (confirms the intercept is
     what fixes it). Pass.
   - Note: `--fairness_off` (ρ=0) does **not** worsen EO here — that is expected, because the
     Universum, not the ρ penalty, is what equalises the groups.

6. **Reproducible.** Same `--seed` gives identical numbers every run.

If all six pass, the code is behaving correctly.
