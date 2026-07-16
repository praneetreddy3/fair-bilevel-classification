# Fair Bilevel Classification — Federated, Fair, Privacy-Aware

A collaborative (federated) binary-classification framework where each client shares only a
small **synthetic + Universum** dataset instead of raw data. The server pools these and trains
a global model that is both **accurate** and **fair** (Equal Opportunity — equal true-positive
rate across sensitive groups), with optional **differential-privacy** noise.

**Lead:** Praneet Chinthala · **Contributor:** Yanjia (German Credit)

**Applications evaluated:** Credit Risk Prediction (Default of Credit Card Clients — main
application), UCI Adult (reference dataset the method was first validated on), German Credit
(secondary, see caveats below).

---

## Setup

```bash
pip install -r requirements.txt
```

Then place the dataset files (not committed for size/licensing where noted):
- **Credit:** download "Default of Credit Card Clients" (UCI id 350) and drop the `.xls`/`.xlsx`/`.csv`
  into `CreditData/` — the loader (`pipeline/load_credit.py`) auto-detects the file and header row.
- **Adult:** `UCIAdultdataset/adult.data` + `adult.test` are already included; if missing, the
  loader falls back to `ucimlrepo` (needs internet).

## Quick start

```bash
# 1) Smoke test (credit)
python -m draft_model.run_draft --data credit --sensitive sex \
  --num_clients 3 --rounds 3 --K_inner 30 --results_file draft_results_credit_smoke.json

# 2) Recommended credit run (post-calibration)
python -m draft_model.run_draft --data credit --sensitive sex \
  --dp_enabled false --rho 0.05 --epsilon_EO 0.1 \
  --add_intercept true --tune_threshold true \
  --rounds 8 --K_inner 100 --num_clients 5 --results_file draft_results_credit.json

# 3) Recommended adult run (raw accuracy; keep threshold untuned)
python -m draft_model.run_draft --data adult --sensitive sex \
  --dp_enabled false --rho 0.1 --epsilon_EO 0.1 \
  --add_intercept true --tune_threshold false \
  --rounds 8 --K_inner 100 --num_clients 5 --results_file draft_results_adult.json

# 4) Regenerate figures from every outputs/draft_results_*.json
python plot_results.py

# 5) Rebuild the paper tables (T1-T5) + docs/PAPER_TABLES.md
python build_tables.py

# 6) Correctness checks (compile, NaN scan, sklearn-baseline parity, ablation sanity, repro)
python verify_step3.py
```

## Key flags

| Flag | Purpose |
|---|---|
| `--data {dummy,adult,2d,credit}` | dataset |
| `--sensitive {sex,race}` | sensitive attribute |
| `--dp_variant {none,pre_server,post_server,both}` `--dp_sigma` | DP placement / strength |
| `--rho` `--epsilon_EO` | fairness penalty / tolerance (trade-off knobs) |
| `--partition {iid,dirichlet}` `--dirichlet_alpha` | client heterogeneity (non-IID) |
| **`--add_intercept true`** | bias term — fixes calibration/accuracy on imbalanced data |
| **`--tune_threshold true`** | validation-calibrated decision threshold (optimizes balanced accuracy) |
| `--no_universum` / `--fairness_off` | ablations: drop Universum / zero the fairness penalty |
| `--num_clients` `--rounds` `--K_inner` | federation / training budget |
| `--seed` | reproducibility (same seed ⇒ identical output, verified) |

Do not modify `draft_model/bilevel_al.py` or `draft_model/losses.py` math without discussion —
everything above is exposed as a flag precisely so experiments don't need to touch the method.

---

## Results (calibrated, 5-seed mean ± std — see `docs/PAPER_TABLES.md` for T1–T5)

| Dataset | Sensitive attr | Baseline acc / EO | Pipeline acc / EO | Read |
|---|---|---|---|---|
| Default of Credit Card Clients | sex | 0.759 / 0.054 | 0.737 / 0.068 | pipeline ≈ baseline; baseline is already fair |
| UCI Adult (reference) | sex | 0.852 / 0.084 | 0.699 / 0.086 | cuts demographic-parity gap, EO flat, real accuracy cost |
| German Credit | foreign worker | 0.700 / 0.174 | 0.615 / 0.299 | EO **worsened** — data-limitation, see caveats |

**Honest picture, not just the win:**
- Adding an intercept term was the single biggest fix — it recovered credit accuracy from
  ~58–68% to ~77–85%. Without it the linear model over-predicts the minority class on
  imbalanced data (see `docs/RESULTS.md`).
- Once the baseline is properly calibrated, its EO gap is often already small — the method's
  earlier "dramatic fairness win" was largely fixing a broken baseline, not beating a fair one.
- **Ablations confirm the mechanism:** removing the Universum pseudo-positives clearly worsens
  EO gap on both datasets (credit 0.00→0.05, adult 0.03→0.16) — Universum is the real fairness
  driver. Zeroing the ρ penalty (`--fairness_off`) barely moves EO, because Universum's
  S-balanced construction is independent of ρ.
- German Credit's "foreign worker" sensitive attribute splits data 96%/4%; the minority group
  has too few samples to reliably estimate group TPR — a data limitation, not a method bug.
- **Known crash:** `--dirichlet_alpha 0.1` (extreme non-IID skew) can starve a client of all
  data and crash `minibatch_design.py`/`bilevel_al.py`. Not patched (core-method files);
  tracked as a documented limitation in `docs/PAPER_TABLES.md` (T4).

Full tables, sweep grid, and per-config numbers: `docs/PAPER_TABLES.md`, `outputs/tables/*.csv`.
Figures: `outputs/results_pareto.png`, `results_bars.png`, `results_convergence.png`.

## Next steps

1. Close the accuracy gap (Adult pipeline ~70% vs baseline ~85%) with a small MLP in
   `draft_model/losses.py` — the linear model is the main structural cap (see
   `docs/ENHANCEMENT_GUIDE.md`).
2. Add a minimum-group-size guard to the client partitioner so extreme Dirichlet skew degrades
   gracefully instead of crashing.
3. Re-run German Credit with a better-balanced sensitive attribute (e.g. binary age).
4. Extend 5-seed reporting to ≥10 seeds for the final paper numbers (EO gap is noisy: Adult
   std ≈ 0.057 at 5 seeds).

---

## Repository layout

```
draft_model/        core method
  run_draft.py        main entry point (loads data -> clients -> bilevel -> server -> eval)
  bilevel_al.py       bilevel Augmented-Lagrangian solver (inner theta + outer feature update)
  minibatch_design.py stratified minibatch B, synthetic D^s, Universum U
  losses.py           loss functions (linear model, smooth TPR-gap EO surrogate)
  server.py           aggregation, global training, metrics (incl. extended + threshold tuning)
  dp.py               DP variants: none / pre_server / post_server / both
  notation.py         data structures
pipeline/           dataset loaders (load_adult.py, load_credit.py, load_2d.py)
CreditData/         credit dataset file goes here (.xls/.xlsx/.csv) — not committed
UCIAdultdataset/    Adult raw files
FairSynData/         reference-paper implementation (Law/Dutch) — gitignored, teammate's code
outputs/            result JSONs + figures + tables/ ; outputs/archive/ = superseded runs (gitignored)
figures/            polished result figures
docs/               guides & reports (index below), paper PDF, project questions
scripts/            reproduction sweep scripts (sweep_*.sh)
archive/            superseded/legacy scripts kept for local reference (gitignored)
plot_results.py     regenerate figures from outputs/draft_results_*.json
build_tables.py     regenerate paper tables (T1-T5) from outputs/*.json
verify_step3.py     code-correctness checks (compile, NaN scan, baseline-vs-sklearn, ablation sanity, repro)
VSCODE_AGENT_BRIEF.md  hand-off brief for the VS Code agent (rename to CLAUDE.md to auto-load)
```

## Documentation index (`docs/`)

| File | What it's for |
|---|---|
| `docs/PROJECT_STATUS.md` | Current status summary across all three datasets (start here). |
| `docs/RESULTS.md` | Calibrated results, verdict, and caveats. |
| `docs/PAPER_TABLES.md` | The five paper tables (T1–T5) with the final numbers. |
| `docs/ALL_IN_ONE_GUIDE.md` | One-stop: strategy, results, run commands for both scenarios. |
| `docs/CREDIT_RISK_RUNBOOK.md` | Credit Risk: datasets, scenarios, why each method is preferred. |
| `docs/ENHANCEMENT_GUIDE.md` | How to raise accuracy: levers, constants, the MLP upgrade path. |
| `docs/CHANGES.md` | What changed vs the previous repo state, and how to verify the code. |
| `docs/STATUS_REPORT.md` | Short status summary (for the professor). |
| `docs/PROJECT_GUIDE.md`, `docs/WORKFLOW.md` | Original method walkthrough + architecture. |
