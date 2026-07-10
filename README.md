# Fair Bilevel Classification — Federated, Fair, Privacy-Aware

A collaborative (federated) binary-classification framework where each client shares only a
small **synthetic + Universum** dataset instead of raw data. The server pools these and
trains a global model that is both **accurate** and **fair** (Equal Opportunity — equal
true-positive rate across sensitive groups), with optional **differential-privacy** noise.

**Active application:** Credit Risk Prediction (Default of Credit Card Clients). The method
was first validated on UCI Adult and reused here via a new data loader.

---

## Repository layout

```
draft_model/        core method
  run_draft.py        main entry point (loads data → clients → bilevel → server → eval)
  bilevel_al.py       bilevel Augmented-Lagrangian solver (inner θ + outer feature update)
  minibatch_design.py stratified minibatch B, synthetic Dˢ, Universum U
  losses.py           loss functions (linear model, smooth TPR-gap EO surrogate)
  server.py           aggregation, global training, metrics (incl. extended + threshold)
  dp.py               DP variants: none / pre_server / post_server / both
  notation.py         data structures
pipeline/           dataset loaders (load_adult.py, load_credit.py, load_2d.py)
CreditData/         put the credit dataset file here (.xls/.csv)
UCIAdultdataset/    Adult raw files
outputs/            result JSONs + canonical figures
figures/            polished result figures
docs/               all guides & reports (see index below)
plot_results.py     regenerate figures from outputs/draft_results_*.json
VSCODE_AGENT_BRIEF.md  hand-off brief for the VS Code coding agent (rename to CLAUDE.md to auto-load)
```

## Documentation index (`docs/`)

| File | What it's for |
|---|---|
| `docs/ALL_IN_ONE_GUIDE.md` | One-stop: strategy, results, run commands for both scenarios. |
| `docs/CREDIT_RISK_RUNBOOK.md` | Credit Risk: datasets, scenarios, why each method is preferred. |
| `docs/ENHANCEMENT_GUIDE.md` | How to raise accuracy: levers, constants, the logical move. |
| `docs/RESULTS.md` | Current results table + figures + open issues. |
| `docs/STATUS_REPORT.md` | Short status summary (for the professor). |
| `docs/PROJECT_GUIDE.md`, `docs/WORKFLOW.md` | Original method walkthrough + architecture. |

---

## Quick start

```bash
# 1) Smoke test (credit)
python -m draft_model.run_draft --data credit --sensitive sex \
  --num_clients 3 --rounds 3 --K_inner 30 --results_file draft_results_credit_smoke.json

# 2) Fairness-track (recommended) WITH the new calibration fixes
python -m draft_model.run_draft --data credit --sensitive sex \
  --dp_enabled true --dp_variant pre_server --dp_sigma 0.25 --rho 0.1 --epsilon_EO 0.05 \
  --add_intercept true --tune_threshold true \
  --rounds 8 --K_inner 100 --num_clients 5 --results_file draft_results_credit.json

# 3) Accuracy-track
python -m draft_model.run_draft --data credit --sensitive sex --dp_enabled false \
  --rho 0.05 --epsilon_EO 0.25 --add_intercept true --tune_threshold true \
  --rounds 8 --K_inner 100 --num_clients 5 --results_file draft_results_credit_accuracy.json

# 4) Figures
python plot_results.py
```

Swap `--data credit` → `--data adult` for the Adult comparison.

## Key flags

| Flag | Purpose |
|---|---|
| `--data {dummy,adult,2d,credit}` | dataset |
| `--sensitive {sex,race}` | sensitive attribute |
| `--dp_variant {none,pre_server,post_server,both}` `--dp_sigma` | DP placement / strength |
| `--rho` `--epsilon_EO` | fairness penalty / tolerance (trade-off knobs) |
| `--partition {iid,dirichlet}` `--dirichlet_alpha` | client heterogeneity (non-IID) |
| **`--add_intercept true`** | bias term — fixes credit calibration/accuracy |
| **`--tune_threshold true`** | validation-calibrated decision threshold |
| `--num_clients` `--rounds` `--K_inner` | federation / training budget |

## Current results (single seed)

| Config | Dataset | Baseline acc / EO | Pipeline acc / EO |
|---|---|---|---|
| Fairness-track | Credit | 0.682 / 0.438 | 0.595 / **0.021** |
| Fairness-track | Adult | 0.846 / 0.012 | 0.653 / 0.119 |

Fairness works (EO gap collapses on credit). Accuracy fix (`--add_intercept`,
`--tune_threshold`) is in the code and awaiting a re-run. Full table: `docs/RESULTS.md`.

## Status & next steps

1. Re-run the four configs with `--add_intercept true --tune_threshold true`.
2. Confirm with 5-seed mean ± std.
3. (Bigger accuracy lever) replace the linear model with a small MLP — see `docs/ENHANCEMENT_GUIDE.md`.
