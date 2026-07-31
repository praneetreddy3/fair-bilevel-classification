# Fair Bilevel Classification — Federated, Fair, Privacy-Aware

A federated binary-classification method where each client shares only a small
**synthetic + Universum** dataset with the server — never raw data. Each client runs an
inner-outer **bilevel Augmented-Lagrangian** solver: the inner level fits a local model on
its synthetic data, the outer level uses implicit differentiation (conjugate-gradient
Hessian-vector products) to nudge that synthetic data so the server's aggregated model is
both **accurate** and **fair** (Equal Opportunity — equal true-positive rate across a
sensitive attribute), with optional **differential-privacy** noise on the client→server
payload.

## Relationship to the reference paper

This repository is a from-scratch reimplementation of the bilevel fairness method from
the accompanying reference implementation in [`FairSynData/`](FairSynData/) (the
original authors' code, including the Law School dataset it was validated on), extended
with:
- **Universum pseudo-positives** (`draft_model/minibatch_design.py`) — S-balanced
  synthetic pseudo-positive points that turn out to be the primary fairness mechanism
  (see the ablations in `docs/PAPER_TABLES.md`, T5).
- **Differential-privacy variants** (`draft_model/dp.py`) — none / pre-server /
  post-server / both, so the privacy-accuracy-fairness trade-off is a flag, not a rewrite.
- **Non-IID client partitioning** (`--partition dirichlet`) and two additional
  applications beyond Law: UCI Adult and the UCI "Default of Credit Card Clients" dataset.

`FairSynData/` is kept for reference and comparison but is gitignored (not our code).

## Repository structure

```
draft_model/        core method (this repo's code)
  run_draft.py         entry point: load -> clients -> bilevel AL -> server -> eval
  bilevel_al.py        bilevel Augmented-Lagrangian solver (inner theta, outer features)
  minibatch_design.py  stratified minibatch B, synthetic D^s, Universum U
  losses.py            loss functions (linear model, smooth TPR-gap EO surrogate)
  server.py            aggregation, global training, metrics, threshold tuning
  dp.py                DP variants: none / pre_server / post_server / both
  notation.py           data structures
pipeline/            dataset loaders (adult, credit, law, 2d example)
scripts/             run_final.sh (final 5-seed runs) + sweep_*.sh (grid search)
outputs/             result JSONs, figures, outputs/tables/ (T1-T5 CSVs)
docs/                guides, reports, paper tables (index below)
verify.py            verification harness (see Reproducibility)
build_tables.py       regenerate outputs/tables/T1-T5 + docs/PAPER_TABLES.md
plot_results.py       regenerate figures from outputs/draft_results_*.json
CreditData/           credit .xls/.xlsx/.csv goes here — gitignored, not committed
UCIAdultdataset/      adult.data / adult.test go here — gitignored, not committed
FairSynData/          reference implementation (incl. bundled Law data) — gitignored
```

## Installation

```bash
pip install -r requirements.txt
```

### Getting the data

- **Adult** (UCI id 2): download from
  https://archive.ics.uci.edu/dataset/2/adult and place `adult.data` + `adult.test` in
  `UCIAdultdataset/`. If the folder is empty, `pipeline/load_adult.py` fetches it
  automatically via `ucimlrepo` (`pip install ucimlrepo`, needs internet).
- **Default of Credit Card Clients** (UCI id 350): download from
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients and place the
  `.xls`/`.xlsx`/`.csv` in `CreditData/`. `pipeline/load_credit.py` auto-detects the file
  and header row; same `ucimlrepo` fallback if the folder is empty.
- **Law School**: bundled at `FairSynData/rawdata/law.csv` — no download needed.

## Usage

```bash
# Smoke test (fast, tiny config)
python -m draft_model.run_draft --data credit --sensitive sex \
  --num_clients 3 --rounds 3 --K_inner 30 --results_file draft_results_credit_smoke.json

# Single run, e.g. the Law School final config
python -m draft_model.run_draft --data law --sensitive race \
  --add_intercept true --tune_threshold true --dp_variant none \
  --rho 0.1 --epsilon_EO 0.1 --num_clients 5 --rounds 8 --K_inner 100 \
  --seed 1 --deterministic true --results_file draft_results_law_seed1.json

# Full final results: adult/credit/law, 5 seeds each, rebuilds tables + figures
bash scripts/run_final.sh

# Regenerate figures / paper tables from whatever is in outputs/
python plot_results.py
python build_tables.py
```

### Key flags

| Flag | Purpose |
|---|---|
| `--data {dummy,adult,2d,credit,law}` | dataset |
| `--sensitive {sex,race}` | sensitive attribute |
| `--dp_variant {none,pre_server,post_server,both}` `--dp_sigma` | DP placement / strength |
| `--rho` `--epsilon_EO` | fairness penalty / tolerance (trade-off knobs) |
| `--partition {iid,dirichlet}` `--dirichlet_alpha` | client heterogeneity (non-IID) |
| `--add_intercept true` | bias term — fixes calibration/accuracy on imbalanced data |
| `--tune_threshold true` | validation-calibrated decision threshold (maximizes balanced accuracy) |
| `--no_universum` / `--fairness_off` | ablations: drop Universum / zero the fairness penalty |
| `--num_clients` `--rounds` `--K_inner` | federation / training budget |
| `--seed` `--deterministic` | reproducibility (same seed + deterministic=true -> bit-identical output) |

`draft_model/bilevel_al.py` and `draft_model/losses.py` hold the method math — everything
above is exposed as a flag precisely so experiments don't need to touch those files.

## Results

Final 5-seed runs (`scripts/run_final.sh`, per-dataset best settings — see
`docs/RESULTS.md` for exact CLI flags and full discussion; T1-T5 tables in
`docs/PAPER_TABLES.md`).

**Performance (mean ± std, 5 seeds)**

| Dataset | Model | Accuracy | F1 | Balanced acc. | PR-AUC |
|---|---|---|---|---|---|
| Credit | baseline | 0.7594 ± 0.0202 | 0.4958 ± 0.0041 | 0.6049 ± 0.0012 | 0.4946 ± 0.0011 |
| Credit | pipeline | 0.7372 ± 0.0229 | 0.4520 ± 0.0153 | 0.6159 ± 0.0095 | 0.4410 ± 0.0141 |
| Adult  | baseline | 0.8452 ± 0.0002 | 0.6528 ± 0.0006 | 0.7660 ± 0.0004 | 0.7410 ± 0.0005 |
| Adult  | pipeline | 0.6962 ± 0.0084 | 0.5599 ± 0.0086 | 0.7383 ± 0.0085 | 0.5613 ± 0.0088 |
| Law    | baseline | 0.7786 ± 0.0144 | 0.8611 ± 0.0108 | 0.6150 ± 0.0039 | 0.9794 ± 0.0001 |
| Law    | pipeline | 0.6999 ± 0.0191 | 0.8057 ± 0.0174 | 0.6508 ± 0.0393 | 0.9578 ± 0.0112 |

**Fairness (mean ± std, 5 seeds)**

| Dataset | Model | DP gap | EO gap | EOD gap |
|---|---|---|---|---|
| Credit | baseline | 0.0235 ± 0.0016 | 0.0537 ± 0.0185 | 0.0279 ± 0.0027 |
| Credit | pipeline | 0.1246 ± 0.0352 | 0.0675 ± 0.0128 | 0.1264 ± 0.0369 |
| Adult  | baseline | 0.1420 ± 0.0014 | 0.0189 ± 0.0025 | 0.0526 ± 0.0014 |
| Adult  | pipeline | 0.1351 ± 0.0586 | 0.0687 ± 0.0459 | 0.1116 ± 0.0309 |
| Law    | baseline | 0.1908 ± 0.0053 | 0.3808 ± 0.0057 | 0.4001 ± 0.0166 |
| Law    | pipeline | 0.2430 ± 0.0713 | 0.2624 ± 0.0769 | 0.2324 ± 0.0934 |

**Headline:** on Credit and Adult, the calibrated baseline is already close to fair, so
the method mainly trades accuracy for little fairness gain. **Law** — the reference
paper's own dataset, and the most direct comparison point — is where the fairness
intervention shows its clearest effect: the baseline is genuinely unfair (EO 0.381) and
the pipeline cuts that by ~31% (to 0.262) and EOD by ~42% (0.400 → 0.232) at a real but
moderate accuracy cost. Ablations confirm the **Universum construction is the active
fairness mechanism** (removing it clearly worsens EO on both Credit and Adult). Full
verdict, caveats, and known limitations: `docs/RESULTS.md`.

*A secondary German Credit experiment (different sensitive attribute, single seed, run
by a separate contributor) is not part of the final 3-dataset pipeline above — see
`docs/PROJECT_STATUS.md` for that result and why it's reported separately.*

## Reproducibility

```bash
python verify.py
```

Runs 6 self-contained checks with no dependency on pre-existing `outputs/*.json`:
compiles cleanly, a smoke run finishes with no NaNs, the baseline matches
`sklearn.LogisticRegression` on the same features (±2%), the Adult default-flags run
matches its known regression numbers, the `--no_universum` ablation measurably worsens
the EO gap, and — critically — **running the identical command twice with the same
`--seed` produces bit-identical output**. That determinism holds because `run_draft.py`
pins single-threaded BLAS/OpenMP (env vars set before numpy/torch import), seeds
`random`/`numpy`/`torch`, and every RNG consumer in the pipeline (including
`draft_model/dp.py`'s DP noise, which previously drew from an unseeded generator) is
threaded through the same seeded `numpy.random.Generator`.

## Documentation index (`docs/`)

| File | What it's for |
|---|---|
| `docs/RESULTS.md` | Final calibrated 5-seed results (Adult/Credit/Law), verdict, and caveats. |
| `docs/PAPER_TABLES.md` | The five paper tables (T1-T5) with per-config numbers. |
| `docs/PROJECT_STATUS.md` | Status snapshot across datasets, including the secondary German Credit run. |
| `docs/ALL_IN_ONE_GUIDE.md` | One-stop: strategy, results, run commands. |
| `docs/CREDIT_RISK_RUNBOOK.md` | Credit Risk: datasets, scenarios, method rationale. |
| `docs/ENHANCEMENT_GUIDE.md` | How to raise accuracy: levers, constants, the MLP upgrade path. |
| `docs/CHANGES.md` | Changelog against earlier repo states. |
| `docs/STATUS_REPORT.md` | Dated single-seed milestone report (historical). |
| `docs/PROJECT_GUIDE.md`, `docs/WORKFLOW.md` | Method walkthrough + architecture reference. |
