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
- **Non-IID client partitioning** (`--partition dirichlet`) and three additional
  applications beyond Law: UCI Adult, the UCI "Default of Credit Card Clients" dataset,
  and (secondary) UCI German Credit.

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
pipeline/            dataset loaders (adult, credit, law, german, 2d example)
scripts/             run_final.sh (final 5-seed runs, credit/adult/law) +
                     run_german.sh (secondary 5-seed run) + sweep_*.sh (grid search)
outputs/             result JSONs, figures, outputs/tables/ (T1-T5 CSVs)
docs/                guides, reports, paper tables (index below)
verify.py            verification harness (see Reproducibility)
build_tables.py       regenerate outputs/tables/T1-T5 + docs/PAPER_TABLES.md
plot_results.py       regenerate figures from the official final-seed runs (--all for every run)
CreditData/           credit .xls/.xlsx/.csv goes here — gitignored, not committed
UCIAdultdataset/      adult.data / adult.test go here — gitignored, not committed
GermanData/           german.data goes here — gitignored, not committed
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
- **German Credit** *(secondary)* (UCI id 144): download from
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data and place
  `german.data` in `GermanData/`. `pipeline/load_german.py` auto-detects the file; same
  `ucimlrepo` fallback if the folder is empty.

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

# Full final results: adult/credit/law, 5 seeds each
bash scripts/run_final.sh

# Secondary: German Credit, 5 seeds (age as sensitive attribute — see docs/PROJECT_STATUS.md)
bash scripts/run_german.sh

# Regenerate figures (official final-seed runs; pass --all for every run in outputs/)
# / paper tables (T1-T5, from outputs/draft_results_*.json)
python plot_results.py
python build_tables.py
```

### Key flags

| Flag | Purpose |
|---|---|
| `--data {dummy,adult,2d,credit,law,german}` | dataset |
| `--sensitive {sex,race,age}` | sensitive attribute (age is German-only) |
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

Final 5-seed runs (`scripts/run_final.sh` for Credit/Adult/Law, `scripts/run_german.sh`
for German — per-dataset best settings; see `docs/RESULTS.md` for exact CLI flags and
full discussion; T1-T5 tables in `docs/PAPER_TABLES.md`). German Credit is a secondary
dataset (see `docs/PROJECT_STATUS.md`) — kept out of `scripts/run_final.sh` and the T3-T5
sweep/ablation/non-IID tables (never run for it), but included here and in T1/T2 and the
figures since it's a full 5-seed result on its own.

**Performance (mean ± std, 5 seeds)**

| Dataset | Model | Accuracy | F1 | Balanced acc. | PR-AUC |
|---|---|---|---|---|---|
| Credit | baseline | 0.7594 ± 0.0202 | 0.4958 ± 0.0041 | 0.6049 ± 0.0012 | 0.4946 ± 0.0011 |
| Credit | pipeline | 0.7372 ± 0.0229 | 0.4520 ± 0.0153 | 0.6159 ± 0.0095 | 0.4410 ± 0.0141 |
| Adult  | baseline | 0.8452 ± 0.0002 | 0.6528 ± 0.0006 | 0.7660 ± 0.0004 | 0.7410 ± 0.0005 |
| Adult  | pipeline | 0.6962 ± 0.0084 | 0.5599 ± 0.0086 | 0.7383 ± 0.0085 | 0.5613 ± 0.0088 |
| Law    | baseline | 0.7786 ± 0.0144 | 0.8611 ± 0.0108 | 0.6150 ± 0.0039 | 0.9794 ± 0.0001 |
| Law    | pipeline | 0.6999 ± 0.0191 | 0.8057 ± 0.0174 | 0.6508 ± 0.0393 | 0.9578 ± 0.0112 |
| German *(secondary)* | baseline | 0.7270 ± 0.0075 | 0.5916 ± 0.0163 | 0.7079 ± 0.0131 | 0.6023 ± 0.0165 |
| German *(secondary)* | pipeline | 0.6420 ± 0.0326 | 0.5381 ± 0.0319 | 0.6567 ± 0.0301 | 0.5255 ± 0.0432 |

**Fairness (mean ± std, 5 seeds)**

| Dataset | Model | DP gap | EO gap | EOD gap |
|---|---|---|---|---|
| Credit | baseline | 0.0235 ± 0.0016 | 0.0537 ± 0.0185 | 0.0279 ± 0.0027 |
| Credit | pipeline | 0.1246 ± 0.0352 | 0.0675 ± 0.0128 | 0.1264 ± 0.0369 |
| Adult  | baseline | 0.1420 ± 0.0014 | 0.0189 ± 0.0025 | 0.0526 ± 0.0014 |
| Adult  | pipeline | 0.1351 ± 0.0586 | 0.0687 ± 0.0459 | 0.1116 ± 0.0309 |
| Law    | baseline | 0.1908 ± 0.0053 | 0.3808 ± 0.0057 | 0.4001 ± 0.0166 |
| Law    | pipeline | 0.2430 ± 0.0713 | 0.2624 ± 0.0769 | 0.2324 ± 0.0934 |
| German *(secondary)* | baseline | 0.4482 ± 0.0232 | 0.4350 ± 0.0339 | 0.4350 ± 0.0339 |
| German *(secondary)* | pipeline | 0.2022 ± 0.0840 | 0.0750 ± 0.0689 | 0.1821 ± 0.1104 |

**Headline:** on Credit and Adult, the calibrated baseline is already close to fair, so
the method mainly trades accuracy for little fairness gain. **Law** and **German** are
where the fairness intervention shows its clearest effect: both baselines are genuinely
unfair (Law EO 0.381, German EO 0.435), and the pipeline cuts that by ~31% (Law, to
0.262) and **~83% (German, to 0.075 — the largest reduction of any dataset)** at a real
but moderate accuracy cost. Ablations confirm the **Universum construction is the active
fairness mechanism** (removing it clearly worsens EO on both Credit and Adult). Full
verdict, caveats, and known limitations: `docs/RESULTS.md`.

*German Credit uses **age** as the sensitive attribute (balanced ~52/48 split) rather
than the original **foreign worker** attribute (96/4 split, unstable single-seed result)
from an earlier contributor's run — see `docs/PROJECT_STATUS.md` for that comparison and
why the fix works.*

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
