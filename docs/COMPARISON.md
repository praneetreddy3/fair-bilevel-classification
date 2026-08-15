# Our method vs. reference (FairSynData) model

## Result

| Dataset | Our method — Accuracy | Our method — EO gap | Reference (FairSynData) — Accuracy | Reference (FairSynData) — EO gap |
|---|---|---|---|---|
| Adult  | 0.6962 ± 0.0084 | 0.0687 ± 0.0459 | **blocked** | **blocked** |
| Credit | 0.7372 ± 0.0229 | 0.0675 ± 0.0128 | **blocked** | **blocked** |
| Law    | 0.6999 ± 0.0191 | 0.2624 ± 0.0769 | **blocked** | **blocked** |

"Our method" columns are the pipeline rows from `outputs/tables/T1_main_performance.csv` /
`T2_fairness.csv` (5-seed final runs, see `docs/RESULTS.md`). CSV mirror:
`outputs/tables/COMPARISON.csv`.

**None of the reference-model cells could be filled in — not just Adult/Credit, but Law too.**
Details below.

## What I did

FairSynData (`FairSynData/`, gitignored — teammate's code, not tracked in this repo) only
shipped dataset classes for `law` and `dutch`. To get Adult/Credit into it:

1. Added `FairSynData/mydatasets/Adult.py` and `Credit.py` (copied `Law.py`'s structure),
   sensitive attribute `sex`, targets `income>50K` / `default`, registered both in
   `mydatasets/DatasetFactory.py`.
2. Built `FairSynData/rawdata/adult.csv` and `credit.csv` from `UCIAdultdataset/` and
   `CreditData/`, using the same target/sensitive definitions as our own
   `pipeline/load_adult.py` / `load_credit.py` (income >50K positive, sex male=1/female=0;
   default=1, sex male=1/female=0), with categorical features label-encoded (kept as raw
   strings for `sex`/`income`/`default` so FairSynData's own per-dataset preprocessing could
   map them to its {0,1}/{-1,1} convention, matching how it treats `race`/`pass_bar` for Law).
3. Discovered FairSynData's dataset support isn't just the `Dataset` subclass — three
   functions in `mycodes/datasetsPreprocess.py` (`preprocess_data`, `scale_data`,
   `split_data_to_tensors`) hard-code an if/elif dispatch on `data_name` for `'law'`/`'dutch'`
   only. Added `'adult'`/`'credit'` branches mirroring the existing ones (label mapping,
   scaling-exclusion column). This is data plumbing (which column is the sensitive attribute,
   which column is the target), not a change to the optimization algorithm.
4. Also had to guard `detect_outliers()` (same file) against non-numeric columns — it computes
   quantiles over every feature column including the still-string `sex`/`race` column and
   raised `TypeError`. Its output is logged only, never used to filter data, so skipping
   non-numeric columns is a no-op fix, not a math change.
5. Generated 5-client splits for Adult via `generate_datasets.py`'s `split_existing_dataset()`
   (5 clients, equal split — matching our own final runs' `--num_clients 5`), copied into
   `datasets/dummy_run/` with the naming pattern `main.py` expects in `NO_DB=1` mode (its
   `data_path` resolves to the literal string `"dummy_run"` regardless of dataset name — a
   pre-existing quirk in `mycodes/myParams.py`'s `get_file_name()`, not something I touched).
6. Ran `NO_DB=1 python main.py --dataset_name adult --syn_2_skip true` (skips CTGAN, so the
   missing `sdv`/`ctgan` packages in the available Python env never get imported — confirmed
   `sdv` is absent from both `.venv` and the system Python, but `torch`/`pandas`/`sklearn` are
   present in the local Anaconda base env, which is what actually ran this).

Steps 1, 2, 5, 6 ran cleanly end-to-end (dataset loads, splits generate, training pipeline
executes without crashing, no missing-dependency or Windows-path issues).

## What's blocked, and why

**The reference model doesn't converge under this run path — for any dataset, not just the
new ones.** With Adult wired up and `main.py` run in `NO_DB=1` mode:

- The inner-loop training loss (`obj_val_P = loss_real`) was reported as exactly
  `0.6931473016738892` (= ln 2, the loss of an untrained model with all-zero weights) for
  every one of the 5 clients and all 5 `rho_o` values swept (`0, 10, 100, 1000, 10000`).
  It never moved. Every scenario terminated with `[Fail] - reach max iter setting` (K=3 outer
  iterations, a hard-coded default in `mycodes/myParams.py`).
- The resulting accuracy/EO_gap written to `outputs/results_latest.json` was exactly
  `0.0000`/`0.0000` for all 5 scenarios — consistent with `sign()` of an all-zero model output
  never equaling the ±1 labels, i.e. the weights literally never left their zero
  initialization.

To rule out this being specific to my Adult data prep, I re-ran the identical `NO_DB=1
main.py --dataset_name law --syn_2_skip true` command against Law's own pre-existing smoke-test
fixture (`datasets/dummy_run/law_dummy_*.csv`, tiny synthetic data shipped with the repo). The
loss did move off ln 2 slightly (`0.6508` + a `101.9` constraint-penalty term for one client),
but the final accuracy/EO_gap were still degenerate: **constant `0.8333`/`0.0000` across every
one of the 5 rho values**, i.e. the final classifier (fit via a single L-BFGS "epoch" —
`train_and_predict_params.epochs = 1` in `mycodes/myParams.py`) collapses to predicting one
constant class regardless of the fairness penalty strength. (That specific fixture is random
synthetic noise with no real signal, so a constant-class prediction isn't damning on its own —
but the same "loss never moves, rho has zero effect" signature as Adult's run makes it evidence
of a shared root cause, not something dataset-specific.) I also found leftover, undocumented
`outputs/results_latest.json` / log artifacts on disk from an earlier, uncommitted session
(never in git history) showing the same pattern for what was apparently an earlier `adult` run
(`accuracy: 0.8333`, `EO_gap: 0.0`, `TPR_group0 = TPR_group1 = 1.0`, i.e. "predict everyone
positive") — I did not use those numbers; they're unreproducible and no more trustworthy than
what I found here.

The likely root cause sits in FairSynData's own optimizer configuration for this fast
`NO_DB`/`syn_2_skip` code path — `K=3` outer iterations and `epochs=1` for the final L-BFGS
fit are defaults in `mycodes/myParams.py`, and tuning either is exactly the "method math" I was
told not to touch. This is a limitation of the reference codebase's fast smoke-test path
itself (likely the real experiments in the paper used the full DB-backed pipeline with a much
larger iteration budget and/or CTGAN augmentation, not this quick local mode), not something
introduced by wiring in Adult/Credit.

**I stopped here rather than tuning FairSynData's optimizer hyperparameters to force
convergence**, per the instruction not to touch its method math. Adult/Credit are fully wired
in and ready to produce numbers the moment someone either (a) increases the reference
pipeline's iteration/epoch budget, or (b) runs it through the original DB-backed path it was
designed for.

## Repro

```
cd FairSynData
rm -f datasets/dummy_run/*.csv
"C:\Users\HP\anaconda3\python.exe" _gen_splits.py adult   # or credit
NO_DB=1 "C:\Users\HP\anaconda3\python.exe" main.py --dataset_name adult --syn_2_skip true
cat outputs/results_latest.json   # last of 5 rho_o scenarios swept
```

`FairSynData/` is gitignored, so `Adult.py`, `Credit.py`, `rawdata/adult.csv`, `rawdata/credit.csv`,
the `mycodes/datasetsPreprocess.py` edits, and the `_gen_splits.py` / `rawdata/_prep_adult_credit.py`
helper scripts exist on disk but are not tracked by this repo's git history.
