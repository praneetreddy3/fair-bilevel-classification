# Reference-model (FairSynData) setup for Adult / Credit

`FairSynData/` is a teammate's reference implementation, checked out locally into this repo but
**gitignored** (see the top-level `.gitignore`: `FairSynData/`). It ships dataset support for
`law` and `dutch` only. This folder holds the files created/modified to add `adult` and `credit`
to it and get the reference model's accuracy/EO-gap numbers used in `docs/COMPARISON.md`, kept
here under version control so the setup survives a fresh `FairSynData/` checkout.

Background and the full convergence investigation (two real bugs found and fixed — a feature-
scale issue and a wrong-column bug, neither of which is a change to FairSynData's optimization
algorithm) are written up in `docs/COMPARISON.md`. This README is just "how do I reproduce it."

## What's in this folder

Paths below are relative to a `FairSynData/` checkout root — copy each file to the same
relative path there.

| File here | Goes to (in `FairSynData/`) | What it does |
|---|---|---|
| `mydatasets/Adult.py` | `mydatasets/Adult.py` | Adult dataset class (sensitive=`sex`, target=`income`), mirrors `Law.py`'s structure |
| `mydatasets/Credit.py` | `mydatasets/Credit.py` | Credit dataset class (sensitive=`SEX`, target=`default`) |
| `mydatasets/DatasetFactory.py` | `mydatasets/DatasetFactory.py` | Registers `Adult`/`Credit` alongside `Law`/`Dutch` |
| `mycodes/datasetsPreprocess.py` | `mycodes/datasetsPreprocess.py` | Adds `adult`/`credit` branches to `preprocess_data`/`scale_data`/`split_data_to_tensors` (same pattern as the existing `law`/`dutch` branches), plus a non-numeric-column guard in `detect_outliers` (diagnostic-only function, its output is logged, never used to filter data) |
| `mycodes/myParams.py` | `mycodes/myParams.py` | `AlgorithmParams.K`: `3` → `20` (outer bilevel iterations — the smoke-test default was too small to let the fairness-constrained synthetic data actually converge; see `docs/COMPARISON.md` for why this wasn't actually the main blocker) |
| `rawdata/_prep_adult_credit.py` | `rawdata/_prep_adult_credit.py` | Builds `rawdata/adult.csv` / `credit.csv` from `UCIAdultdataset/` / `CreditData/`. Includes the two data-prep fixes: `log1p` on heavy-tailed columns (`capital-gain`/`capital-loss`, `BILL_AMT*`/`PAY_AMT*`) and explicit column reordering so the sensitive attribute is physically second-to-last and the target last (FairSynData's `trainAndPredict.py` reads them positionally via `iloc[:, -2]`/`iloc[:, -1]`) |
| `_gen_splits.py` | `_gen_splits.py` | Driver that calls `generate_datasets.py`'s `split_existing_dataset()` for a named dataset (5 equal clients, matching our own pipeline's `--num_clients 5`) and copies the output into `datasets/dummy_run/` with the naming pattern `main.py`'s `NO_DB=1` mode expects |

None of these touch FairSynData's optimization/fairness-constraint code
(`mycodes/exactInnerMinOpt.py`, `mycodes/solveNablaOuter.py`, `mycodes/myFLAlg.py`, etc.) —
only dataset registration, data-prep, and one stated hyperparameter default.

`Law` itself needed **no fixes** — its raw `rawdata/law.csv` already ends with
`[..., race, pass_bar]` (the positional convention) and has no catastrophic-scale column.

## How to reproduce the reference numbers

Requires a Python env with `torch`/`pandas`/`sklearn` (the reference numbers in this repo were
produced with a local Anaconda base env — see `docs/COMPARISON.md`'s "Repro" section for the
exact commands). `sdv`/`ctgan` are **not** required — the runs use `syn_2_skip=true`, which
skips CTGAN augmentation entirely and never imports those packages.

1. Get `adult.data`/`adult.test` (UCI Adult) and a `default of credit card clients.xls`-style
   file (UCI Credit) into `UCIAdultdataset/` and `CreditData/` at the project root (same raw
   sources `pipeline/load_adult.py`/`load_credit.py` use).
2. Copy the files from the table above into your `FairSynData/` checkout at the listed paths.
3. From inside `FairSynData/`:
   ```
   python rawdata/_prep_adult_credit.py         # writes rawdata/adult.csv, rawdata/credit.csv
   rm -f datasets/dummy_run/*.csv
   python _gen_splits.py adult                  # or credit, or law (no prep script needed for law)
   NO_DB=1 python main.py --dataset_name adult --syn_2_skip true
   grep -n "rho_o=\|(NO_DB) Wrote metrics" logs/1_algorithm.log
   ```
4. Repeat step 3's last three lines for `credit` and `law` (clearing `datasets/dummy_run/*.csv`
   between datasets — `main.py`'s `NO_DB=1` loader reads every CSV in that folder regardless of
   dataset name).

## Caveats

- **Single seed only.** FairSynData has no `--seed` CLI argument — `set_seed(seed=42)` is called
  at fixed points in `main.py` with no plumbing to vary it. The numbers in `docs/COMPARISON.md`
  are one run each at that hardcoded seed, not a 5-seed mean like our own method's numbers.
- **Headline number uses `rho_o=100`** (moderate constraint strength) out of the 5-point sweep
  `[0, 10, 100, 1000, 10000]` FairSynData runs by default — the full sweep is in
  `outputs/tables/COMPARISON.csv` for each dataset.
