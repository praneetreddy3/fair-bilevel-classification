# Our method vs. reference (FairSynData) model

## Result

| Dataset | Our method — Accuracy | Our method — EO gap | Reference (FairSynData) — Accuracy | Reference (FairSynData) — EO gap |
|---|---|---|---|---|
| Adult  | 0.6962 ± 0.0084 | 0.0687 ± 0.0459 | 0.7738 | 0.1192 |
| Credit | 0.7372 ± 0.0229 | 0.0675 ± 0.0128 | 0.7445 | 0.0098 |
| Law    | 0.6999 ± 0.0191 | 0.2624 ± 0.0769 | **blocked** | **blocked** |

"Our method" columns are the pipeline rows from `outputs/tables/T1_main_performance.csv` /
`T2_fairness.csv` (5-seed final runs, see `docs/RESULTS.md`). Reference-model numbers are a
single run each (see "Seeds" below). CSV mirror: `outputs/tables/COMPARISON.csv`, which also
carries FairSynData's full 5-point `rho_o` sweep per dataset, not just the headline value.

Law's reference cell is still blocked — the convergence fixes below (scale + column-order) were
diagnosed on Adult/Credit and not re-applied to Law in this run; see "What's still blocked".

## What changed since the last attempt

The previous NO_DB/`syn_2_skip` run used FairSynData's smoke-test defaults (`K=3`,
`epochs=1`) and produced degenerate results (loss frozen at exactly ln 2 on Adult). Dr. Mousavi
authorized tuning the reference model's iteration budget to realistic values. Investigating with
that authorization turned up a different, more specific set of problems than "budget too small":

1. **`K=3` → `K=20`** (`FairSynData/mycodes/myParams.py`, `AlgorithmParams.K`) — genuinely a
   smoke-test value for the outer bilevel loop that refines the fairness-constrained synthetic
   data `xhat`. No CLI flag exposes this (checked `utils/parse_args.py`); it's a direct
   default-value edit. `epochs`/`max_iter`/`optimSet` were **not** touched — FairSynData's
   custom L-BFGS (`mycodes/mylbfgs.py`) already defaults to `max_iter=1000`, and `epochs=1` is
   correct by design for an L-BFGS optimizer driven by a full-batch closure (one `step()` call
   already runs up to `max_iter` internal line-search iterations).

2. **Adult's `capital-gain`/`capital-loss` scale bug.** These columns are ~91%/95% zero, so
   their IQR is 0. FairSynData's `RobustScaler`-based `scale_data()` (unmodified) falls back to
   `scale_=1` for zero-IQR columns (sklearn's documented behavior), leaving them completely
   unscaled — `capital-gain` up to 99,999 next to every other feature's roughly -4 to +9 robust-
   scaled range. That ~5-order-of-magnitude imbalance broke the line search on the very first
   step (theta never left its all-zero init). **Fix:** `log1p` on both columns in my own
   `FairSynData/rawdata/_prep_adult_credit.py` (not FairSynData code) — this doesn't give the
   column a nonzero IQR (it's still >75% zero after the transform), but it bounds the nonzero
   tail to ~0-11.5 instead of ~0-99,999, which is what actually mattered for the scaler's
   zero-IQR passthrough.

3. **Credit's `BILL_AMT*`/`PAY_AMT*` scale issue**, same class of problem — `PAY_AMT2` has a
   raw max of ~1.68M against a median of ~2,000. One client hit a NaN gradient
   (`norm_grad_2_xhat is nan`) mid-run at `K=20`. **Fix:** signed `log1p`
   (`sign(x) * log1p(|x|)`, since `BILL_AMT*` can be negative) on both column families, same
   script.

4. **A real, separate bug: wrong column read as the sensitive attribute.** After fixing 1–3,
   Credit still crashed with `ZeroDivisionError` in `mycodes/dataArrange.py:arrange_pred_details`
   at the *system*-level prediction (not per-client) — division by the count of
   `sensitive_attr == 1`, i.e. that count was zero for the whole concatenated test set. Tracing
   it: `mycodes/trainAndPredict.py:predict_client_test_data` reads the sensitive attribute and
   target **positionally** — `test_data.iloc[:, -2]` / `iloc[:, -1]` — not by column name. Law's
   and Dutch's raw CSVs happen to already end with `[..., sensitive, target]` by construction,
   but neither `rawdata/adult.csv` nor `rawdata/credit.csv` actually had the sensitive column in
   that position (Credit's `SEX` was the 2nd column out of 24, not the 2nd-to-last; Adult's
   `sex` also wasn't literally last-but-one). For Credit this crashed outright; for Adult it very
   likely explains the earlier run's suspicious `TPR_group1 = 0.0` result (some other column,
   not `sex`, was silently read as "sensitive attribute" — no crash, because that column had a
   nonzero count of value `1`, but it wasn't measuring the sex-based EO gap at all). **Fix:**
   both `build_adult()` and `build_credit()` in `_prep_adult_credit.py` now explicitly reorder
   columns to end with `[..., sensitive, target]`, matching `Adult.py`/`Credit.py`'s declared
   `all_columns` order. This is a real bug in my own data-prep script, not in FairSynData —
   FairSynData's positional convention is legitimate given Law/Dutch's raw files already satisfy
   it; I just hadn't matched it for the two new datasets.

None of the above touches FairSynData's optimization/fairness-constraint code
(`mycodes/exactInnerMinOpt.py`, `solveNablaOuter.py`, `myFLAlg.py`, etc.) — only `K` (a stated
hyperparameter default, explicitly authorized) and my own CSV-generation script.

## Convergence sanity check

After the fixes, both datasets show `obj_val_P`/`loss_real` moving substantially off `ln(2)`
(e.g. Adult: ~0.69 → ~0.46 by outer iteration 20) and — the check that actually matters — a
clean, monotonic fairness/accuracy tradeoff across the `rho_o` sweep `[0, 10, 100, 1000, 10000]`
(higher `rho_o` = stronger fairness constraint):

| rho_o | Adult accuracy | Adult EO gap | Credit accuracy | Credit EO gap |
|---|---|---|---|---|
| 0     | 0.7780 | 0.2141 | 0.7683 | 0.2327 |
| 10    | 0.7760 | 0.2005 | 0.7698 | 0.0686 |
| 100   | 0.7738 | 0.1192 | 0.7445 | 0.0098 |
| 1000  | 0.7619 | 0.0522 | 0.7247 | 0.0078 |
| 10000 | 0.7605 | 0.0434 | 0.7225 | 0.0068 |

EO gap decreases monotonically as `rho_o` increases on both datasets, with a modest accuracy
cost — exactly the behavior a working fairness-constrained method should show, and the opposite
of the earlier frozen/degenerate runs. `TPR_group0 != TPR_group1` at every setting (not a
constant-class predictor). The headline numbers in the table above use **`rho_o=100`** (a
moderate constraint strength) — the two methods don't share a comparable "rho" axis, so this is
a reasonable representative point rather than a tuned/cherry-picked best case; the full sweep is
in `outputs/tables/COMPARISON.csv` for transparency.

## What's still blocked / out of scope for this run

- **Law's reference cell.** The scale bug and column-order bug were diagnosed and fixed for
  Adult/Credit specifically; I didn't re-run Law under `K=20` to check whether it has its own
  version of either issue (Law's raw CSV already satisfies the positional `[..., sensitive,
  target]` convention, so the column-order bug likely doesn't apply, but that hasn't been
  verified against a live run in this session).
- **Seeds.** FairSynData has no `--seed` CLI argument — `set_seed(seed=42)` is called at fixed
  points in `main.py` with no plumbing to vary it. Adding that would be new code, not a data-prep
  or hyperparameter fix, and wasn't part of what was authorized. Both reference numbers above are
  a single run at the codebase's one supported seed (42), not a 5-seed mean like our own method's
  numbers — stated plainly rather than presented as equivalent.
- **CTGAN / full DB-backed path.** Stayed on `NO_DB=1` + `syn_2_skip=true`. `sdv`/`ctgan` aren't
  installed in any available Python environment on this machine (`.venv`, system Python, or the
  Anaconda base env that has `torch`) — the fixes above were sufficient to get real convergence
  without needing that path, so it wasn't attempted.

## Repro

```
cd FairSynData
"C:\Users\HP\anaconda3\python.exe" rawdata\_prep_adult_credit.py   # regenerates rawdata/{adult,credit}.csv
rm -f datasets/dummy_run/*.csv
"C:\Users\HP\anaconda3\python.exe" _gen_splits.py adult   # or credit
NO_DB=1 "C:\Users\HP\anaconda3\python.exe" main.py --dataset_name adult --syn_2_skip true
grep -n "rho_o=\|(NO_DB) Wrote metrics" logs/1_algorithm.log   # full 5-point sweep
```

`FairSynData/` is gitignored, so `Adult.py`, `Credit.py`, `rawdata/adult.csv`, `rawdata/credit.csv`,
the `mycodes/datasetsPreprocess.py`/`mycodes/myParams.py` edits, and the `_gen_splits.py` /
`rawdata/_prep_adult_credit.py` helper scripts exist on disk but are not tracked by this repo's
git history.
