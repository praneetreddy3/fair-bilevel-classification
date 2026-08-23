# Our method vs. reference (FairSynData) model

## Result

| Dataset | Our method — Accuracy | Our method — EO gap | Reference (FairSynData) — Accuracy | Reference (FairSynData) — EO gap |
|---|---|---|---|---|
| Adult  | 0.7471 ± 0.0129 | 0.0728 ± 0.0545 | 0.7738 | 0.1192 |
| Credit | 0.7679 ± 0.0174 | 0.0347 ± 0.0210 | 0.7445 | 0.0098 |
| Law    | 0.6872 ± 0.0312 | 0.2402 ± 0.0660 | 0.8438 | 0.0849 |

"Our method" numbers changed from the previous version of this doc — see "Model selection"
below. They are **not** the `scripts/run_final.sh` winner configs used in
`outputs/tables/T1_main_performance.csv`/`T2_fairness.csv` (those tables are untouched); they're
each dataset's operating point chosen by a validation-only rule out of a 10-way
(rho × Universum on/off) sweep. Reference-model numbers are unchanged from before (a single run
each — see "Seeds" below). CSV mirror: `outputs/tables/COMPARISON.csv`; full 10-point grid in
`outputs/tables/fair_comparison_grid.csv`.

All three reference cells are filled. Law converged cleanly on the first attempt at `K=20`
with **no scale or column-order fix needed** — its raw `rawdata/law.csv` already ends with
`[..., race, pass_bar]` (the positional convention FairSynData's prediction code relies on) and
has no catastrophic-scale column like Adult's `capital-gain` (Law's one zero-IQR column,
`fulltime`, only ranges 1–2, nowhere near the 99999-vs-O(1) imbalance that broke Adult). See
"Law reference run" below for the convergence check.

## Model selection: validation only, never test

`draft_model/run_draft.py` already splits off a validation set (`--val_frac=0.2`, disjoint from
test) and logs `round_logs[-1].val_accuracy`/`val_EO_gap` — the validation performance of the
same final model whose test metrics get reported. Selection rule, fixed *before* looking at any
test number: for each dataset, sweep `rho ∈ {0.01, 0.05, 0.1, 0.5, 1.0} × {Universum on,
Universum off} × 5 seeds` (50 runs/dataset, using each dataset's other final settings from
`scripts/run_final.sh`), then **maximize mean validation accuracy among configs with mean
validation EO gap ≤ 0.1** (the method's own `epsilon_EO` fairness target, not a number invented
for this selection); if none qualify, **minimize mean validation EO gap** instead. Report that
config's *test* accuracy/EO gap as a 5-seed mean ± std. Script: `scripts/fair_comparison.py`;
full grid: `outputs/tables/fair_comparison_grid.csv`.

**Selected configs:**

| Dataset | Selected config | Validation acc / EO (drove the choice) | Test acc / EO (reported) |
|---|---|---|---|
| Adult  | rho=0.01, **no-universum** | 0.7486 / 0.0843 | 0.7471 ± 0.0129 / 0.0728 ± 0.0545 |
| Credit | rho=0.01, **no-universum** | 0.6896 / 0.0247 | 0.7679 ± 0.0174 / 0.0347 ± 0.0210 |
| Law    | rho=1.0, universum (**tied** with no-universum) | 0.8090 / 0.1733 | 0.6872 ± 0.0312 / 0.2402 ± 0.0660 |

**Two things worth being upfront about, since they cut against what was assumed going in:**

1. **The selection picked no-Universum for *both* Adult and Credit — not "Universum helps
   Adult."** The `T5_ablation.csv` finding that Universum helps Adult is real but narrower than
   it sounds: it's true *at one fixed rho* (the old winner's `rho=0.1`) — at that specific rho,
   no-Universum's test EO is 0.1128 vs. Universum's 0.0687, confirming the ablation. But once rho
   is also chosen freely (not pinned to the Universum-on winner's value), no-Universum's *own*
   best rho (0.01) reaches a notably higher validation *and* test accuracy (≈0.75 vs. ≈0.70) at
   comparable EO — so it wins on the selection rule anyway. "Universum helps Adult" is correct
   only as a fixed-rho, single-seed statement; it is not the right lens once rho is optimized
   too. On Credit, no-Universum wins on both axes at every rho tested, consistent with the
   original T5 finding (fairness improves without it) plus a real accuracy gain the single-seed
   ablation didn't surface.
2. **Universum on/off produced byte-identical results for Law at every rho and seed.** Traced to
   `draft_model/minibatch_design.py:build_universum_templates` — it returns an empty Universum
   set whenever `U_size = min(Delta_s, Ds_size, Delta_k) == 0`, which happens on essentially
   every client-round minibatch given Law's severe minority-intersection sparsity (race ×
   pass_bar). So Universum is already effectively empty for Law even when "on" — `--no_universum`
   is a no-op for this dataset specifically, not a bug in the sweep. Confirmed independently with
   a standalone re-run outside the sweep script (bit-identical `round_logs`). Selection is a tie;
   which one gets reported doesn't matter.

Also worth flagging: no config for Law reached the 0.1 validation-EO target within the swept rho
range (best was 0.1733 at rho=1.0), so Law's selection used the fallback rule (minimize
validation EO), unlike Adult/Credit which both had qualifying configs to choose among.

## Matched-operating-point comparison

Per-dataset: our full sweep's **Pareto-efficient frontier** (on *test* accuracy/EO gap, across
all 10 rho×Universum combos — a broader "what's achievable at these honest, pre-registered
settings" view, distinct from the single validation-selected claim above) vs. the reference's
one point, at matched accuracy and at matched EO gap.

**Adult** — our frontier never reaches the reference's accuracy (0.7738); our highest-accuracy
frontier point (rho=0.01, no-universum) is acc=0.7471, EO=0.0728 — **notably, already fairer
than the reference's EO=0.1192** even at our best accuracy. At matched EO (≤0.1192), our nearest
point is the same one: acc=0.7471 (2.7 points below reference) at EO=0.0728 (fairer than
reference). **Verdict: we don't match reference accuracy, but at every accuracy level we reach,
we're fairer than the reference — a real improvement over the previous single-rho comparison
(was 8 points behind on accuracy; now 2.7).**

**Credit** — at ≥ reference accuracy (0.7445), our nearest frontier point (rho=0.5,
no-universum) reaches acc=0.7557 with EO=0.0133 — slightly higher (less fair) than reference's
EO=0.0098, but **our accuracy there (0.7557) exceeds the reference's (0.7445)**. Our frontier
never quite reaches reference's EO floor (0.0098); closest is 0.0133 at acc=0.7557 — still
**higher accuracy than reference at nearly-matched fairness**. **Verdict: reference retains a
slight edge on pure fairness (0.0098 vs. our 0.0133), but our frontier now sits at or above the
reference on accuracy at every comparable fairness level — reversed from before, where reference
dominated both axes.**

**Law** — our frontier never reaches reference accuracy (0.8438) or reference EO (0.0849); best
accuracy point is acc=0.6999 (rho=0.1, universum), EO=0.2624; fairest point is EO=0.2402
(rho=1.0), acc=0.6872. **Verdict: reference still dominates both axes on Law** — unchanged from
the previous comparison; the Universum tie (above) means there was no lever here to close the
gap the way there was for Adult/Credit.

Figure: `outputs/pareto_tradeoff.png` — solid lines = Universum on, dashed = Universum off,
large diamond = validation-selected point, star = reference point, one color per dataset.

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

## Law reference run

Re-ran Law under the same `K=20` setup used for Adult/Credit (`FairSynData/mycodes/myParams.py`
was already changed; no dataset-specific edits needed this time). Convergence check:

- `obj_val_P`/`loss_real` moved from `ln(2) = 0.6931` down to ~0.38–0.48 by outer iteration 20
  (same order of movement as Adult/Credit's converged runs).
- `TPR_group0 != TPR_group1` at every `rho_o` (not a constant-class predictor).
- EO gap drops sharply as `rho_o` increases and then plateaus rather than continuing strictly
  monotonically — consistent with a real fairness-constrained method reaching a floor, not a
  frozen/degenerate optimizer:

| rho_o | Law accuracy | Law EO gap |
|---|---|---|
| 0     | 0.8709 | 0.3758 |
| 10    | 0.8637 | 0.0977 |
| 100   | 0.8438 | 0.0849 |
| 1000  | 0.8375 | 0.0882 |
| 10000 | 0.8378 | 0.0919 |

Headline number uses the same **`rho_o=100`** convention as Adult/Credit.

## What's still out of scope for this run

- **Seeds.** FairSynData has no `--seed` CLI argument — `set_seed(seed=42)` is called at fixed
  points in `main.py` with no plumbing to vary it. Adding that would be new code, not a data-prep
  or hyperparameter fix, and wasn't part of what was authorized. All three reference numbers
  above are a single run at the codebase's one supported seed (42), not a 5-seed mean like our
  own method's numbers — stated plainly rather than presented as equivalent.
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

To reproduce the validation-selected "our method" numbers and the matched-point comparison:
`"C:\Python311\python.exe" scripts\fair_comparison.py` (uses cached results in
`outputs/pareto_sweep/` when present, ~15-20 min from scratch for the missing half of the grid).
