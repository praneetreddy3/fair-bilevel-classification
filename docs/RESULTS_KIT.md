# Results kit

Everything needed to drop the fair-bilevel-classification results into the manuscript, pulled
from `outputs/tables/` and `outputs/*.png`. Reproducibility: `python verify.py` (6/6 checks,
including bit-identical determinism for a fixed `--seed`); full setup/config detail in
`docs/RESULTS.md`.

## 1. Final results — our method (mean ± std, 5 seeds)

Final per-dataset settings (from `scripts/run_final.sh`): Credit —
`--sensitive sex --add_intercept true --tune_threshold true --dp_variant none --rho 0.05
--epsilon_EO 0.1`; Adult — `--sensitive sex --add_intercept false --tune_threshold false
--dp_variant none --rho 0.1 --epsilon_EO 0.1`; Law — `--sensitive race --add_intercept true
--tune_threshold true --dp_variant none --rho 0.1 --epsilon_EO 0.1`. All: `--num_clients 5
--rounds 8 --K_inner 100 --deterministic true`, seeds 1–5.

| Dataset | Accuracy | F1 | Balanced acc. | PR-AUC | EO gap | DP gap | EOD gap |
|---|---|---|---|---|---|---|---|
| Credit | 0.7372 ± 0.0229 | 0.4520 ± 0.0153 | 0.6159 ± 0.0095 | 0.4410 ± 0.0141 | 0.0675 ± 0.0128 | 0.1246 ± 0.0352 | 0.1264 ± 0.0369 |
| Adult  | 0.6962 ± 0.0084 | 0.5599 ± 0.0086 | 0.7383 ± 0.0085 | 0.5613 ± 0.0088 | 0.0687 ± 0.0459 | 0.1351 ± 0.0586 | 0.1116 ± 0.0309 |
| Law    | 0.6999 ± 0.0191 | 0.8057 ± 0.0174 | 0.6508 ± 0.0393 | 0.9578 ± 0.0112 | 0.2624 ± 0.0769 | 0.2430 ± 0.0713 | 0.2324 ± 0.0934 |

Source: `outputs/tables/T1_main_performance.csv` + `T2_fairness.csv` (pipeline rows). Uncalibrated
baseline rows (same tables) show what each dataset looks like *without* the fairness
intervention, for context: Credit 0.7594/0.0537, Adult 0.8452/0.0189, Law 0.7786/0.3808
(accuracy/EO gap).

## 2. Our method vs. reference (FairSynData) model

| Dataset | Our method — Accuracy | Our method — EO gap | Reference — Accuracy | Reference — EO gap |
|---|---|---|---|---|
| Adult  | 0.6962 ± 0.0084 | 0.0687 ± 0.0459 | 0.7738 | 0.1192 |
| Credit | 0.7372 ± 0.0229 | 0.0675 ± 0.0128 | 0.7445 | 0.0098 |
| Law    | 0.6999 ± 0.0191 | 0.2624 ± 0.0769 | 0.8438 | 0.0849 |

Reference numbers are a single run each at FairSynData's one hardcoded seed (no `--seed` CLI
exists there), at `rho_o=100` out of its 5-point sweep. Full detail, convergence checks, and the
two data-prep bugs that had to be fixed to get real (non-degenerate) reference numbers are in
`docs/COMPARISON.md`; reproduction steps for the reference setup are in `reference_setup/README.md`.
Source: `outputs/tables/COMPARISON.csv`.

## 3. Figures

- **`outputs/pareto_tradeoff.png`** — accuracy vs. EO gap for our method, swept over `rho ∈
  {0.01, 0.05, 0.1, 0.5, 1.0}` at 5 seeds each (one line per dataset), with each dataset's single
  reference-model point overlaid as a star. Shows where our method's fairness/accuracy curve
  sits relative to the reference at a comparable operating point.
- **`outputs/results_pareto.png`** — accuracy vs. EO gap across every logged single-seed config
  (final runs, sensitivity sweep, ablations), baseline marked with a star per dataset; broader
  but noisier view than the 5-seed Pareto curve above.
- **`outputs/results_bars.png`** — baseline vs. pipeline accuracy and EO gap, side by side, for
  every logged config — quick visual for "did the fairness intervention help or hurt, and by how
  much."

## 4. Summary

Two different comparisons matter here and shouldn't be conflated: our method **vs. its own
uncalibrated baseline** (docs/RESULTS.md), and our method **vs. the external FairSynData
reference model** (this kit, section 2, plus the `rho`-sweep in `pareto_tradeoff.png`). Against
its own baseline, the fairness intervention's clearest win is on **Law** — EO gap drops from
0.381 to 0.262 (a ~31% reduction) at a moderate accuracy cost (0.779 → 0.700) — while on
**Adult** the intervention actually *raises* EO gap relative to baseline (0.019 → 0.069) at a
much larger accuracy cost, and on **Credit** the baseline is already fair, so the intervention
adds little on the primary metric and costs on DP/EOD. Against the *external* reference model,
the picture is narrower than "our method wins": across the full `rho ∈ {0.01, ..., 1.0}` sweep,
our method is fairer than the reference **only on Adult** (best EO ≈0.069 vs. the reference's
0.119, at roughly an 8-point accuracy cost) — on **Credit** and **Law** the reference is both
more accurate *and* substantially fairer than anything our sweep reached (Credit: reference EO
0.010 vs. our best ≈0.067; Law: reference EO 0.085 vs. our best ≈0.240). Ablations (T5, single
seed) show removing the S-balanced Universum term worsens EO gap on Adult (0.064 → 0.157) but
*improves* it on Credit (0.082 → 0.049) — the opposite of what `docs/RESULTS.md`'s text
currently claims for Credit, so Universum is confirmed as the active fairness mechanism on
Adult specifically, not uniformly across datasets (worth a follow-up fix to that doc). All
numbers above are 5-seed reproducible: `verify.py` passes all 6 checks including bit-identical
determinism for a fixed `--seed`, and `scripts/run_final.sh` / `scripts/pareto_sweep.py`
regenerate every config used in this kit from scratch.
