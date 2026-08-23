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
(accuracy/EO gap). **Note:** section 2 below uses a *different* set of per-dataset configs than
this table — a validation-selected operating point (rho, Universum on/off) chosen specifically
for a fair comparison against the reference model, not the `run_final.sh` configs above. See
`docs/COMPARISON.md` for why the two differ.

## 2. Our method vs. reference (FairSynData) model

| Dataset | Our method — Accuracy | Our method — EO gap | Reference — Accuracy | Reference — EO gap |
|---|---|---|---|---|
| Adult  | 0.7471 ± 0.0129 | 0.0728 ± 0.0545 | 0.7738 | 0.1192 |
| Credit | 0.7679 ± 0.0174 | 0.0347 ± 0.0210 | 0.7445 | 0.0098 |
| Law    | 0.6872 ± 0.0312 | 0.2402 ± 0.0660 | 0.8438 | 0.0849 |

"Our method" here is each dataset's **validation-selected** operating point (never chosen by
looking at test numbers) out of a 10-way `rho × Universum on/off` sweep — see
`docs/COMPARISON.md`'s "Model selection" section for the exact rule and two honest surprises it
turned up (no-Universum wins the selection for *both* Adult and Credit, not just Credit as the
single-rho ablation suggested; Universum is a no-op for Law specifically). Reference numbers are
a single run each at FairSynData's one hardcoded seed (no `--seed` CLI exists there), at
`rho_o=100` out of its 5-point sweep. Full detail, convergence checks, and the two data-prep bugs
fixed to get real (non-degenerate) reference numbers are in `docs/COMPARISON.md`; reproduction
steps for the reference setup are in `reference_setup/README.md`. Source:
`outputs/tables/COMPARISON.csv` (headline) / `outputs/tables/fair_comparison_grid.csv` (full grid).

## 3. Figures

- **`outputs/pareto_tradeoff.png`** — accuracy vs. EO gap for our method, swept over `rho ∈
  {0.01, 0.05, 0.1, 0.5, 1.0} × Universum {on, off}` at 5 seeds each (solid = Universum on,
  dashed = off, one color per dataset), the validation-selected point marked with a large
  diamond, and each dataset's reference-model point overlaid as a star. Shows where our method's
  achievable frontier sits relative to the reference, and which point we'd actually report.
- **`outputs/results_pareto.png`** — accuracy vs. EO gap across every logged single-seed config
  (final runs, sensitivity sweep, ablations), baseline marked with a star per dataset; broader
  but noisier view, uses the `run_final.sh` configs rather than the validation-selected ones.
- **`outputs/results_bars.png`** — baseline vs. pipeline accuracy and EO gap, side by side, for
  every logged config — quick visual for "did the fairness intervention help or hurt, and by how
  much."

## 4. Summary

Two different comparisons matter here and shouldn't be conflated: our method **vs. its own
uncalibrated baseline** (`docs/RESULTS.md`, using the `run_final.sh` configs), and our method
**vs. the external FairSynData reference model** (this kit's section 2, using
validation-selected configs — a materially better result than the `run_final.sh` configs give,
picked without ever looking at test numbers). Against its own baseline, the fairness
intervention's clearest *relative* EO reduction is on **Law** (0.381 → 0.262, a ~31% cut, at a
moderate accuracy cost) — that's a within-method comparison, not a claim about beating the
reference (the reference is far fairer than us on Law either way, see below). Against the
*external* reference at validation-selected operating points: on **Adult**, our frontier never
matches the reference's accuracy (0.7738), but at our best-accuracy point (0.7471) we're already
fairer (EO 0.0728 vs. reference 0.1192) — a real improvement over the fixed-rho comparison (was
8 points behind on accuracy; now 2.7). On **Credit**, our frontier now sits at or above the
reference on accuracy at every comparable fairness level (e.g. acc=0.7557 at EO=0.0133 vs.
reference's acc=0.7445 at EO=0.0098) — reference keeps a slight fairness edge, but no longer
dominates both axes as it did before. On **Law**, the reference still dominates both axes
outright — Universum turned out to be a no-op for Law (empty by construction given its data
sparsity, not a bug), so there was no lever here the way there was for Adult/Credit. Ablations
(`T5_ablation.csv`, single seed, fixed rho) show removing Universum worsens EO on Adult
(0.064 → 0.157) but *improves* it on Credit (0.082 → 0.049) — the opposite of what
`docs/RESULTS.md` previously claimed for Credit (now corrected there); note that once rho is
also optimized (this kit's sweep), no-Universum actually wins the validation selection on Adult
too, so "Universum helps Adult" only holds at a fixed rho, not as a general statement. All
numbers above are 5-seed reproducible: `verify.py` passes all 6 checks including bit-identical
determinism for a fixed `--seed`, and `scripts/run_final.sh` / `scripts/fair_comparison.py`
regenerate every config used in this kit from scratch.
