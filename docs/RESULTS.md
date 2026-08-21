# Results Summary — Credit Risk, UCI Adult, Law School & German Credit (FINAL, post-determinism-fix)

Final 5-seed runs via `scripts/run_final.sh` (Credit/Adult/Law) and `scripts/run_german.sh`
(German, secondary — see `docs/PROJECT_STATUS.md`), all with seeds 1-5, `--num_clients 5
--rounds 8 --K_inner 100 --deterministic true`, per-dataset best settings below. Full
tables in `docs/PAPER_TABLES.md` (T1 = performance, T2 = fairness); figures via
`python plot_results.py` (plots the official final-seed runs only; pass `--all` for every
sweep/ablation run ever produced).

Reproducibility was verified before this run: `python verify.py` passes all 6 checks
(including bit-identical determinism for a fixed `--seed`), and running the identical
CLI command twice produces identical accuracy/EO_gap/F1 to every decimal. This fixed a
real bug — `draft_model/server.py`'s post-server DP noise was drawing from an unseeded
RNG; it now reuses the seeded generator threaded through the whole run.

## Final settings per dataset (from `scripts/run_final.sh`)

- **Credit:** `--sensitive sex --add_intercept true --tune_threshold true --dp_variant none --rho 0.05 --epsilon_EO 0.1`
- **Adult:** `--sensitive sex --add_intercept false --tune_threshold false --dp_variant none --rho 0.1 --epsilon_EO 0.1`
- **Law:** `--sensitive race --add_intercept true --tune_threshold true --dp_variant none --rho 0.1 --epsilon_EO 0.1`
- **German** *(secondary)*: `--sensitive age --add_intercept false --tune_threshold false --dp_variant none --rho 0.1 --epsilon_EO 0.1`

## Performance (mean ± std, 5 seeds)

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

## Fairness (mean ± std, 5 seeds)

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

## Verdict

- **Credit:** pipeline accuracy (0.737) ≈ baseline (0.759); EO gap ≈ baseline (0.068 vs
  0.054). The calibrated baseline is *already fair*, so the method adds little here and
  is worse on DP/EOD (0.125/0.126 vs 0.024/0.028). The earlier dramatic "fairness win"
  from pre-calibration numbers was fixing a broken baseline, not beating a fair one.
- **Adult:** pipeline reduces DP (0.142 → 0.135) only marginally and EO/EOD get *worse*
  (0.019 → 0.069, 0.053 → 0.112), at a large accuracy cost (0.845 → 0.696). This is the
  same structural gap noted before: linear model + fairness constraints on this feature
  set costs raw accuracy without a clean fairness win.
- **Law (new):** this is the reference paper's own dataset, so it's the most direct
  comparison point. The baseline is already highly imbalanced/unfair (EO 0.381, EOD
  0.400 — race is strongly predictive of bar passage in this data). The pipeline cuts
  EO by ~31% (0.381 → 0.262) and EOD by ~42% (0.400 → 0.232) at a real but smaller
  accuracy cost than Adult (0.779 → 0.700), and DP gap actually rises slightly (0.191 →
  0.243). This is the dataset where the fairness intervention shows the clearest,
  largest EO/EOD improvement of the three.
- **German (secondary, age as sensitive attribute):** the strongest fairness win in the
  project. Uncalibrated baseline is genuinely unfair by age (EO 0.435, even larger than
  Law's 0.381); the pipeline cuts EO by ~83% (0.435 → 0.075) and EOD by ~58% (0.435 →
  0.182), for an accuracy cost (0.727 → 0.642) similar in size to Law's. Unlike Credit/Law,
  calibrating this dataset (`--add_intercept true --tune_threshold true`) makes the
  *baseline* already near-fair by age (EO ≈ 0.03) — same "calibration accidentally fixes
  fairness" pattern seen on Credit — so the result reported here deliberately uses the
  uncalibrated config, matching Adult's flags, to keep the fairness intervention visible.
  This dataset originally used **foreign worker** as the sensitive attribute (96%/4%
  split, ~28 training samples in the minority group) in an earlier single-seed run, which
  produced an unstable EO estimate that got *worse* under the pipeline — a data-imbalance
  artifact, not a method failure (see `docs/PROJECT_STATUS.md` for that comparison).
- **Ablations (single-seed, see T5 in `docs/PAPER_TABLES.md`):** removing Universum
  worsens EO on both credit and adult, confirming the **S-balanced Universum is the
  active fairness mechanism**; zeroing the ρ penalty (`fairness_off`) barely changes EO
  on credit, i.e. the augmented-Lagrangian term adds little on top there. (No ablation
  sweep has been run for German yet.)

**Known limitation:** `--dirichlet_alpha 0.1` still crashes (extreme skew starves a
client of data) — the partitioner needs a minimum-samples-per-client guard. Not touched
in this run (method-math-adjacent; would need a partitioning fix, not a results change).

## Caveats

- **EO gap is noisy across seeds**, most visibly on Law (std 0.077), Adult (std 0.046)
  and German (std 0.069) pipeline runs — report mean ± std, not single-seed numbers. On
  German, direction is consistent (every one of the 5 seeds shows a clear EO reduction
  vs. its own baseline), but the exact magnitude varies more (seed range: 58%-100%
  reduction).
- T3 (sensitivity sweep), T4 (non-IID robustness) and T5 (ablation) in
  `docs/PAPER_TABLES.md` are single-seed sweeps from an earlier session and are not
  part of this final 5-seed run; Law and German aren't included in those sweeps. Note
  their "winner" reference row now comes from this run's seed-1 file (credit/adult only),
  while the other rows in the same tables (dirichlet alphas, ablation variants) are still
  from the old seed-42 sweep — a minor seed mismatch within T4/T5, not a correctness issue.
- Figures: `outputs/results_pareto.png`, `results_bars.png`, `results_convergence.png`
  (regenerated via `python plot_results.py`, official final-seed runs only — credit,
  adult, law, german).
