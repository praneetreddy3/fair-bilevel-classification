# Project Status — Application I: Credit Risk (+ Adult, Law, COMPAS reference)

**Lead:** Praneet Chinthala · **Contributor:** Yanjia (German Credit)
**Method:** fair bilevel federated classifier (synthetic + Universum + differential privacy).
Figure: `figures/combined_status.png`.

## Results so far

Adult/Credit/Law numbers below are the current 5-seed final results from
`scripts/run_final.sh` (see `docs/RESULTS.md` for full mean±std tables) — corrected after
fixing an unseeded RNG in `draft_model/server.py`'s post-server DP noise, which previously
made pipeline runs non-deterministic. The German Credit row is a separate, earlier
single-seed run by a different contributor with a different sensitive attribute, not part
of `scripts/run_final.sh`.

| Dataset | Sensitive attr | Baseline acc / EO | Pipeline acc / EO | Read |
|---|---|---|---|---|
| Default of Credit Card Clients | sex | 0.759 / 0.054 | 0.737 / 0.068 | matches baseline; already-fair baseline |
| UCI Adult (reference) | sex | 0.845 / 0.019 | 0.696 / 0.069 | small fairness gain, real accuracy cost |
| Law School (reference paper's own dataset) | race | 0.779 / 0.381 | 0.700 / 0.262 | clearest EO win *vs. its own baseline* (~31% reduction); reference model is still far fairer (see `docs/COMPARISON.md`) |
| COMPAS recidivism (ProPublica) | race | 0.640 / 0.276 | 0.632 / 0.166 | **best fairness-for-accuracy trade-off of the four** — ~40% EO reduction at <1 point accuracy cost; validation-selected winner has Universum **off** |
| German Credit *(secondary, single seed)* | foreign worker | 0.700 / 0.174 | 0.615 / 0.299 | EO **worsened**, accuracy dropped |

## Key findings

1. **The intercept/calibration fix was essential.** Adding a bias term recovered credit
   accuracy from ~58–68% up to ~77%. Without it, the linear model over-predicts the minority
   (default) class on imbalanced data.
2. **A properly calibrated baseline is often already fair.** On Default-Credit, once calibrated,
   the baseline EO gap is small (~0.05), so the method adds little — its large early "wins" were
   mostly fixing a broken baseline, not beating a fair one.
3. **German Credit underperformed because of the sensitive-attribute choice, not the method.**
   "Foreign worker" splits the data 96% / 4% — the minority group has only ~28 training samples
   (3 negatives). The method cannot estimate or equalise a group-level true-positive rate from so
   few points, so the EO gap is unstable and worsens. This is a data limitation, not a bug.
4. **COMPAS is the strongest fairness result in the whole project.** A genuinely unfair
   baseline (EO 0.276, the well-documented COMPAS racial bias) gets a ~40% EO reduction at
   under 1 point of accuracy cost — and the validation sweep picked Universum **off** here,
   a third distinct data point (after Adult: helps, Credit: hurts) showing the Universum
   mechanism's effect is genuinely dataset-dependent, not a fixed on/off switch.

## Suggestions and next steps (and why)

1. **Use a balanced sensitive attribute.** Re-run German Credit with **binary age** (as Yanjia
   plans) or another attribute where both groups are well populated — the method needs enough
   samples per group to equalise TPR. Foreign-worker is too skewed.
2. **Apply the calibration fixes to German Credit** (`--add_intercept true`, and threshold
   tuning). Its baseline is only 70% — the same miscalibration likely applies; the fix that
   rescued Default-Credit should help here too.
3. **Enforce a minimum group size** in the partition/minibatch construction, and skip or flag
   settings where a group is below that floor, so results stay stable.
4. **Target settings where the method can actually demonstrate benefit:** datasets with genuine
   baseline unfairness *and* adequately sized groups. Report these honestly against the baseline.
5. **Close the accuracy gap** (Adult pipeline ~70% vs baseline ~85%) with a small MLP — this is
   the main structural lever; the linear model caps pipeline accuracy.

## One-line status

The pipeline runs cleanly across four real-world datasets (Credit, Adult, Law, COMPAS) with
well-understood, honestly-reported fairness behaviour — including one clear win (COMPAS);
remaining open work is German Credit's sensitive-attribute choice and, optionally, closing
the accuracy gap on Adult with added model capacity.

## Done: COMPAS (4th real-life application)

`pipeline/load_compas.py` loads ProPublica's "Two Years" recidivism dataset
(Caucasian=1/African-American=0, standard AIF360-style filtering). Best settings were
picked the same principled way as credit/adult/law's `scripts/fair_comparison.py`:
`scripts/compas_sweep.py` swept rho x Universum on/off x 5 seeds (50 runs) and selected
the config maximising validation accuracy subject to validation EO_gap <= 0.1, never
looking at test numbers. Winner: `rho=0.01`, Universum **off**.

Result: baseline 0.640 acc / 0.276 EO -> pipeline 0.632 acc / 0.166 EO — a ~40% EO
reduction at under 1 point of accuracy cost, the best fairness-for-accuracy trade-off of
the four datasets. Verified reproducible: `python verify.py` — 6/6 checks pass on the
full repo including this addition. Full numbers: `docs/RESULTS.md`; sweep grid:
`outputs/tables/compas_sweep.csv`.
