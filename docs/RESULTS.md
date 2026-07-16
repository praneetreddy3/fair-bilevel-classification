# Results Summary — Credit Risk & UCI Adult (FINAL, post-calibration)

Sweep of 12 configs per dataset (`--add_intercept true --tune_threshold true`, 5 rounds,
K=5 clients) plus 5-seed reruns of the winner. Figures via `python plot_results.py`.

## Winner per dataset (rule: highest accuracy with EO gap ≤ 0.05)

| Dataset | Winner config | Pipeline acc / EO | Baseline acc / EO |
|---|---|---|---|
| Credit | `none, rho0.05, eps0.1` | **0.770 / 0.000** | 0.754 / 0.059 |
| Adult  | `none, rho0.1, eps0.1`  | 0.731 / 0.048 | 0.795 / 0.164 |

## 5-seed confirmation (mean ± std)

| Dataset | Pipeline acc | Pipeline EO gap | Pipeline F1 | Baseline acc / EO |
|---|---|---|---|---|
| Credit | 0.737 ± 0.023 | 0.068 ± 0.013 | 0.452 ± 0.015 | 0.742 / 0.069 |
| Adult  | 0.720 ± 0.029 | 0.083 ± 0.057 | 0.568 ± 0.021 | 0.794 / 0.162 |

## Verdict (calibrated, 5-seed — see docs/PAPER_TABLES.md)

The intercept fix was essential and worked: both baselines are now sane
(**Adult 85.2%**, Credit 75.9%). But once the baselines are properly calibrated, the earlier
dramatic "fairness win" shrinks — because the huge pre-fix EO gaps were themselves artifacts
of the miscalibration. Honest picture:

- **Credit:** pipeline accuracy (0.737) ≈ baseline (0.759); EO gap ≈ baseline (0.068 vs
  0.054). The calibrated baseline is *already fair*, so the method adds little here and is
  slightly worse on DP/EOD (0.125 vs 0.024). The old 0.438→0.021 story was fixing a broken
  baseline, not beating a fair one.
- **Adult:** pipeline reduces the demographic-parity gap (0.174 → 0.108) but not EO
  (~0.085 either way), at a large accuracy cost (0.852 → 0.699). Turning `--tune_threshold`
  off restored the *baseline* to 85%, but the *pipeline* stays ~70% — that gap is
  structural (linear model + fairness + tiny synthetic set), and needs the MLP capacity
  upgrade, not a flag.
- **Ablations:** removing Universum worsens EO (credit 0.00→0.05, adult 0.03→0.16), so the
  **S-balanced Universum is the active fairness mechanism**; zeroing the ρ penalty
  (`fairness_off`) barely changes EO, i.e. the augmented-Lagrangian term adds little on top.

**Known limitation:** `--dirichlet_alpha 0.1` crashes (extreme skew starves a client of
data) — the partitioner needs a minimum-samples-per-client guard.

## Caveats

- **EO gap is noisy across seeds** (Adult std 0.057). The single-seed "winner" EO (0.000 /
  0.048) does not fully hold at 5 seeds (0.068 / 0.083, slightly above 0.05). Report
  mean ± std and consider ≥10 seeds for the final paper numbers.
- Figures: `outputs/results_pareto.png`, `results_bars.png`, `results_convergence.png`.

## Recommended final settings

- **Credit:** `--add_intercept true --tune_threshold true --dp_variant none --rho 0.05 --epsilon_EO 0.1`
  (or `pre_server, dp_sigma 0.25` for the privacy story).
- **Adult:** `--add_intercept true --tune_threshold false --rho 0.1 --epsilon_EO 0.1`
  to keep raw accuracy high; report balanced accuracy + PR-AUC for the imbalanced view.
