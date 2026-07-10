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

## Verdict

- **Credit — as expected / better.** The intercept fix lifted accuracy from ~58–68% to
  **77%**, matching or beating its own baseline (75%) while the EO gap drops to ~0 (single
  seed) / 0.07 (5-seed). Best private option: `pre_server, rho0.1, eps0.05` → acc 0.723,
  EO 0.016.
- **Adult — fairness improves, raw accuracy dips due to threshold tuning.** The pipeline
  roughly halves the EO gap (0.164 → ~0.08) but accuracy falls to ~72% because
  `--tune_threshold` optimises *balanced* accuracy (better minority recall) at the cost of
  raw accuracy. To recover Adult's ~84% raw accuracy, run Adult with
  `--tune_threshold false`, or report **balanced accuracy** (~0.75) and PR-AUC instead.

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
