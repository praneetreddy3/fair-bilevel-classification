# Final Adult Results Summary

Rerun timestamp: Apr 29, 2026  
Source of truth: `outputs/final_model_suite_results.json`

## Model versions

- **Reference baseline**
  - Standard ERM baseline in `run_draft.py` (`baseline` block in each JSON).
  - No synthetic pipeline DP stage is applied to this baseline branch.

- **Tuned model (without DP)**
  - `dp_variant=none`, `rho=0.1`, `epsilon_EO=0.05`, `stop_criterion=eo_gap`.

- **Tuned model (with DP, noise captured)**
  - Fairness-focused: `dp_variant=pre_server`, `dp_sigma=0.25`, `rho=0.05`, `epsilon_EO=0.05`.
  - Accuracy-focused: `dp_variant=pre_server`, `dp_sigma=0.5`, `rho=0.1`, `epsilon_EO=0.05`.

## Final rerun metrics (Adult)

- **Reference baseline**: accuracy `84.63%`, EO gap `1.19%`, F1 `0.6548`
- **Tuned no-DP** (`none`): accuracy `69.65%`, EO gap `6.74%`, F1 `0.5694`
- **Tuned DP (pre, sigma=0.25)**: accuracy `68.89%`, EO gap `1.31%`, F1 `0.5654`
- **Tuned DP (pre, sigma=0.5)**: accuracy `70.51%`, EO gap `19.64%`, F1 `0.5403`
- **DP post-server (sigma=1.0)**: accuracy `67.67%`, EO gap `7.11%`, F1 `0.5604`
- **DP both stages (sigma=1.0)**: accuracy `68.40%`, EO gap `21.27%`, F1 `0.4885`
- **FairSynData-style stop (grad_inf, no-DP)**: accuracy `67.23%`, EO gap `15.47%`, F1 `0.5334`

## Conclusion

- Baseline reference remains strongest on accuracy.
- Best tuned accuracy is `70.51%` (pre-server DP, sigma `0.5`) but fairness degrades.
- Best tuned fairness is near baseline EO at sigma `0.25`, with lower accuracy.
- Stronger DP noise/coverage (post or both with sigma `1.0`) is not beneficial in this setup.
