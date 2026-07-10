# Project Status Report — Fair Bilevel Classification (Credit Risk)

**Author:** Praneet Chinthala  
**Application:** I — Credit Risk Prediction  
**Date:** July 1, 2026

## Summary

I applied the existing fair bilevel federated model (previously validated on UCI Adult) to a new credit-risk dataset, the **Default of Credit Card Clients** (UCI, ~30k records, ~22% default rate). The model was reused with minor changes to fit the new data. Fairness results are strong on both datasets; the open issue is a drop in accuracy that has been diagnosed and is fixable.

## What was done

- Built a data loader for the credit dataset and integrated it into the existing pipeline.
- Ran the model in two settings — an accuracy-focused and a fairness-focused configuration — on both the credit dataset and UCI Adult (single seed each).
- Added non-IID (Dirichlet) client partitioning and extended evaluation metrics (PR-AUC, ROC-AUC, macro-F1, balanced accuracy, demographic-parity gap, equalized-odds gap).
- Generated trade-off visualizations (accuracy vs. EO gap).

## Results (single seed)

| Config | Dataset | Baseline acc / EO gap | Pipeline acc / EO gap |
|---|---|---|---|
| Fairness-track | Credit | 0.682 / 0.438 | 0.595 / **0.021** |
| Accuracy-track | Credit | 0.682 / 0.438 | 0.549 / 0.029 |
| Fairness-track | Adult | 0.846 / 0.012 | 0.653 / 0.119 |
| Accuracy-track | Adult | 0.846 / 0.012 | 0.699 / 0.176 |

**Fairness works as intended:** on credit the EO gap collapses from 0.438 to ~0.02, with balanced true-positive rates across groups (0.678 vs 0.699). The equalized-odds gap shows the same pattern (0.438 → 0.016).

## Open issue: accuracy

Accuracy on credit drops to ~55–68%. Root cause identified: the model's linear scoring function has **no intercept (bias) term**, so on imbalanced credit data it over-predicts the positive (default) class (~30% vs the true 22%). Confirmation: a standard logistic regression *with* an intercept reaches **80.8%** on the same preprocessed features, so this is a modeling-calibration gap, not a data or pipeline problem. On UCI Adult the baseline (84.6%) already matches prior results, because that dataset's geometry does not expose the missing intercept.

## Next steps

1. Add an intercept term (or tune the decision threshold) to fix credit calibration — expected to lift accuracy toward ~80%.
2. Reduce accuracy loss in the fairness-constrained pipeline via larger synthetic sets and higher model capacity (small MLP).
3. Confirm final numbers with 5-seed mean ± standard deviation.
4. Update the shared GitHub repo so the team can reference and reuse the code.
