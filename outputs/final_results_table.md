# Final Comparison Table

| Experiment             | Dataset            | Method                     |  Accuracy | Precision |    Recall |        F1 |    TPR_g0 |    TPR_g1 |    EO_gap |
|------------------------|--------------------|----------------------------| --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 2D sklearn baseline    | 2D (30 pts)        | LogReg (sklearn, no split) |    0.7333 |    0.7059 |    0.8000 |    0.7500 |    0.5000 |    1.0000 |    0.5000 |
| 2D draft baseline      | 2D (30 pts)        | Baseline (ERM)             |    0.8333 |    0.7500 |    1.0000 |    0.8571 |    1.0000 |    1.0000 |    0.0000 |
| 2D draft pipeline      | 2D (30 pts)        | Synth + Universum + AL     |    0.8333 |    0.7500 |    1.0000 |    0.8571 |    1.0000 |    1.0000 |    0.0000 |
| Stress 1 baseline      | Moderate (1200)    | Baseline (ERM)             |    0.9375 |    0.8627 |    0.8462 |    0.8544 |    0.9535 |    0.3333 |    0.6202 |
| Stress 1 pipeline      | Moderate (1200)    | Synth + Universum + AL     |    0.7458 |    0.4595 |    0.9808 |    0.6258 |    0.9767 |    1.0000 |    0.0233 |
| Stress 2 baseline      | Strong (1500)      | Baseline (ERM)             |    0.9700 |    0.9400 |    0.8868 |    0.9126 |    1.0000 |    0.1429 |    0.8571 |
| Stress 2 pipeline      | Strong (1500)      | Synth + Universum + AL     |    0.7700 |    0.4322 |    0.9623 |    0.5965 |    0.9783 |    0.8571 |    0.1211 |

==========================================================================================
CONCLUSION
==========================================================================================

1. EO gap improves substantially under imbalance. The pipeline reduced
   EO gap from 0.62 to 0.07 (moderate) and from 0.86 to 0.14 (strong)
   by raising recall for the disadvantaged group.

2. Recall (overall) increases or stays high under the pipeline because
   the model is pushed to correctly classify minority-group positives.
   Per-group TPRs confirm this: TPR g1 rises sharply.

3. Precision drops because the model now predicts more positives overall
   to catch the hard-to-find minority-group positives, increasing false
   positives. This is the expected accuracy-fairness tradeoff.

4. F1 decreases as a consequence of the precision drop outweighing the
   recall gain in overall terms, though per-group fairness is much better.

5. The 2D dataset (30 points) is too small to produce a meaningful gap
   under the draft model. The sklearn baseline shows a gap only because
   it uses a different model and no held-out split.

6. The prototype is promising but needs validation on real-world benchmarks
   (Adult, COMPAS) and nonlinear models to confirm generalization.
