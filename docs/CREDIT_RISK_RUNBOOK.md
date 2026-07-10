# Credit Risk — Run Instructions (Application I)

**Purpose.** Step-by-step instructions to run the existing fair bilevel federated pipeline on the **Credit Risk** application. Because the pipeline is already a tabular fair classifier, the only real new code is a **dataset loader**; everything else (bilevel + Universum + DP + server aggregation) runs unchanged.

---

## 1. Which datasets to target

| Dataset | Size | Sensitive attributes | Imbalance | Use it for |
|---|---|---|---|---|
| **German Credit** (UCI id 144) | ~1,000 rows | sex/age, marital, foreign-worker | mild (~30% bad) | **Start here** — tiny, runs in seconds, perfect for wiring up and debugging. |
| **Default of Credit Card Clients** (UCI id 350) | ~30,000 rows | sex, age, marriage, education | moderate (~22% default) | **Main results dataset** — large enough to be credible, still fast. |
| **LendingClub** (Kaggle) | very large | derived (state, employment) | strong | **Scale-up / stress test** only, once the pipeline is solid. Optional. |

**Binary label:** `y = 1` if default / bad loan, `0` otherwise.
**Sensitive attribute (`A`):** use **sex** as the primary (binary, well-populated), **age group** as a secondary. Encode as `{0,1}` exactly like `pipeline/load_adult.py` does (`Female/Male`, etc.).

Recommendation: **German Credit to build it, Default-of-Credit-Card for the reported numbers.** LendingClub is optional and only worth it for a communication-efficiency / scale argument.

---

## 2. Add the loader (the only required code)

The pipeline loads data through small functions in `pipeline/` (see `load_adult.py`). Add a parallel `pipeline/load_credit.py` that returns the same shape:

```python
def prepare_credit_for_draft(sensitive="sex", dataset="default_credit"):
    # ...load + preprocess...
    return (X_train, A_train, Y_train), (X_test, A_test, Y_test)
# X: (n, d) float; A: (n,) in {0,1}; Y: (n,) in {0,1}
```

Then wire it into `draft_model/run_draft.py`:
1. Add `"credit"` to `--data` choices (line with `choices=["dummy", "adult", "2d"]`).
2. Add a branch `elif args.data == "credit": ... = load_credit_data(...)` mirroring the `adult` branch.

**Preprocessing rules (follow the project's existing answers in `project questions.txt`):**
- Numerical variables → **median** imputation; categorical → **mode** imputation; add a missing-indicator flag if missingness is meaningful.
- **Encode categoricals first** (one-hot/ordinal), then **scale only continuous columns**.
- **Split first, scale second** — fit `StandardScaler` on train only, transform val/test. (`run_draft.py` already applies scaling this way after the split — keep raw features in the loader and let the runner scale.)

---

## 3. Run it — baseline first, then the pipeline

```bash
# Smoke test on the tiny dataset
python -m draft_model.run_draft --data credit --sensitive sex \
  --num_clients 3 --rounds 3 --K_inner 30 --results_file draft_results_credit_smoke.json

# Main run (Default of Credit Card Clients)
python -m draft_model.run_draft --data credit --sensitive sex \
  --num_clients 5 --rounds 8 --K_inner 100 --dp_enabled true --dp_variant pre_server \
  --dp_sigma 0.25 --rho 0.1 --epsilon_EO 0.05 --results_file draft_results_credit.json
```

Each run prints and saves a JSON with a `baseline` (plain ERM on the real data) and the `pipeline` (the fair method) so you always have the head-to-head comparison.

---

## 4. Scenarios and what each one changes (and why)

Run these as controlled sweeps — change **one** axis at a time so every result is attributable.

**a) Differential-privacy placement (`--dp_variant`, `--dp_sigma`).**
- `none` → no privacy noise; highest accuracy, weakest privacy. Use as the accuracy ceiling for the method.
- `pre_server` → noise on the client payload before it leaves the device. **Preferred** here: it matches the federated privacy intent (protect what is transmitted) and empirically gave the best fairness/accuracy balance (σ≈0.25). Start here.
- `post_server` → noise after aggregation. Avoid high σ — in prior runs σ=1.0 degraded both accuracy and EO.
- `both` → both stages; strongest privacy, usually worst utility. Ablation only.
- `--dp_sigma` sweep `{0.1, 0.25, 0.5, 1.0}`: higher σ = more privacy, more accuracy loss, and unstable EO at the high end.

**b) Fairness strength (`--rho`, `--epsilon_EO`).**
- `--rho` (penalty weight): higher = fairness pushed harder, accuracy drops. Sweep `{0.05, 0.1, 0.5}`.
- `--epsilon_EO` (EO tolerance to stop): smaller = stricter fairness, lower accuracy. Sweep `{0.05, 0.10, 0.25}`.
- Together these trace the **accuracy–fairness trade-off curve** — the key deliverable.

**c) Federation size (`--num_clients`, `--rounds`).**
- More clients = data split thinner per client, but more total synthetic points reach the server. The paper expects K ∈ {5, 10, 20} — report at least 5 and 10.
- More rounds = better convergence (the global model warm-starts each round). Keep ≥5 when comparing settings.

**d) Sensitive attribute (`--sensitive sex` vs `race`/age).** Fairness behavior differs by attribute because group sizes and base rates differ; report sex as primary, age as secondary.

**e) Stopping criterion (`--stop_criterion eo_gap` vs `grad_inf`).** `eo_gap` stops once the fairness target is met (use for fairness runs); `grad_inf` stops on feature-gradient flatness (use for convergence/ablation studies).

**f) Universum on/off (`--no_universum`).** Ablation: confirms the Universum pseudo-positives are actually helping minority-group TPR. Expect fairness to worsen when removed.

---

## 5. Why these methods are preferred here

- **`pre_server` DP** aligns with the federated privacy story (protect the transmitted payload, not just the server's output) and was empirically the most stable accuracy↔fairness setting. `post_server` at high σ over-perturbs already-mixed synthetic data.
- **Equal Opportunity (TPR-gap) objective** is the right fairness notion for credit: default is the minority/positive class, and the harm of interest is denying good outcomes to a disadvantaged group — i.e., unequal *true-positive* rates. The pipeline equalizes EO by **raising** the lagging group's TPR (via balanced Universum pseudo-positives), not by degrading the other group.
- **Importance weighting + rolling positive buffer** (on by default) keep minority-class positives represented despite class imbalance — important because credit defaults are the rarer class.
- **Median/mode imputation, scale-after-split** avoid leakage and stay reproducible across federated clients.

---

## 6. Metrics and reporting

The current evaluator (`server.py: compute_eo_gap_and_accuracy`) reports **accuracy, EO gap, F1, per-group TPR**. The professor's tables also require **PR-AUC, ROC-AUC, macro-F1, balanced accuracy, ΔDP, ΔEOD** — add these (sklearn) to the evaluator before the final reporting pass. Because credit data is imbalanced, lead with **PR-AUC, F1, balanced accuracy**, not raw accuracy.

Report every config as **mean ± std over 5 seeds**, with a **Wilcoxon signed-rank test** vs the strongest baseline (p < 0.05). Always show the plain-ERM baseline alongside the pipeline.

---

## 7. Known gaps to close (so results match the paper tables)

1. **Non-IID partitioning.** `run_draft.py` currently splits clients with a random shuffle = roughly **IID**. The paper wants IID **and** Dirichlet label-skew non-IID (α ∈ {10, 1, 0.5, 0.1}). Add a Dirichlet partition option to populate the heterogeneity table (Table 2).
2. **Extra metrics** (PR-AUC, ROC-AUC, macro-F1, balanced acc, ΔDP, ΔEOD) — see §6.
3. **Standard baselines** the table expects (centralized LogReg/SVM/RF/XGBoost/LightGBM/MLP; federated FedAvg/FedProx/SCAFFOLD; fairness Reweighing/Adversarial/FairFed). The current "baseline" is a single plain ERM — add the rest for the comparison tables.

---

## 8. Quick checklist

1. Download German Credit + Default-of-Credit-Card.
2. Write `pipeline/load_credit.py`; add `credit` to `run_draft.py`.
3. Smoke-test on German Credit (3 clients, 3 rounds).
4. Main run on Default-of-Credit-Card with `pre_server` σ=0.25.
5. Sweep DP, ρ, ε_EO, clients → build the trade-off plot.
6. Add missing metrics + non-IID partition.
7. 5-seed final runs + Wilcoxon + figures.
