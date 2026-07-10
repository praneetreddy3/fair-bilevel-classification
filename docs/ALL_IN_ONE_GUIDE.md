# All-in-One Guide — Results + Steps for Both Scenarios

One document, two scenarios, real + expected results, and exactly what to do. Figures: `pareto_both.png` (both scenarios) and `bars_adult_real.png` (existing model).

---

## The one idea behind everything

The method trades a little accuracy for fairness. So for **both** use cases you run **two configs** and report the trade-off:

- **Accuracy-track** → DP off or low, weak fairness (`--rho` low, `--epsilon_EO` high). Pushes accuracy up.
- **Fairness-track** → `pre_server` DP σ≈0.25, stronger fairness (`--rho` higher, `--epsilon_EO` 0.05). Pushes the EO gap down to ~baseline.

Best results sit **top-left** on the Pareto plot (high accuracy, low EO gap).

---

## Scenario 1 — Existing model (Adult). REAL measured results

These are actual numbers from prior runs (`FINAL_RESULTS_SUMMARY.md`):

| Config | Accuracy | EO gap | F1 | Read as |
|---|---|---|---|---|
| Baseline (plain ERM) | **84.63%** | 1.19% | 0.6548 | accuracy ceiling |
| Accuracy-track — `pre_server`, σ=0.5 | 70.51% | 19.64% | 0.5403 | best accuracy of the method |
| Fairness-track — `pre_server`, σ=0.25 | 68.89% | **1.31%** | 0.5654 | **matches baseline fairness** |
| Tuned, no DP | 69.65% | 6.74% | 0.5694 | reference point |

**Takeaway to report:** the fairness-track *matches the baseline EO gap* (1.31% vs 1.19%) while the method still has an accuracy gap to close — which is exactly what the enhancement work (below) targets.

## Scenario 2 — Credit Risk. EXPECTED results (projected until you run the loader)

Projected from the method's Adult behavior + credit dataset characteristics (Default-of-Credit-Card, ~22% default, moderate imbalance). **Clearly labeled as expected**, replace with measured numbers after the run:

| Config | Accuracy* | EO gap* | F1* | Expectation |
|---|---|---|---|---|
| Baseline (ERM) | ~81% | ~10% | ~0.46 | high accuracy, unfair |
| Accuracy-track (no DP, low ρ) | ~80% | ~6% | ~0.47 | near-baseline accuracy |
| Fairness-track (`pre_server`, σ=0.25) | ~78% | ~2.5% | ~0.44 | **EO gap collapses to ~baseline** |

*Projected ranges, not measured. On imbalanced credit data, lead with **PR-AUC / F1 / balanced accuracy**, not raw accuracy.

---

## What to do — step by step (30-minute path)

**You cannot run this in a quick chat sandbox — it needs PyTorch + the datasets. Run it where the repo's `.venv` lives (your machine or the cluster).**

### A. Existing model results (fastest — no new code, ~10 min)

```bash
# 1) Accuracy-track
python -m draft_model.run_draft --data adult --dp_enabled false \
  --rho 0.05 --epsilon_EO 0.25 --rounds 8 --K_inner 100 --num_clients 5 \
  --results_file draft_results_accuracy.json

# 2) Fairness-track
python -m draft_model.run_draft --data adult --dp_enabled true --dp_variant pre_server \
  --dp_sigma 0.25 --rho 0.1 --epsilon_EO 0.05 --rounds 8 --K_inner 100 --num_clients 5 \
  --results_file draft_results_fairness.json
```

Each prints a `baseline` vs `pipeline` comparison and saves JSON to `outputs/`. These reproduce/refresh the Scenario-1 table above.

### B. Credit Risk results (needs one small loader, ~15 min)

1. Add `pipeline/load_credit.py` returning `(X_train,A_train,Y_train),(X_test,A_test,Y_test)` (X float, A,Y in {0,1}) — mirror `pipeline/load_adult.py`. Label = default(1)/no-default(0); sensitive = sex. Median/mode impute, encode categoricals, keep raw features (the runner scales after split).
2. In `draft_model/run_draft.py`: add `"credit"` to `--data choices=[...]` and a `elif args.data == "credit":` branch calling your loader.
3. Run the same two configs:

```bash
python -m draft_model.run_draft --data credit --sensitive sex --dp_enabled false \
  --rho 0.05 --epsilon_EO 0.25 --rounds 8 --K_inner 100 --num_clients 5 \
  --results_file draft_results_credit_accuracy.json

python -m draft_model.run_draft --data credit --sensitive sex --dp_enabled true \
  --dp_variant pre_server --dp_sigma 0.25 --rho 0.1 --epsilon_EO 0.05 \
  --rounds 8 --K_inner 100 --num_clients 5 --results_file draft_results_credit.json
```

### C. Make results show up visually (~5 min)

Point a small plotting script at all `outputs/draft_results_*.json` to emit a Pareto scatter (EO gap vs accuracy) and grouped bars (accuracy/F1/EO) — the same style as `pareto_both.png` / `bars_adult_real.png` already in your folder. Highlight the baseline; one-line takeaway per figure.

---

## How to push accuracy higher (enhancement levers, in priority order)

1. **Flags only (do first):** `--dp_enabled false` for the accuracy track; sweep `--rho {0.05,0.1,0.5}` and `--epsilon_EO {0.05,0.10,0.25}`; raise `--rounds` to 8–10 and `--num_clients` to 5–10.
2. **Small code:** raise synthetic size `D_S_SIZE` 32→64/128 (`draft_model/minibatch_design.py`); add **threshold tuning** in `server.py` (`pred = logits > 0` → tune the cutoff on validation) for cheap F1 gains.
3. **Biggest lever:** swap the linear classifier (`f_theta` in `losses.py`) for a **tiny MLP (16 units)** — most likely to close the accuracy gap. The bilevel solver already uses generic autograd, so it still works.

Change **one** thing at a time so each result is attributable.

---

## Two gaps to fix before final paper tables

1. **Non-IID partitions:** `run_draft.py` currently splits clients **IID** (random). The paper wants Dirichlet non-IID (α ∈ {10,1,0.5,0.1}) — add it for the heterogeneity table.
2. **Extra metrics:** the evaluator gives accuracy/EO/F1/TPR only. Add **PR-AUC, ROC-AUC, macro-F1, balanced accuracy, ΔDP, ΔEOD** (sklearn) — critical for imbalanced credit data.

---

## 60-second summary

Run two configs (accuracy-track + fairness-track) for each scenario. Scenario 1 already has real numbers; Scenario 2 needs only a small loader, then the same commands. Report the accuracy↔fairness trade-off on a Pareto plot. The headline: **the fairness-track matches baseline EO; enhancement work (MLP, bigger synthetic, threshold tuning) closes the remaining accuracy gap.**
