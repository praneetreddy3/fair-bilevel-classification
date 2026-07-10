# Project Brief for the VS Code Coding Agent

> **How to use this file:** Open this repo in VS Code with the Claude Code extension. Either let the agent read this file, or paste the **"Kickoff prompt"** at the bottom into the chat. Tip: rename this file to `CLAUDE.md` at the repo root and Claude Code will load it automatically as project context every session.

---

## 1. What this project is

A **fair, collaborative (federated) binary-classification framework**. Each client keeps its raw data private and instead sends a small **synthetic dataset + "Universum" pseudo-positive points**; the server pools these and trains a global model that is both **accurate** and **fair** in the **Equal-Opportunity** sense (equal true-positive rate across sensitive groups). Differential-privacy (DP) noise can be added to the shared points.

The team is evaluating this on four application domains. **My assignment is Application I: Credit Risk Prediction** (tabular, sensitive attribute = sex, label = loan default). The existing code already works on the UCI Adult dataset; Credit Risk reuses the same machinery with a new data loader.

## 2. The dataset I'm using

- **Default of Credit Card Clients** (UCI id 350) — ~30,000 rows, ~22% default (imbalanced).
- File location: **`CreditData/`** in the repo root (`.xls`, `.xlsx`, or `.csv`). The loader auto-detects the file and the UCI descriptive header row.
- Label `Y = 1` if default, else 0. Sensitive `A = 1` if male, 0 if female.
- (German Credit was skipped for time.)

## 3. Key files (what to touch, what not to)

| File | Role |
|---|---|
| `draft_model/run_draft.py` | **Main entry point.** Loads data, splits clients, runs rounds, evaluates, saves JSON. `--data credit` already wired in. |
| `pipeline/load_credit.py` | **Credit loader** (already scaffolded). Reads `CreditData/`, median-imputes, returns `(X,A,Y)` train/test. |
| `pipeline/load_adult.py` | Reference loader (Adult). |
| `draft_model/bilevel_al.py` | Core bilevel Augmented-Lagrangian solver (inner θ training + outer feature update via implicit CG). |
| `draft_model/minibatch_design.py` | Builds stratified minibatch B, synthetic Dˢ, Universum U. Constants: `D_S_SIZE=32`, `BMIN=32`, `BMAX=128`, `PTARGET=8`. |
| `draft_model/losses.py` | Loss functions. `f_theta` is **linear** (`Xa @ theta`) — the accuracy bottleneck. `g_EO` = smooth TPR-gap. |
| `draft_model/server.py` | Aggregation, global ridge-ERM training, `compute_eo_gap_and_accuracy` (reports accuracy/EO/F1/TPR only). |
| `draft_model/dp.py` | DP variants: `none`, `pre_server`, `post_server`, `both`. |

Do **not** rewrite the method. The task is to run it on credit data, verify, and add the missing evaluation pieces.

## 4. What's already done

- `pipeline/load_credit.py` created and `credit` wired into `run_draft.py` (`--data credit`).
- Guides in the repo: `ALL_IN_ONE_GUIDE.md`, `ENHANCEMENT_GUIDE.md`, `CREDIT_RISK_RUNBOOK.md`. Real Adult results in `FINAL_RESULTS_SUMMARY.md`.

## 5. What I need the agent to do

1. **Confirm the loader works**: run the smoke test, fix any column-name mismatch in `load_credit.py` if the file's headers differ from the UCI default.
2. **Run two configs on credit** (accuracy-track and fairness-track) and save JSONs.
3. **Run the same two configs on Adult** for comparison.
4. **Verify** the results against the checklist below.
5. **Produce clean figures**: an accuracy-vs-EO-gap Pareto scatter and a grouped bar chart, baseline highlighted, saved to `outputs/`.
6. **Then (if time) close the paper-table gaps**: add a **Dirichlet non-IID** client-partition option to `run_draft.py`, and add **PR-AUC, ROC-AUC, macro-F1, balanced accuracy, ΔDP, ΔEOD** to `compute_eo_gap_and_accuracy` (use sklearn). These matter because credit data is imbalanced and accuracy alone is misleading.

## 6. How to run

```bash
# Smoke test
python -m draft_model.run_draft --data credit --sensitive sex \
  --num_clients 3 --rounds 3 --K_inner 30 --results_file draft_results_credit_smoke.json

# Accuracy-track (DP off, weak fairness)
python -m draft_model.run_draft --data credit --sensitive sex --dp_enabled false \
  --rho 0.05 --epsilon_EO 0.25 --rounds 8 --K_inner 100 --num_clients 5 \
  --results_file draft_results_credit_accuracy.json

# Fairness-track (pre-server DP, strong fairness)
python -m draft_model.run_draft --data credit --sensitive sex --dp_enabled true \
  --dp_variant pre_server --dp_sigma 0.25 --rho 0.1 --epsilon_EO 0.05 \
  --rounds 8 --K_inner 100 --num_clients 5 --results_file draft_results_credit.json
```

Same commands with `--data adult` for the Adult comparison.

## 7. How to verify ("check them properly")

Open each `outputs/draft_results_*credit*.json` and confirm:
- Both `baseline` and `pipeline` blocks are populated; `accuracy` ≈ 0.78–0.82; `F1` > 0.
- **Fairness-track** `EO_gap` is clearly **lower** than the baseline `EO_gap`, and `TPR_group0` ≈ `TPR_group1`.
- **Accuracy-track** `accuracy` is **close to baseline** (higher EO gap acceptable).
- No NaNs; run completes without errors across all `rounds`.
- Sanity: the method should trade a little accuracy for a large EO-gap reduction. If EO gap does NOT drop on the fairness-track, something is wrong — inspect `rho`, `epsilon_EO`, and the Universum construction.

## 8. Guardrails

- Change **one variable at a time** so results are attributable.
- Keep the method fixed; only add evaluation/partitioning/loader code.
- Report every final config as **mean ± std over 5 seeds**.
- Save all figures to `outputs/` at 300 dpi with titles, axis labels, legends.

---

## Kickoff prompt (paste this into Claude Code)

```
You are working in my fair-bilevel federated-learning repo. Read VSCODE_AGENT_BRIEF.md
first for full context. My assignment is the Credit Risk application using the
"Default of Credit Card Clients" dataset, which is in the CreditData/ folder.

Do this in order, showing me results at each step:

1. Run the credit smoke test:
   python -m draft_model.run_draft --data credit --sensitive sex --num_clients 3 --rounds 3 --K_inner 30 --results_file draft_results_credit_smoke.json
   If pipeline/load_credit.py errors on column names, inspect the file in CreditData/ and fix the column detection, then re-run.

2. Run the accuracy-track and fairness-track configs for credit (commands in section 6 of the brief), and the same two for --data adult.

3. Verify every output JSON against the checklist in section 7. Tell me clearly whether the fairness-track reduces the EO gap versus baseline and whether TPR is balanced across groups.

4. Create outputs/pareto.png (EO gap vs accuracy, baseline as a star, best = top-left) and outputs/bars.png (accuracy / F1 / EO gap per config), for both credit and adult.

5. If steps 1-4 pass, add: (a) a Dirichlet non-IID client-partition option (alpha in {10,1,0.5,0.1}) to run_draft.py, and (b) PR-AUC, ROC-AUC, macro-F1, balanced accuracy, demographic-parity gap and equalized-odds gap to compute_eo_gap_and_accuracy in draft_model/server.py using sklearn. Re-run the credit fairness-track and show the expanded metrics.

Do not modify the core method (bilevel_al.py, losses.py math). Change one thing at a time.
Report a short results table after each run. Use plan mode and show me the plan before editing files.
```
