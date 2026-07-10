# Model Enhancement Guide — Fair Bilevel Pipeline

**Purpose of this document.** Use it as a self-contained base/spec for improving the *accuracy* of the existing fair bilevel federated pipeline. It is written so a lighter assistant (in a separate, cheaper chat) can apply the changes and run experiments without re-deriving project context. Every change below names the **exact file, constant/flag, current value, and what to try**.

**Scope.** This is about the existing model (currently validated on the Adult dataset, and reusable for Credit Risk). It is the "earlier application." The goal is **higher accuracy while keeping the Equal-Opportunity (EO) gap near baseline** — not accuracy at any cost.

---

## 1. The logical move (read this first)

The current results show the core tension:

| Model | Accuracy | EO gap | F1 |
|---|---|---|---|
| Reference baseline (plain ERM) | **84.63%** | 1.19% | 0.6548 |
| Tuned, no DP | 69.65% | 6.74% | 0.5694 |
| Tuned DP (`pre_server`, σ=0.25) | 68.89% | **1.31%** | 0.5654 |
| Tuned DP (`pre_server`, σ=0.5) | 70.51% | 19.64% | 0.5403 |

The pipeline is ~14 points below the baseline on accuracy. **Do not chase a single "best" number.** The correct move is to **run two tracks and report a trade-off (Pareto) frontier**:

1. **Accuracy-focused track** — push accuracy as close to baseline as possible (turn DP off or low, relax fairness, give the model more capacity and more synthetic data).
2. **Fairness-focused track** — keep EO near baseline (`pre_server`, σ≈0.25, stronger fairness).

Then plot accuracy vs EO for every config. A fairness paper wins by showing *"we match baseline EO while recovering most of the accuracy,"* not by a single cell. The enhancements below are the levers that move points up-and-left on that plot (higher accuracy, lower EO gap).

---

## 2. Why accuracy is being lost (diagnosis)

Four root causes, each with a corresponding lever:

1. **The classifier is linear.** In `draft_model/losses.py`, `f_theta(theta, Xa) = (Xa @ theta)` — a single linear layer on `[x; a]`. Credit/Adult data are not linearly separable, so this caps accuracy. **Biggest lever.**
2. **The server trains on tiny synthetic data.** Each client sends only `D_S_SIZE = 32` synthetic points (`draft_model/minibatch_design.py`). With the default 3 clients that is ~96 points total to fit the global model. More synthetic data → better fit.
3. **DP noise degrades accuracy.** `dp_sigma` Gaussian noise is added to features. Default run uses `dp_variant=post_server`, `dp_sigma=1.0` — strong noise.
4. **The fairness constraint pulls the boundary.** Large `rho` and small `epsilon_EO` force EO down at the cost of accuracy.

---

## 3. Enhancement levers (ranked by impact-to-effort)

### Tier A — Zero-code changes (just CLI flags / constants). Do these first.

All flags are on `python -m draft_model.run_draft`. Defaults are in `run_draft.py`.

| Lever | Flag / constant | Current default | Try | Why / expected effect |
|---|---|---|---|---|
| Turn DP off for the accuracy track | `--dp_enabled false` (or `--dp_variant none`) | `dp_enabled=true`, `dp_variant=post_server`, `dp_sigma=1.0` | `none` first, then `pre_server` σ∈{0.1,0.25} | Removes the noise that costs the most accuracy. `post_server` σ=1.0 is the worst case — avoid for accuracy. |
| Relax fairness tolerance | `--epsilon_EO` | `0.25` | sweep `0.05, 0.10, 0.25` | Larger ε lets the solver stop sooner with a less aggressive boundary → higher accuracy, higher EO. Use to draw the trade-off curve. |
| Lower the fairness penalty | `--rho` | `0.5` | `0.05, 0.1, 0.5` | Smaller ρ = weaker fairness push = higher accuracy. Pair with the ε sweep. |
| More communication rounds | `--rounds` | `3` | `5, 10` | Each round the global model warm-starts the next (`zeta = theta_glob`); more rounds = better convergence. Keep ≥5 when comparing settings. |
| More inner steps | `--K_inner` | `50` | `100` | Better-trained inner θ each round → cleaner hypergradient. |
| More clients = more synthetic total | `--num_clients` | `3` | `5, 10` | More clients → more pooled synthetic points at the server → better global fit (also matches the paper's K∈{5,10,20}). |
| Stopping criterion | `--stop_criterion` | `eo_gap` | keep `eo_gap` for fairness track; `grad_inf` for convergence studies | `eo_gap` stops when fairness target met; `grad_inf` stops on feature-gradient flatness. |

**Two ready-to-run reference configs:**

```bash
# Accuracy-focused
python -m draft_model.run_draft --data adult --dp_enabled false \
  --rho 0.05 --epsilon_EO 0.25 --rounds 8 --K_inner 100 --num_clients 5 \
  --results_file draft_results_accuracy.json

# Fairness-focused
python -m draft_model.run_draft --data adult --dp_enabled true --dp_variant pre_server \
  --dp_sigma 0.25 --rho 0.1 --epsilon_EO 0.05 --rounds 8 --K_inner 100 --num_clients 5 \
  --results_file draft_results_fairness.json
```

### Tier B — Small, safe code changes.

1. **Increase synthetic size.** In `draft_model/minibatch_design.py`, `D_S_SIZE = 32`. Raise to `64` or `128`. Also `BMAX = 128` (max real minibatch) can rise to `256` so synthetic templates have richer statistics. Expected: higher accuracy, slightly slower.
   - Note: `build_synthetic_templates` clamps `Ds_size = min(Ds_size, B_size)`, so also raise minibatch size (`PTARGET = 8` → `12`, or `BMIN = 32` → `64`) for the larger synthetic set to take effect.
2. **Threshold tuning.** In `draft_model/server.py`, `compute_eo_gap_and_accuracy` predicts with `pred = (logits > 0)` — a fixed 0 threshold. On imbalanced credit data this is rarely optimal. Add a tunable threshold (sweep on the validation set to maximize F1 or balanced accuracy). Cheap, often +1–3 points F1 with no fairness cost.
3. **Server training budget.** In `train_global_ridge_erm` (called from `run_draft.py`), `max_iter=500, lr=0.05`. Try `max_iter=1000` or `lr=0.03` if the global loss hasn't converged.
4. **Universum strength.** `lambda_U=0.5` (set in `run_draft.py` call to `client_round_al`). Sweep `0.25, 0.5, 1.0`; and test the `--no_universum` ablation to confirm Universum's contribution.

### Tier C — Bigger change, biggest payoff: give the model capacity.

Replace the linear classifier with a **small MLP** (1 hidden layer, 16–32 units, ReLU). This is the single change most likely to close the accuracy gap, because the bilevel solver (`bilevel_al.py`) already uses generic autograd + conjugate-gradient for the Hessian — it does **not** assume linearity in the optimization, only in `f_theta`.

What to change:
- `draft_model/losses.py` → `f_theta` and `logistic_loss_per_sample`: route logits through a small MLP instead of `Xa @ theta`. `theta` becomes a small parameter bundle (e.g. `W1,b1,W2,b2`) instead of one vector.
- `draft_model/bilevel_al.py` and `server.py`: anywhere `theta` is treated as a flat vector (`Xa @ theta`, `(theta - zeta)`), update to the new parameterization. Keep the regularizer as an L2 over all params.
- Keep it **small** to save time/credits — a 16-unit hidden layer is enough to test the hypothesis. Validate on Adult first (target: recover accuracy toward 80%+ while EO stays ≤ ~baseline).

**Logical caution:** add capacity to the *accuracy* track first and confirm the gain, before combining with strong DP + fairness. Change one thing at a time so each result is attributable.

---

## 4. Lighter / cheaper experiment mode (save credits and time)

To iterate fast and cheap in a separate chat:

- Use the **small dataset and small configs** while developing: `--data 2d` or German Credit (1k rows), `--num_clients 3 --rounds 3 --K_inner 30`. Confirm the pipeline runs end-to-end, then scale up only for final numbers.
- Use the **simplified path** for smoke tests: `--no_full_al` runs inner-only optimization (no expensive Hessian/CG), good for quickly checking data loading and plumbing.
- Keep the **MLP hidden size tiny** (16 units) during prototyping.
- Run **one seed** while exploring; only do the full **5-seed mean±std** for the configs you report.

---

## 5. Show results visually and neatly (required)

Every run writes a JSON (`outputs/draft_results_*.json`) containing `baseline`, `pipeline`, and `round_logs` (per-round `val_accuracy`, `val_EO_gap`, `val_F1`). Build three figures from these:

1. **Accuracy–Fairness Pareto scatter** — x = EO gap, y = accuracy; one point per config (no-DP, pre σ0.25, pre σ0.5, etc.); mark the baseline as a star. The story-telling plot: best configs sit top-left.
2. **Grouped bar chart** — accuracy / F1 / EO gap for baseline vs each tuned config, side by side.
3. **Convergence curves** — `val_accuracy` and `val_EO_gap` vs round, one line per config, from `round_logs`.

The repo already has `visualize_results.py` and `build_figures.py` — extend them, or add a small script that loads all `outputs/draft_results_*.json` and emits the three PNGs at 300 dpi into `outputs/`. Keep styling clean: titles, axis labels, legends, consistent colors, baseline highlighted. Put the final accuracy/EO/F1 numbers in one tidy `RESULTS.md` table alongside the figures.

**Reporting checklist:** every reported config = mean ± std over 5 seeds; always show the baseline on the same plot; one sentence per figure stating the takeaway.

---

## 6. Suggested order of work

1. Tier A flag sweeps → generate the first Pareto plot (fastest signal, no code).
2. Tier B threshold tuning + bigger synthetic → cheap accuracy gains.
3. Tier C small MLP on the accuracy track → confirm the big gain.
4. Re-combine the winning capacity + fairness/DP settings → final 5-seed runs.
5. Regenerate the three figures + `RESULTS.md`.

> Note for full paper tables: the current evaluator reports accuracy, EO gap, F1, and per-group TPR only. The professor's tables also need **PR-AUC, ROC-AUC, macro-F1, balanced accuracy, ΔDP, ΔEOD**. Adding these metrics to `compute_eo_gap_and_accuracy` (sklearn) is a small, high-value task — do it before the final reporting pass.
