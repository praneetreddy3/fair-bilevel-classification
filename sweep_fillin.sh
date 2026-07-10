#!/bin/bash
# Fill-in runs: adult no-tune 5-seed, dirichlet non-IID robustness, ablations.
set -e
cd "c:\Users\HP\Desktop\Research project"

COMMON="--num_clients 5 --rounds 8 --K_inner 100 --sensitive sex"

# --- STEP 1: adult no-tune 5-seed reruns ---
for s in 1 2 3 4 5; do
  echo "=== adult notune seed $s ==="
  python -m draft_model.run_draft --data adult $COMMON \
    --add_intercept true --tune_threshold false \
    --dp_enabled false --dp_variant none --rho 0.1 --epsilon_EO 0.1 \
    --seed "$s" --results_file "draft_results_adult_notune_seed${s}.json" > /dev/null
done

# --- STEP 2a: non-IID dirichlet, winner configs ---
for alpha in 1.0 0.5 0.1; do
  echo "=== credit dirichlet alpha=$alpha ==="
  python -m draft_model.run_draft --data credit $COMMON \
    --add_intercept true --tune_threshold true \
    --dp_enabled false --dp_variant none --rho 0.05 --epsilon_EO 0.1 \
    --partition dirichlet --dirichlet_alpha "$alpha" \
    --results_file "draft_results_credit_dirichlet_alpha${alpha}.json" > /dev/null

  echo "=== adult dirichlet alpha=$alpha ==="
  python -m draft_model.run_draft --data adult $COMMON \
    --add_intercept true --tune_threshold false \
    --dp_enabled false --dp_variant none --rho 0.1 --epsilon_EO 0.1 \
    --partition dirichlet --dirichlet_alpha "$alpha" \
    --results_file "draft_results_adult_dirichlet_alpha${alpha}.json" > /dev/null
done

# --- STEP 2b: ablations on each winner ---
echo "=== credit ablation no_universum ==="
python -m draft_model.run_draft --data credit $COMMON \
  --add_intercept true --tune_threshold true \
  --dp_enabled false --dp_variant none --rho 0.05 --epsilon_EO 0.1 \
  --no_universum --results_file draft_results_credit_ablation_nouniversum.json > /dev/null

echo "=== credit ablation fairness_off ==="
python -m draft_model.run_draft --data credit $COMMON \
  --add_intercept true --tune_threshold true \
  --dp_enabled false --dp_variant none --rho 0.05 --epsilon_EO 0.1 \
  --fairness_off --results_file draft_results_credit_ablation_fairnessoff.json > /dev/null

echo "=== credit ablation no_intercept ==="
python -m draft_model.run_draft --data credit $COMMON \
  --add_intercept false --tune_threshold true \
  --dp_enabled false --dp_variant none --rho 0.05 --epsilon_EO 0.1 \
  --results_file draft_results_credit_ablation_nointercept.json > /dev/null

echo "=== adult ablation no_universum ==="
python -m draft_model.run_draft --data adult $COMMON \
  --add_intercept true --tune_threshold false \
  --dp_enabled false --dp_variant none --rho 0.1 --epsilon_EO 0.1 \
  --no_universum --results_file draft_results_adult_ablation_nouniversum.json > /dev/null

echo "=== adult ablation fairness_off ==="
python -m draft_model.run_draft --data adult $COMMON \
  --add_intercept true --tune_threshold false \
  --dp_enabled false --dp_variant none --rho 0.1 --epsilon_EO 0.1 \
  --fairness_off --results_file draft_results_adult_ablation_fairnessoff.json > /dev/null

echo "=== adult ablation no_intercept ==="
python -m draft_model.run_draft --data adult $COMMON \
  --add_intercept false --tune_threshold false \
  --dp_enabled false --dp_variant none --rho 0.1 --epsilon_EO 0.1 \
  --results_file draft_results_adult_ablation_nointercept.json > /dev/null

echo "FILLIN DONE"
