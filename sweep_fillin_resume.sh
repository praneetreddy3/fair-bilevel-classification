#!/bin/bash
# Resume after sweep_fillin.sh crashed on credit dirichlet alpha=0.1 (ZeroDivisionError:
# extreme label-skew starves a client of all data -- a minibatch_design.py edge case,
# not touched per "don't modify core method"). Skip that one, continue the rest.
cd "c:\Users\HP\Desktop\Research project"
COMMON="--num_clients 5 --rounds 8 --K_inner 100 --sensitive sex"

echo "=== adult dirichlet alpha=0.1 (may also fail; not fatal) ==="
python -m draft_model.run_draft --data adult $COMMON \
  --add_intercept true --tune_threshold false \
  --dp_enabled false --dp_variant none --rho 0.1 --epsilon_EO 0.1 \
  --partition dirichlet --dirichlet_alpha 0.1 \
  --results_file draft_results_adult_dirichlet_alpha0.1.json > /dev/null 2>outputs/_adult_alpha0.1_err.log
echo "adult alpha=0.1 exit=$?"

set -e
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

echo "RESUME DONE"
