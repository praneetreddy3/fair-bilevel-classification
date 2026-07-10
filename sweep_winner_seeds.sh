#!/bin/bash
# 5-seed reruns of the sweep winners
set -e
cd "c:\Users\HP\Desktop\Research project"

for seed in 1 2 3 4 5; do
  echo "=== credit winner seed $seed ==="
  python -m draft_model.run_draft --data credit --sensitive sex \
    --num_clients 5 --rounds 8 --K_inner 100 \
    --add_intercept true --tune_threshold true \
    --dp_enabled false --dp_variant none --rho 0.05 --epsilon_EO 0.1 \
    --seed "$seed" \
    --results_file "draft_results_credit_winner_seed${seed}.json" > /dev/null

  echo "=== adult winner seed $seed ==="
  python -m draft_model.run_draft --data adult --sensitive sex \
    --num_clients 5 --rounds 8 --K_inner 100 \
    --add_intercept true --tune_threshold true \
    --dp_enabled false --dp_variant none --rho 0.1 --epsilon_EO 0.1 \
    --seed "$seed" \
    --results_file "draft_results_adult_winner_seed${seed}.json" > /dev/null
done
echo "WINNER SEEDS DONE"
