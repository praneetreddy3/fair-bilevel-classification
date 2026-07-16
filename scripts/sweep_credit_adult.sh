#!/bin/bash
# Constant-value sweep on credit and adult: dp{none, pre_server@0.25} x rho{0.05,0.1} x eps{0.05,0.1,0.25}
# Fixed: --num_clients 5 --rounds 8 --K_inner 100 --add_intercept true --tune_threshold true --sensitive sex
set -e
cd "c:\Users\HP\Desktop\Research project"

for data in credit adult; do
  for dp in none pre_server; do
    if [ "$dp" = "none" ]; then
      dp_flags="--dp_enabled false --dp_variant none"
    else
      dp_flags="--dp_enabled true --dp_variant pre_server --dp_sigma 0.25"
    fi
    for rho in 0.05 0.1; do
      for eps in 0.05 0.1 0.25; do
        fname="draft_results_${data}_${dp}_rho${rho}_eps${eps}.json"
        echo "=== $fname ==="
        python -m draft_model.run_draft --data "$data" --sensitive sex \
          --num_clients 5 --rounds 8 --K_inner 100 \
          --add_intercept true --tune_threshold true \
          $dp_flags --rho "$rho" --epsilon_EO "$eps" \
          --results_file "$fname" > /dev/null
      done
    done
  done
done
echo "SWEEP DONE"
