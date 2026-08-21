#!/usr/bin/env bash
# German Credit fix — our model with a balanced sensitive attribute (age), 5 seeds.
# Secondary/standalone: deliberately not folded into run_final.sh, which is scoped to
# the final 3-dataset pipeline (Credit/Adult/Law) that build_tables.py/docs/RESULTS.md
# depend on. Run from the repo root:  bash scripts/run_german.sh
# Requires the German Credit data in GermanData/, or internet for the ucimlrepo fallback.
set -u
SEEDS="1 2 3 4 5"
COMMON="--num_clients 5 --rounds 8 --K_inner 100 --deterministic true"

echo "===== GERMAN CREDIT (age sensitive attr, balanced ~52/48; intercept+threshold OFF) ====="
for s in $SEEDS; do
  python -m draft_model.run_draft --data german --sensitive age \
    --add_intercept false --tune_threshold false \
    --dp_variant none --rho 0.1 --epsilon_EO 0.1 --seed $s $COMMON \
    --results_file draft_results_german_final_seed$s.json
done

echo "DONE. Results in outputs/draft_results_german_final_seed{1..5}.json"
