#!/usr/bin/env bash
# Final results: our model on Adult, Credit, and Law — best per-dataset settings, 5 seeds each.
# Run from the repo root:  bash scripts/run_final.sh
# Requires PyTorch + the datasets (CreditData/ for credit; law is bundled in FairSynData/rawdata/).
#
# COMPAS is NOT run here -- its "best settings" are picked via a validation-only sweep, same
# rule as scripts/fair_comparison.py: run `python scripts/compas_sweep.py` once beforehand (needs
# CompasData/compas-scores-two-years.csv, or `pip install responsibly` as a fallback source for
# the same file — see pipeline/load_compas.py). It writes draft_results_compas_final_seed{1..5}.json
# itself; this script's table/figure rebuild step below then picks those up automatically.
set -u
SEEDS="1 2 3 4 5"
COMMON="--num_clients 5 --rounds 8 --K_inner 100 --deterministic true"

echo "===== CREDIT (intercept + threshold ON; imbalanced) ====="
for s in $SEEDS; do
  python -m draft_model.run_draft --data credit --sensitive sex \
    --add_intercept true --tune_threshold true \
    --dp_variant none --rho 0.05 --epsilon_EO 0.1 --seed $s $COMMON \
    --results_file draft_results_credit_final_seed$s.json
done

echo "===== ADULT (intercept OFF — already calibrated; best raw accuracy) ====="
for s in $SEEDS; do
  python -m draft_model.run_draft --data adult --sensitive sex \
    --add_intercept false --tune_threshold false \
    --dp_variant none --rho 0.1 --epsilon_EO 0.1 --seed $s $COMMON \
    --results_file draft_results_adult_final_seed$s.json
done

echo "===== LAW (intercept + threshold ON; label-imbalanced ~89% pass) ====="
for s in $SEEDS; do
  python -m draft_model.run_draft --data law --sensitive race \
    --add_intercept true --tune_threshold true \
    --dp_variant none --rho 0.1 --epsilon_EO 0.1 --seed $s $COMMON \
    --results_file draft_results_law_final_seed$s.json
done

echo "===== COMPAS: run scripts/compas_sweep.py separately first ====="
echo "(picks rho/Universum via validation-only selection -- same rule as credit/adult/law's"
echo " fair_comparison.py -- then writes draft_results_compas_final_seed{1..5}.json itself)."
if ls "$(dirname "$0")/../outputs"/draft_results_compas_final_seed*.json >/dev/null 2>&1; then
  echo "  (compas final files already present -- skipping, run compas_sweep.py again to redo)"
else
  echo "  Run: python scripts/compas_sweep.py"
fi

echo "===== Rebuild tables + figures ====="
python build_tables.py 2>/dev/null || echo "(build_tables.py: check it picks up *_final_* files)"
python plot_results.py
echo "DONE. Results in outputs/, figures in outputs/*.png"
