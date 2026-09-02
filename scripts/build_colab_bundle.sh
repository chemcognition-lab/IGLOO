#!/bin/bash
# Assemble the self-contained Colab bundle (colab_standalone/) and zip it.
#
# Only the notebook, its wrapper module and the README are tracked in git; this
# script regenerates everything else that makes the folder standalone:
#   - a vendored copy of src/igloo and the two scripts the notebook imports
#   - the synthetic dataset + ground-truth annotations
#   - two pre-computed runs (best model, shuffled-label random baseline)
#
# Those runs must exist first. If they do not:
#   sbatch jobs/run_sweep_cell_gb1_deciles.sh   # NOISE=0.0 SEED=1 INTERP=linear
#   SEED=0 sbatch jobs/run_random_baseline_cell.sh
#
# Usage:  bash scripts/build_colab_bundle.sh [--zip]

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DEST="${ROOT}/colab_standalone"

BEST_CELL="results/synthetic/GB1_additive/noise_0.0/seed_1"
RANDOM_CELL="results/synthetic/GB1_additive/random_baseline/seed_0"

for cell in "${BEST_CELL}" "${RANDOM_CELL}"; do
    if [[ ! -d "${ROOT}/${cell}/linear" ]]; then
        echo "ERROR: missing ${cell}/linear -- generate it first (see header)." >&2
        exit 1
    fi
done

echo "Vendoring package and scripts ..."
mkdir -p "${DEST}/src" "${DEST}/scripts"
rm -rf "${DEST}/src/igloo"
cp -r "${ROOT}/src/igloo" "${DEST}/src/igloo"
find "${DEST}/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cp "${ROOT}/scripts/run_integrated_gradients.py" \
   "${ROOT}/scripts/plot_ndcg_vs_model_performance.py" "${DEST}/scripts/"

echo "Copying data ..."
mkdir -p "${DEST}/data/synthetic" "${DEST}/data/embeddings"
cp "${ROOT}/data/synthetic/GB1_additive.csv" \
   "${ROOT}/data/synthetic/GB1_ground_truth_weights.json" "${DEST}/data/synthetic/"
for f in GB1_additive_signal_families_vs_model_perf_raw_linear.csv \
         GB1_additive_random_baseline_linear.csv; do
    src="${ROOT}/results/synthetic/figures/${f}"
    [[ -f "${src}" ]] && cp "${src}" "${DEST}/data/synthetic/" \
        || echo "  note: ${f} not found; regenerate with scripts/plot_ndcg_vs_model_performance.py"
done

echo "Copying pre-computed runs ..."
for cell in "${BEST_CELL}" "${RANDOM_CELL}"; do
    mkdir -p "${DEST}/${cell}/linear"
    cp "${ROOT}/${cell}"/*.pth "${DEST}/${cell}/" 2>/dev/null || true
    cp "${ROOT}/${cell}/linear"/*_integrated_gradients_results.csv \
       "${ROOT}/${cell}/linear"/*_config_snapshot_*.yaml "${DEST}/${cell}/linear/" 2>/dev/null || true
done

echo "Bundle assembled at ${DEST} ($(du -sh "${DEST}" | cut -f1))"

if [[ "${1:-}" == "--zip" ]]; then
    rm -f "${ROOT}/colab_standalone.zip"
    (cd "${ROOT}" && zip -qr colab_standalone.zip colab_standalone \
        -x '*__pycache__*' '*.ipynb_checkpoints*')
    echo "Zipped: ${ROOT}/colab_standalone.zip ($(du -sh "${ROOT}/colab_standalone.zip" | cut -f1))"
fi
