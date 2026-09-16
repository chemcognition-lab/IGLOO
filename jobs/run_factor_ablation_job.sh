#!/bin/bash
# Path-2 factor ablation: recompute IG on an EXISTING trained model with each
# attribution "factor" isolated:
#   signed       = <grad, displacement>   (product; existing completeness-exact baseline)
#   gradient     = ||mean_k grad_k||      (model gradient only, no displacement weighting)
#   displacement = ||x1 - x0||            (pLM embedding displacement only, NO model)
#
# Reuses the model already trained by the single-seed noise sweep
# (results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/), so this is
# IG-only (no training).
#
# Required env vars (submit_factor_ablation.sh --export): PROTEIN DATA_MODEL NOISE METHOD

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4

cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
MODEL_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}"
RESULTS_DIR="results/synthetic/factor_ablation/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/${METHOD}"

echo "FACTOR ABLATION: ${PROTEIN}/${DATA_MODEL} noise=${NOISE} method=${METHOD}"

python scripts/run_integrated_gradients.py "${BASE_CONFIG}" \
    integrated_gradients.interpolation_path=linear \
    integrated_gradients.attribution_method="${METHOD}" \
    integrated_gradients.completeness_validation.enabled=false \
    training.noise.label_noise_std="${NOISE}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=false
