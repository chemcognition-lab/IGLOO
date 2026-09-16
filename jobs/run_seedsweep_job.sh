#!/bin/bash
# One cell of the multi-seed noise sweep: (protein, data_model, noise, seed).
# Interpolation path is fixed to `linear` (the interp axis was shown to be a
# no-op in the earlier single-seed sweep).  Each job trains its own model on a
# fresh label-noise realisation + init, then runs IG once.
#
# Required env vars (set by submit_seedsweep.sh via --export):
#   PROTEIN     e.g. GB1
#   DATA_MODEL  e.g. additive
#   NOISE       e.g. 0.3
#   SEED        e.g. 3
#
# Resource directives (--mem/--time/--gres/--output) are passed on the sbatch
# command line by submit_seedsweep.sh.

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4

cd /project/aip-bensl/ranaab/IGLOO || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
MODEL_DIR="results/synthetic/seedsweep/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/seed_${SEED}"
RESULTS_DIR="${MODEL_DIR}/linear"

echo "=========================================="
echo "SEED-SWEEP JOB: ${PROTEIN} / ${DATA_MODEL} / noise=${NOISE} / seed=${SEED}"
echo "  config    : ${BASE_CONFIG}"
echo "  results   : ${RESULTS_DIR}"
echo "=========================================="

python scripts/run_integrated_gradients.py "${BASE_CONFIG}" \
    ig.interpolation_path=linear \
    training.noise.label_noise_std="${NOISE}" \
    training.noise.label_noise_seed="${SEED}" \
    training.seed="${SEED}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=true
