#!/bin/bash
# Shuffled-label baseline: model trained on permuted fitness -> ~0 test
# performance -> attribution "random-model" reference (Shen-style red star).
#
# Required env vars (set by submit_shuffle.sh via --export):
#   PROTEIN, DATA_MODEL, SEED

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4

cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
MODEL_DIR="results/synthetic/seedsweep/${PROTEIN}_${DATA_MODEL}/shuffled/seed_${SEED}"
RESULTS_DIR="${MODEL_DIR}/linear"

echo "SHUFFLE-LABEL JOB: ${PROTEIN} / ${DATA_MODEL} / seed=${SEED}"

python scripts/run_integrated_gradients.py "${BASE_CONFIG}" \
    ig.interpolation_path=linear \
    training.noise.label_noise_std=0.0 \
    training.shuffle_labels=true \
    training.shuffle_labels_seed="${SEED}" \
    training.seed="${SEED}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=true
