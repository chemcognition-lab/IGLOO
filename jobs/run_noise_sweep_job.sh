#!/bin/bash
# Option-B rebuild: single-seed noise-sweep cell (train + IG), float32.
# Env: PROTEIN DATA_MODEL NOISE
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 45 ))

BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
MODEL_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}"
RESULTS_DIR="${MODEL_DIR}/linear"

python scripts/run_integrated_gradients.py "${BASE_CONFIG}" \
    integrated_gradients.interpolation_path=linear \
    training.noise.label_noise_std="${NOISE}" \
    training.noise.label_noise_seed=0 \
    training.seed=42 \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=true
