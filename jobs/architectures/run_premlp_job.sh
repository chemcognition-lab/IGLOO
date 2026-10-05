#!/bin/bash
# Architecture test: retrain with a per-residue MLP BEFORE the attention pool
# (pooling_method=esm_attention_premlp), so IG can 'see' the learned function.
# Required env: PROTEIN DATA_MODEL NOISE SEED

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4

cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 60 ))

BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
MODEL_DIR="results/synthetic/premlp/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/seed_${SEED}"
RESULTS_DIR="${MODEL_DIR}/linear"

python scripts/pipeline/run_integrated_gradients.py "${BASE_CONFIG}" \
    model.pooling_method=esm_attention_premlp \
    integrated_gradients.interpolation_path=linear \
    training.noise.label_noise_std="${NOISE}" \
    training.noise.label_noise_seed="${SEED}" \
    training.seed="${SEED}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=true
