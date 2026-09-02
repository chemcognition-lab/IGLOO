#!/bin/bash
# "Complete random model" negative-control cell for GB1 additive, linear
# interp, REDUCED model (attention pooling + single linear layer). Mirrors
# jobs/run_random_baseline_cell.sh but against GB1_additive_reduced_config.yaml
# and a separate results_reduced_model root, for direct comparison against
# the full model's random baseline.
# Env var: SEED
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --time=1:30:00
# --job-name / --output set by the submitting loop

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"
cd "${IGLOO_ROOT}" || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

SEED="${SEED:?SEED not set}"
PROTEIN="GB1"
DATA_MODEL="additive"
BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_reduced_config.yaml"
MODEL_DIR="results/synthetic_reduced_model/${PROTEIN}_${DATA_MODEL}/random_baseline/seed_${SEED}"
RESULTS_DIR="${MODEL_DIR}/linear"

echo "=========================================="
echo "REDUCED-MODEL RANDOM BASELINE CELL: ${PROTEIN}/${DATA_MODEL} seed=${SEED}"
echo "  model_dir : ${MODEL_DIR}"
echo "  results   : ${RESULTS_DIR}"
echo "=========================================="

export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"

python scripts/run_integrated_gradients.py "${BASE_CONFIG}" \
    integrated_gradients.interpolation_path=linear \
    integrated_gradients.batch_size=32 \
    training.noise.shuffle_labels=true \
    training.noise.label_noise_seed="${SEED}" \
    training.seed="${SEED}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=false
