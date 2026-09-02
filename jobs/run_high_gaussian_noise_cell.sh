#!/bin/bash
# Large-Gaussian-noise negative-control cell for GB1 additive, linear interp.
# Unlike the shuffle-based random baseline (training.noise.shuffle_labels=true,
# an exact by-construction zero-relationship control), this uses additive
# Gaussian label noise (training.noise.label_noise_std) pushed far past the
# highest value in the original 0.0-1.0 decile sweep, to see how close a
# large-but-finite noise_std can get to spearman~0 through the SAME
# methodology the original noise sweep used.
# Env vars: SEED (default 1), NOISE_STD (default 20.0)
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --time=1:30:00
#SBATCH --job-name=gb1_high_gauss_noise
#SBATCH --output=jobs/logs/slurm-%j.out

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"
cd "${IGLOO_ROOT}" || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

SEED="${SEED:-1}"
NOISE_STD="${NOISE_STD:-20.0}"
PROTEIN="GB1"
DATA_MODEL="additive"
BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
MODEL_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/gaussian_noise_std_${NOISE_STD}/seed_${SEED}"
RESULTS_DIR="${MODEL_DIR}/linear"

echo "=========================================="
echo "HIGH GAUSSIAN NOISE CELL: ${PROTEIN}/${DATA_MODEL} seed=${SEED} noise_std=${NOISE_STD}"
echo "  model_dir : ${MODEL_DIR}"
echo "  results   : ${RESULTS_DIR}"
echo "=========================================="

export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"

python scripts/run_integrated_gradients.py "${BASE_CONFIG}" \
    integrated_gradients.interpolation_path=linear \
    integrated_gradients.batch_size=32 \
    training.noise.label_noise_std="${NOISE_STD}" \
    training.noise.label_noise_seed="${SEED}" \
    training.seed="${SEED}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=false
