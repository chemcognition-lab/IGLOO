#!/bin/bash
# One cell of the GB1 additive noise x seed x interp decile-fill sweep.
# Env vars (set by submit_gb1_decile_fill.sh via --export): NOISE, SEED, INTERP
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
# --job-name / --output / --time / --dependency set by the submit wrapper

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"

cd "${IGLOO_ROOT}" || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

SEED="${SEED:?SEED not set}"
NOISE="${NOISE:?NOISE not set}"
INTERP="${INTERP:?INTERP not set}"
PROTEIN="GB1"
DATA_MODEL="additive"

BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
# seed_N kept in the path to match the layout of the recovered noise_{0,0.05,0.1,0.3,0.5,0.8,1.0}
# backup (results_seed0_4.zip), which discover_result_dirs_multiseed() reads as the "legacy" layout.
MODEL_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/seed_${SEED}"
RESULTS_DIR="${MODEL_DIR}/${INTERP}"

echo "=========================================="
echo "CELL: ${PROTEIN}/${DATA_MODEL} interp=${INTERP} noise=${NOISE} seed=${SEED}"
echo "  model_dir : ${MODEL_DIR}"
echo "  results   : ${RESULTS_DIR}"
echo "=========================================="

export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Bump IG batch size to make better use of the L40s (default is 10; GB1 is a
# tiny 56-aa protein so headroom is large) -- requested to speed up wall time.
EXTRA_ARGS=(integrated_gradients.batch_size=32)
if [[ "${INTERP}" == "nearest_neighbor" ]]; then
    EXTRA_ARGS+=(integrated_gradients.steps=1100)
    EXTRA_ARGS+=("integrated_gradients.completeness_validation.enabled=false")
fi

python scripts/run_integrated_gradients.py "${BASE_CONFIG}" \
    integrated_gradients.interpolation_path="${INTERP}" \
    training.noise.label_noise_std="${NOISE}" \
    training.seed="${SEED}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=false \
    "${EXTRA_ARGS[@]}"
