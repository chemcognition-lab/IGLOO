#!/bin/bash
# Generic SLURM job for one cell of the interpolation-path × noise sweep.
#
# Required env vars (set by submit_sweep.sh via --export):
#   PROTEIN        e.g. GB1
#   DATA_MODEL     e.g. additive
#   INTERP         e.g. slerp
#   NOISE          e.g. 0.1
#   SEED           e.g. 42  (default: 42)
#
# Resource directives are intentionally absent here — submit_sweep.sh passes
# them via `sbatch --mem=... --time=... --gres=...` on the command line.

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ranamoneim@gmail.com
# --output is set by submit_sweep.sh on the sbatch command line

cd /scratch/ranaab/IGLOO || exit 1


module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

# Derived paths
SEED="${SEED:-42}"  # default to 42 if not set
BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
# Model is shared across interpolation paths for the same (protein, data_model, noise, seed)
MODEL_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}"
RESULTS_DIR="${MODEL_DIR}/${INTERP}"

echo "=========================================="
echo "SWEEP JOB: ${PROTEIN} / ${DATA_MODEL} / ${INTERP} / noise=${NOISE} / seed=${SEED}"
echo "  config    : ${BASE_CONFIG}"
echo "  model_dir : ${MODEL_DIR}"
echo "  results   : ${RESULTS_DIR}"
echo "=========================================="

# Redirect HuggingFace cache to local node storage to avoid mmap bus errors
# on network filesystems (Lustre/GPFS on /scratch does not support mmap).
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"

# For nearest_neighbor, use more integration steps so each segment gets
# ~100 sub-steps (1100 steps / 11 segments) instead of the default ~45.
EXTRA_ARGS=()
if [[ "${INTERP}" == "nearest_neighbor" ]]; then
    EXTRA_ARGS+=(integrated_gradients.steps=1100)
    # Skip the completeness pre-screen (iterates all seqs at 25 steps each)
    # which dominates wall time for large datasets like p53.
    EXTRA_ARGS+=("integrated_gradients.completeness_validation.enabled=false")
fi

python scripts/pipeline/run_integrated_gradients.py "${BASE_CONFIG}" \
    integrated_gradients.interpolation_path="${INTERP}" \
    training.noise.label_noise_std="${NOISE}" \
    training.seed="${SEED}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=false \
    "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
