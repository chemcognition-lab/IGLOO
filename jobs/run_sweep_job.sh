#!/bin/bash
# Generic SLURM job for one cell of the interpolation-path × noise sweep.
#
# Required env vars (set by submit_sweep.sh via --export):
#   PROTEIN        e.g. GB1
#   DATA_MODEL     e.g. additive
#   INTERP         e.g. slerp
#   NOISE          e.g. 0.1
#
# Resource directives are intentionally absent here — submit_sweep.sh passes
# them via `sbatch --mem=... --time=... --gres=...` on the command line.

#SBATCH --account=rrg-mahadeva
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ranamoneim@gmail.com
# --output is set by submit_sweep.sh on the sbatch command line

cd /home/ranaab/projects/rrg-mahadeva/ranaab/IGLOO || exit 1


module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

# Derived paths
BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
# Model is shared across interpolation paths for the same (protein, data_model, noise)
MODEL_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}"
RESULTS_DIR="${MODEL_DIR}/${INTERP}"

echo "=========================================="
echo "SWEEP JOB: ${PROTEIN} / ${DATA_MODEL} / ${INTERP} / noise=${NOISE}"
echo "  config    : ${BASE_CONFIG}"
echo "  model_dir : ${MODEL_DIR}"
echo "  results   : ${RESULTS_DIR}"
echo "=========================================="

# Redirect HuggingFace cache to local node storage to avoid mmap bus errors
# on network filesystems (Lustre/GPFS on /scratch does not support mmap).
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"

python scripts/run_integrated_gradients.py "${BASE_CONFIG}" \
    integrated_gradients.interpolation_path="${INTERP}" \
    training.noise.label_noise_std="${NOISE}" \
    model_dir="${MODEL_DIR}" \
    results_dir="${RESULTS_DIR}" \
    force_retrain=true
