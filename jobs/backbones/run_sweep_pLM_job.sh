#!/bin/bash
# Generic SLURM job for one cell of the pLM × interpolation-path sweep.
#
# Required env vars (set by submit_sweep_pLM.sh via --export):
#   PROTEIN        e.g. GB1
#   DATA_MODEL     e.g. additive
#   PLM            e.g. ankh_base
#   INTERP         e.g. linear
#   SEED           e.g. 42  (default: 42)
#
# Noise is fixed at 0.0 (clean data) — this sweep isolates pLM quality,
# not noise robustness.  Run submit_sweep.sh for the noise sweep.
#
# Resource directives are intentionally absent here — submit_sweep_pLM.sh
# passes them via `sbatch --mem=... --time=... --gres=...` on the command line.

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ranamoneim@gmail.com

cd /scratch/ranaab/IGLOO || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

SEED="${SEED:-42}"
BASE_CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"

# Results are grouped by pLM first, then protein/model, then interp path.
MODEL_DIR="results/synthetic/plm_${PLM}/${PROTEIN}_${DATA_MODEL}"
RESULTS_DIR="${MODEL_DIR}/${INTERP}"

echo "=========================================="
echo "PLM SWEEP: ${PROTEIN} / ${DATA_MODEL} / ${PLM} / ${INTERP} / seed=${SEED}"
echo "  config    : ${BASE_CONFIG}"
echo "  model_dir : ${MODEL_DIR}  (plm_${PLM}/${PROTEIN}_${DATA_MODEL})"
echo "  results   : ${RESULTS_DIR}"
echo "=========================================="

# Redirect HuggingFace cache to local node storage (avoids mmap errors on Lustre).
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"

# E1 uses flex_attention which requires Triton for JIT compilation.
# Disable dynamo/inductor so PyTorch falls back to eager SDPA (no Triton needed).
export TORCHDYNAMO_DISABLE=1

# For nearest_neighbor, use more integration steps (≈100 per segment).
EXTRA_ARGS=()
if [[ "${INTERP}" == "nearest_neighbor" ]]; then
    EXTRA_ARGS+=(integrated_gradients.steps=1100)
    EXTRA_ARGS+=("integrated_gradients.completeness_validation.enabled=false")
fi

# Token-level IG is only valid for ESM-2 family; force embedding for all other pLMs.
IG_METHOD="embedding"
if [[ "${PLM}" == esm2* ]]; then
    IG_METHOD="token"
fi

python scripts/pipeline/run_integrated_gradients.py "${BASE_CONFIG}" \
    "model.esm_model_name=${PLM}" \
    "integrated_gradients.interpolation_path=${INTERP}" \
    "integrated_gradients.method=${IG_METHOD}" \
    "training.noise.label_noise_std=0.0" \
    "training.seed=${SEED}" \
    "model_dir=${MODEL_DIR}" \
    "results_dir=${RESULTS_DIR}" \
    force_retrain=true \
    "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
