#!/bin/bash
# In-silico mutagenesis attribution for one (protein, data_model, noise, [seed]).
# Required env vars: PROTEIN DATA_MODEL NOISE   (optional SEED for the seedsweep tree)

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4

cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

# stagger start; the mutant caches are shared across noise levels of a protein/landscape
sleep $(( RANDOM % 60 ))

if [[ -n "${SEED:-}" ]]; then
    RUN_DIR="results/synthetic/seedsweep/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/seed_${SEED}/linear"
    MODEL_DIR="results/synthetic/seedsweep/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/seed_${SEED}"
else
    RUN_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/linear"
    MODEL_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}"
fi

python scripts/pipeline/ism_attribution.py \
    --run-dir  "${RUN_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --out results/synthetic/ism_attribution
