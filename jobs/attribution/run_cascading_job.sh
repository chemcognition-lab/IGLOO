#!/bin/bash
# Cascading model-parameter randomization for one (protein, data_model, noise).
# Reuses the existing single-seed noise-sweep model.
#
# Required env vars (submit loop --export): PROTEIN DATA_MODEL NOISE

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4

cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

RUN_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/linear"
MODEL_DIR="results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}"

echo "CASCADING RANDOMIZATION: ${PROTEIN}/${DATA_MODEL} noise=${NOISE}"

python scripts/sanity_checks/cascading_randomization.py \
    --run-dir "${RUN_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --n-variants 50 \
    --steps 100 \
    --out results/synthetic/cascading_randomization
