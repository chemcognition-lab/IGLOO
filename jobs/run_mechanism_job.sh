#!/bin/bash
# Attribution-mechanism verification for one (protein, data_model, noise).
# Required env vars: PROTEIN DATA_MODEL NOISE

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4

cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

# stagger start to avoid a thundering herd on the shared embedding cache
sleep $(( RANDOM % 90 ))

python scripts/attribution_mechanism.py \
    --run-dir  "results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/linear" \
    --model-dir "results/synthetic/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}" \
    --n-variants 300 \
    --steps 64 \
    --out results/synthetic/attribution_mechanism
