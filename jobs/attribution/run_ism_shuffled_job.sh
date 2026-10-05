#!/bin/bash
# ISM on a shuffled-label model (random-model baseline). Env: PROTEIN DATA_MODEL SEED
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 60 ))
python scripts/pipeline/ism_attribution.py \
    --run-dir  "results/synthetic/seedsweep/${PROTEIN}_${DATA_MODEL}/shuffled/seed_${SEED}/linear" \
    --model-dir "results/synthetic/seedsweep/${PROTEIN}_${DATA_MODEL}/shuffled/seed_${SEED}" \
    --out results/synthetic/ism_attribution
