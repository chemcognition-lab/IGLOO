#!/bin/bash
# Mechanism + cascading diagnostics on a pre-MLP-architecture model.
# Env: PROTEIN DATA_MODEL NOISE SEED DIAG   (DIAG = mech | casc)
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4

cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 60 ))

RUN_DIR="results/synthetic/premlp/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/seed_${SEED}/linear"
MODEL_DIR="results/synthetic/premlp/${PROTEIN}_${DATA_MODEL}/noise_${NOISE}/seed_${SEED}"

if [[ "$DIAG" == "mech" ]]; then
    python scripts/attribution_mechanism.py \
        --run-dir "$RUN_DIR" --model-dir "$MODEL_DIR" \
        --n-variants 300 --steps 64 \
        --out results/synthetic/attribution_mechanism_premlp
else
    python scripts/cascading_randomization.py \
        --run-dir "$RUN_DIR" --model-dir "$MODEL_DIR" \
        --n-variants 50 --steps 100 \
        --out results/synthetic/cascading_randomization_premlp
fi
