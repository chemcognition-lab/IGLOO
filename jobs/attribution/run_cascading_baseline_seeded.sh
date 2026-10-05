#!/bin/bash
# Multi-seed cascading randomization for the IG baseline (esm_attention,
# pool-first) architecture, GB1 + T4Lysozyme, seeds 1-3, noise=0.0.
#
# CORRECTED from the first version of this script: results/synthetic/seedsweep/
# ALREADY has baseline-architecture checkpoints trained at seeds 1-5 across
# all 9 noise levels (confirmed via config_snapshot: pooling_method=esm_attention,
# label_noise_mode=blend) -- no new training needed, just point
# cascading_randomization.py at the existing seed_1/2/3 checkpoints there.
#
# Output: results/synthetic/cascading_randomization_baseline/
#
# Task list: 6 protein/landscape combos x 3 seeds = 18
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --array=1-18
#SBATCH --output=jobs/logs/rb_cascading_baseline_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

PROTS=(GB1 GB1 GB1 T4Lysozyme T4Lysozyme T4Lysozyme)
LAND=(additive epistatic nonlinear additive epistatic nonlinear)
SEEDS=(1 2 3)

TASKS=()
for i in "${!PROTS[@]}"; do
  for S in "${SEEDS[@]}"; do
    TASKS+=("${PROTS[$i]} ${LAND[$i]} $S")
  done
done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read P D S <<< "$t"
MD="results/synthetic/seedsweep/${P}_${D}/noise_0.0/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] P=$P D=$D seed=$S MD=$MD"

python scripts/sanity_checks/cascading_randomization.py \
  --run-dir "${MD}/linear" --model-dir "$MD" \
  --n-variants 50 --steps 100 \
  --out results/synthetic/cascading_randomization_baseline
