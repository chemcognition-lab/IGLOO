#!/bin/bash
# Extend cascading randomization to seeds 2 and 3 for PreMLP and MeanPool
# (seed_1 already done in cascading_randomization_premlp/ and
# cascading_randomization_meanpool/) -- checkpoints for seeds 2/3 already
# exist (results/synthetic/premlp/ and results/synthetic/newarch_perres_mean/),
# so this just runs the same cascading_randomization.py recipe against them.
# No multi-seed baseline (IG) variant exists since only one model was ever
# trained per noise level there (no seed_* subfolders).
#
# Task list: 2 archs x 6 protein/landscape combos x 2 seeds (2,3) = 24
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --array=1-24
#SBATCH --output=jobs/logs/rb_cascading_multiseed_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

ARCHS=(premlp newarch_perres_mean)
declare -A OUTDIR=([premlp]=results/synthetic/cascading_randomization_premlp
                   [newarch_perres_mean]=results/synthetic/cascading_randomization_meanpool)
PROTS=(GB1 GB1 GB1 T4Lysozyme T4Lysozyme T4Lysozyme)
LAND=(additive epistatic nonlinear additive epistatic nonlinear)
SEEDS=(2 3)

TASKS=()
for A in "${ARCHS[@]}"; do
  for i in "${!PROTS[@]}"; do
    for S in "${SEEDS[@]}"; do
      TASKS+=("$A ${PROTS[$i]} ${LAND[$i]} $S")
    done
  done
done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read A P D S <<< "$t"
MD="results/synthetic/${A}/${P}_${D}/noise_0.0/seed_${S}"
OUT="${OUTDIR[$A]}"
echo ">>> [$SLURM_ARRAY_TASK_ID] arch=$A P=$P D=$D seed=$S MD=$MD OUT=$OUT"

python scripts/sanity_checks/cascading_randomization.py \
  --run-dir "${MD}/linear" --model-dir "$MD" \
  --n-variants 50 --steps 100 \
  --out "$OUT"
