#!/bin/bash
# Cascading randomization for the AttnPool architecture (esm_perres_attn) --
# the one architecture with NO cascading-randomization coverage at all
# (Pool First/IG, PreMLP, and MeanPool already have full GB1 + T4Lysozyme
# coverage in cascading_randomization_baseline/_premlp/_meanpool/).
#
# Checkpoints already exist for all 3 seeds at noise_0.0
# (results/synthetic/newarch_perres_attn/<protein>_<landscape>/noise_0.0/seed_{1,2,3}/),
# so -- same as the PreMLP/MeanPool multiseed job -- this just runs
# cascading_randomization.py against them, no training needed. Unlike that
# job, ALL 3 seeds are new here (seed_1 was never run for AttnPool).
#
# Task list: 1 arch x 6 protein/landscape combos (GB1 x3, T4Lysozyme x3) x 3 seeds = 18
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --array=1-18
#SBATCH --output=jobs/logs/rb_cascading_attnpool_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

ARCH=newarch_perres_attn
OUT=results/synthetic/cascading_randomization_attnpool
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
MD="results/synthetic/${ARCH}/${P}_${D}/noise_0.0/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] arch=AttnPool P=$P D=$D seed=$S MD=$MD OUT=$OUT"

python scripts/sanity_checks/cascading_randomization.py \
  --run-dir "${MD}/linear" --model-dir "$MD" \
  --n-variants 50 --steps 100 \
  --out "$OUT"
