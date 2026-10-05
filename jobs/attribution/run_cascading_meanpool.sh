#!/bin/bash
# Cascading randomization for the per-residue-mean-pool head (MeanPool), GB1
# and T4Lysozyme, seed_1 -- same recipe/params as
# results/synthetic/cascading_randomization_premlp/ (see
# jobs/attribution/run_attr_array_v2.sh's "pdiagC" case), just pointed at the
# newarch_perres_mean checkpoints instead of premlp.
#
# Task list: 2 proteins x 3 landscapes = 6
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --array=1-6
#SBATCH --output=jobs/logs/rb_cascading_meanpool_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

PROTS=(GB1 GB1 GB1 T4Lysozyme T4Lysozyme T4Lysozyme)
LAND=(additive epistatic nonlinear additive epistatic nonlinear)

P="${PROTS[$((SLURM_ARRAY_TASK_ID - 1))]}"
D="${LAND[$((SLURM_ARRAY_TASK_ID - 1))]}"
MD="results/synthetic/newarch_perres_mean/${P}_${D}/noise_0.0/seed_1"
echo ">>> [$SLURM_ARRAY_TASK_ID] P=$P D=$D MD=$MD"

python scripts/sanity_checks/cascading_randomization.py \
  --run-dir "${MD}/linear" --model-dir "$MD" \
  --n-variants 50 --steps 100 \
  --out results/synthetic/cascading_randomization_meanpool
