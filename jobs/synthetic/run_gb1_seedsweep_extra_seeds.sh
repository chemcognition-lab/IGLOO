#!/bin/bash
# Extend GB1's Pool First (IG) seedsweep from 5 to 10 seeds, so the
# ensemble-size (K) variance-reduction figure
# (scripts/figures/make_poolfirst_ensemble_boxplot_figure.py) can be extended to
# K=6..10 with genuinely new independent seeds, not resampled from the
# existing 5.
#
# Only trains at each landscape's already-established best-noise operating
# point (all three GB1 landscapes: noise=0.0, confirmed from the existing
# 5-seed seedsweep data) rather than the full 9-level noise sweep, since
# that's the only noise level the downstream figure actually uses -- keeps
# this addition small or and proportionate.
#
# Task list: 3 landscapes x 5 new seeds (6-10) = 15
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-15
#SBATCH --output=jobs/logs/gb1_seedsweep_extra_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

LAND=(additive epistatic nonlinear)
NOISE=0.0
SEEDS=(6 7 8 9 10)

TASKS=()
for D in "${LAND[@]}"; do
  for S in "${SEEDS[@]}"; do
    TASKS+=("$D $S")
  done
done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read D S <<< "$t"
CFG="configs/synthetic/GB1_${D}_config.yaml"
MD="results/synthetic/seedsweep/GB1_${D}/noise_${NOISE}/seed_${S}"

echo ">>> [$SLURM_ARRAY_TASK_ID] GB1 D=$D noise=$NOISE S=$S MD=$MD"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$NOISE" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
