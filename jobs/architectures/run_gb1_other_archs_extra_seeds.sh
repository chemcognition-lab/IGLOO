#!/bin/bash
# Extend GB1's PreMLP/MeanPool/AttnPool seedsweeps from 3 to 10 seeds each
# (seeds 4-10 new), matching Pool First's extension to 10 seeds
# (run_gb1_seedsweep_extra_seeds.sh) so the ensemble-size (K) variance
# figure can go out to K=10 for all four architectures, not just Pool First.
#
# IMPORTANT: each (architecture, landscape) combo's best-noise operating
# point is NOT uniformly 0.0 here (unlike Pool First, where all 3 GB1
# landscapes happened to share noise=0.0) -- verified directly from the
# existing 3-seed data before writing this, since the ensemble-variance
# script picks a single best-noise level per combo and only uses whichever
# seeds exist AT that noise. New seeds MUST land at the same noise level
# already established by the existing seeds, or they're invisible to the
# analysis. Measured via make_architecture_ensemble_variance_figure.py's
# _load_seed_vectors + best-noise selection:
#   PreMLP additive=0.1, PreMLP epistatic=0.0, PreMLP nonlinear=0.0
#   MeanPool additive/epistatic/nonlinear = 0.0
#   AttnPool additive=0.0, AttnPool epistatic=0.05, AttnPool nonlinear=0.0
#
# Task list: 3 archs x 3 landscapes x 7 new seeds (4-10) = 63
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-63
#SBATCH --output=jobs/logs/gb1_other_archs_extra_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

SEEDS=(4 5 6 7 8 9 10)

declare -A OVR=(
  [PreMLP]="model.pooling_method=esm_attention_premlp"
  [MeanPool]="model.pooling_method=esm_perres_mean"
  [AttnPool]="model.pooling_method=esm_perres_attn"
)
declare -A OUTDIR=(
  [PreMLP]="results/synthetic/premlp"
  [MeanPool]="results/synthetic/newarch_perres_mean"
  [AttnPool]="results/synthetic/newarch_perres_attn"
)
declare -A NOISE=(
  [PreMLP,additive]=0.1   [PreMLP,epistatic]=0.0   [PreMLP,nonlinear]=0.0
  [MeanPool,additive]=0.0 [MeanPool,epistatic]=0.0 [MeanPool,nonlinear]=0.0
  [AttnPool,additive]=0.0 [AttnPool,epistatic]=0.05 [AttnPool,nonlinear]=0.0
)

ARCHS=(PreMLP MeanPool AttnPool)
LAND=(additive epistatic nonlinear)

TASKS=()
for A in "${ARCHS[@]}"; do
  for D in "${LAND[@]}"; do
    for S in "${SEEDS[@]}"; do
      TASKS+=("$A $D $S")
    done
  done
done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read A D S <<< "$t"
N="${NOISE[$A,$D]}"
CFG="configs/synthetic/GB1_${D}_config.yaml"
MD="${OUTDIR[$A]}/GB1_${D}/noise_${N}/seed_${S}"

echo ">>> [$SLURM_ARRAY_TASK_ID] GB1 arch=$A D=$D noise=$N S=$S MD=$MD"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  ${OVR[$A]} integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
