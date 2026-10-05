#!/bin/bash
# Full parity for p53 at the esmc_300m backbone, matching the GB1/T4Lysozyme
# method-comparison pipeline exactly, across all 5 methods: IG (Pool First),
# PreMLP, MeanPool, AttnPool, OneHot.
#
# Deliberately uses a CLEAN 3-seed design for IG here (not GB1/T4L's
# asymmetric "1 default run + separate 5-seed seedsweep" pattern) -- all 5
# methods get identical, directly comparable 3-seed x 9-noise x 3-landscape
# coverage. Writes IG into results/synthetic/seedsweep/p53_<D>/... (same
# directory metric_suite.py's ig_runs() already scans generically for any
# protein in PROTEINS, which already includes p53 -- no code changes needed).
#
# Resource profile matches the other p53 jobs (128G/5:30:00), not the
# smaller GB1/T4L premlp fill-in profile (32G/2:50:00) -- p53 is a much
# longer sequence (393 aa vs 56/164) with a correspondingly larger
# include_all IG analysis.
#
# Task list: 5 methods x 3 landscapes x 9 noise x 3 seeds = 405
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=5:30:00
#SBATCH --array=1-405
#SBATCH --output=jobs/logs/rb_p53_esmc300m_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

METHODS=(IG PreMLP MeanPool AttnPool OneHot)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3)

TASKS=()
for M in "${METHODS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do for S in "${SEEDS[@]}"; do
    TASKS+=("$M $D $N $S")
  done; done
done; done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read M D N S <<< "$t"
CFG="configs/synthetic/p53_${D}_config.yaml"

case "$M" in
  IG)
    MD="results/synthetic/seedsweep/p53_${D}/noise_${N}/seed_${S}"
    OVR=""
    ;;
  PreMLP)
    MD="results/synthetic/premlp/p53_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_attention_premlp"
    ;;
  MeanPool)
    MD="results/synthetic/newarch_perres_mean/p53_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_perres_mean"
    ;;
  AttnPool)
    MD="results/synthetic/newarch_perres_attn/p53_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_perres_attn"
    ;;
  OneHot)
    MD="results/synthetic/onehot/p53_${D}/noise_${N}/seed_${S}"
    OVR="model.esm_model_name=onehot"
    ;;
esac

echo ">>> [$SLURM_ARRAY_TASK_ID] method=$M D=$D N=$N S=$S MD=$MD"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  $OVR integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
