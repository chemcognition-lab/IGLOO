#!/bin/bash
# 5-seed ensemble for the real-protein "production" runs (alpha-amylase +
# beta-lactamase, ankh_base + MeanPool/AttnPool) -- adds 4 new seeds
# (1,2,3,4) alongside the existing single seed=42 run from
# run_real_proteins_array.sh, so each combo has 5 independent seeds total.
# Motivation: the attribution-agreement analysis found within-model seed
# agreement of only ~0.70 Spearman (not ~0.95+), meaning individual seeds
# are genuinely noisy -- an ensemble mean (and per-residue reliability
# ratio |mean|/std across seeds) gives a materially more trustworthy
# attribution than any single seed alone. 5 seeds (not 10): standard-error
# reduction scales as 1/sqrt(N), so 3->5 buys a real gain while 5->10 buys
# much less for double the cost.
#
# Each task gets its own results_dir (seed_${S} subfolder) and explicit
# force_retrain=true -- without a unique directory, force_retrain=false
# would just find the existing seed-42 checkpoint and reuse it, defeating
# the point of an independently-seeded ensemble member.
#
# Requires the by_seq cache warm from rebuild_warm_cache_real_proteins.sh
# (ankh_base L48 for both protein families) -- already done.
#
# --time=4:00:00 / --mem=32G: matches run_real_proteins_array.sh (same
# configs/archs, just a different seed).
#
# Task list: 8 configs x 2 archs x 4 new seeds = 64
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=1-64
#SBATCH --output=jobs/logs/run_real_proteins_ensemble_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

CONFIGS=(
  configs/FLIP2/alpha_amylase__mutation__by_mutation.yaml
  configs/FLIP2/alpha_amylase__number__one_to_many.yaml
  configs/FLIP2/alpha_amylase__position__close_to_far.yaml
  configs/FLIP2/alpha_amylase__position__far_to_close.yaml
  configs/ProteinGym/BLAT_ECOLX_Deng_2012_config.yaml
  configs/ProteinGym/BLAT_ECOLX_Firnberg_2014_config.yaml
  configs/ProteinGym/BLAT_ECOLX_Jacquier_2013_config.yaml
  configs/ProteinGym/BLAT_ECOLX_Stiffler_2015_config.yaml
)
BASE_RESULTS_DIRS=(
  results/FLIP2/alpha_amylase__mutation__by_mutation
  results/FLIP2/alpha_amylase__number__one_to_many
  results/FLIP2/alpha_amylase__position__close_to_far
  results/FLIP2/alpha_amylase__position__far_to_close
  results/ProteinGym/BLAT_ECOLX_Deng_2012
  results/ProteinGym/BLAT_ECOLX_Firnberg_2014
  results/ProteinGym/BLAT_ECOLX_Jacquier_2013
  results/ProteinGym/BLAT_ECOLX_Stiffler_2015
)
ARCHS=(esm_perres_mean esm_perres_attn)
SEEDS=(1 2 3 4)

TASKS=()
for i in "${!CONFIGS[@]}"; do
  for a in "${ARCHS[@]}"; do
    for s in "${SEEDS[@]}"; do
      TASKS+=("$i $a $s")
    done
  done
done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read CFG_IDX ARCH SEED <<< "$t"
CFG="${CONFIGS[$CFG_IDX]}"
RESULTS_DIR="${BASE_RESULTS_DIRS[$CFG_IDX]}_${ARCH}/seed_${SEED}"
echo ">>> [$SLURM_ARRAY_TASK_ID] config=$CFG arch=$ARCH seed=$SEED results_dir=$RESULTS_DIR"

NBYSEQ=$(find "data/embeddings/by_seq/ankh_base/L48" -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$NBYSEQ" -lt 3000 ]]; then
    echo "ABORT: by_seq cache for ankh_base L48 not warm (only $NBYSEQ files)." >&2
    exit 1
fi

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.pooling_method="$ARCH" training.seed="$SEED" \
  results_dir="$RESULTS_DIR" force_retrain=true
