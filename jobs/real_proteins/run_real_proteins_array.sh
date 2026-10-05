#!/bin/bash
# Run alpha-amylase (FLIP2) and beta-lactamase/BLAT_ECOLX (ProteinGym) on
# our best-validated setup from the GB1/T4Lysozyme/p53 comparison:
# ankh_base backbone with both MeanPool (esm_perres_mean) and AttnPool
# (esm_perres_attn) architectures. The 8 configs already point at
# ankh_base/esm_perres_mean by default (see configs/FLIP2/alpha_amylase*.yaml
# and configs/ProteinGym/BLAT_ECOLX_*_config.yaml) -- this script runs each
# as-is for the MeanPool variant and overrides pooling_method + results_dir
# for the AttnPool variant, so both land in separate result directories.
#
# Requires the by_seq cache warm from rebuild_warm_cache_real_proteins.sh
# (ankh_base L48 for both protein families).
#
# --time=4:00:00 (bumped from 2:00:00, which TIMEOUT'd 6/16 tasks): the 4
# alpha-amylase configs use include_all=true (full 3,706-sequence IG
# analysis, same expensive pattern as p53's pilot) on top of training;
# BLAT_ECOLX_Deng_2012 (the largest beta-lactamase study, 4,996 rows) alone
# took 1h22m to train with esm_perres_attn, and its esm_perres_mean sibling
# didn't even finish training within 2h.
#
# Task list: 8 configs x 2 archs = 16
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=1-16
#SBATCH --output=jobs/logs/run_real_proteins_%A_%a.out

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

TASKS=()
for i in "${!CONFIGS[@]}"; do
  for a in "${ARCHS[@]}"; do
    TASKS+=("$i $a")
  done
done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read CFG_IDX ARCH <<< "$t"
CFG="${CONFIGS[$CFG_IDX]}"
RESULTS_DIR="${BASE_RESULTS_DIRS[$CFG_IDX]}_${ARCH}"
echo ">>> [$SLURM_ARRAY_TASK_ID] config=$CFG arch=$ARCH results_dir=$RESULTS_DIR"

NBYSEQ=$(find "data/embeddings/by_seq/ankh_base/L48" -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$NBYSEQ" -lt 3000 ]]; then
    echo "ABORT: by_seq cache for ankh_base L48 not warm (only $NBYSEQ files)." >&2
    exit 1
fi

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.pooling_method="$ARCH" results_dir="$RESULTS_DIR"
