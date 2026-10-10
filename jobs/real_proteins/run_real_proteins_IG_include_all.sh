#!/bin/bash
# Re-run IG (no retraining) with include_all=true on the real-protein combos
# that previously only attributed the 5 default sequences (WT, PolyAlanine,
# highest/lowest/median):
#   - FLIP2 alpha_amylase__position__far_to_close: config had a typo
#     ("include all" instead of "include_all"), silently read as false.
#   - ProteinGym BLAT_ECOLX_* (4 studies): include_all was set to false.
# Both configs are now fixed, so each task just reloads the existing
# *_training_model.pth (force_retrain=false; checkpoint config is checked
# against the current model config) and runs IG on every sequence.
#
# Seeds match run_real_proteins_array.sh (seed 42, base dir) and
# run_real_proteins_ensemble_array.sh (seeds 1-4, seed_* subdirs), so
# ensemble_real_protein_attributions.py can be re-run unchanged afterwards.
#
# --time=2:00:00: a checkpoint-reload FLIP2 run did IG on 3,707 x 425-residue
# sequences in ~29 min; BLAT is <=~4,500 x 286 residues.
#
# Task list: 5 configs x 2 archs x 5 seeds = 50
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-50
#SBATCH --output=jobs/logs/run_real_proteins_IG_include_all_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

CONFIGS=(
  configs/FLIP2/alpha_amylase__position__far_to_close.yaml
  configs/ProteinGym/BLAT_ECOLX_Deng_2012_config.yaml
  configs/ProteinGym/BLAT_ECOLX_Firnberg_2014_config.yaml
  configs/ProteinGym/BLAT_ECOLX_Jacquier_2013_config.yaml
  configs/ProteinGym/BLAT_ECOLX_Stiffler_2015_config.yaml
)
BASE_RESULTS_DIRS=(
  results/FLIP2/alpha_amylase__position__far_to_close
  results/ProteinGym/BLAT_ECOLX_Deng_2012
  results/ProteinGym/BLAT_ECOLX_Firnberg_2014
  results/ProteinGym/BLAT_ECOLX_Jacquier_2013
  results/ProteinGym/BLAT_ECOLX_Stiffler_2015
)
ARCHS=(esm_perres_mean esm_perres_attn)
SEEDS=(42 1 2 3 4)

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
RESULTS_DIR="${BASE_RESULTS_DIRS[$CFG_IDX]}_${ARCH}"
[[ "$SEED" != "42" ]] && RESULTS_DIR="${RESULTS_DIR}/seed_${SEED}"
echo ">>> [$SLURM_ARRAY_TASK_ID] config=$CFG arch=$ARCH seed=$SEED results_dir=$RESULTS_DIR"

# Refuse to run without a checkpoint: force_retrain=false would otherwise
# silently train a fresh model.
if ! ls "$RESULTS_DIR"/*_training_model.pth >/dev/null 2>&1; then
    echo "ABORT: no trained checkpoint in $RESULTS_DIR" >&2
    exit 1
fi

NBYSEQ=$(find "data/embeddings/by_seq/ankh_base/L48" -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$NBYSEQ" -lt 3000 ]]; then
    echo "ABORT: by_seq cache for ankh_base L48 not warm (only $NBYSEQ files)." >&2
    exit 1
fi

SECONDS=0
python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.pooling_method="$ARCH" training.seed="$SEED" \
  results_dir="$RESULTS_DIR" force_retrain=false
echo ">>> elapsed: ${SECONDS}s"
