#!/bin/bash
# p53 intermediate-layer sweep for ankh_base/MeanPool -- same design as
# jobs/backbones/run_ankh_layersweep_train_array.sh (GB1/T4Lysozyme), scoped to p53
# alone. Always passes model.esm_layer explicitly per task (never relies on
# the config's shared esm_layer=33 default), so this script is unaffected
# by the final-layer convention issue described in
# rebuild_warm_cache_p53_backbones.sh / run_p53_backbone_array.sh.
#
# Deliberately excludes layer 33 -- see rebuild_warm_cache_ankh_layersweep.sh
# for the ankh_base/L33 mislabeled-cache landmine this sidesteps.
#
# --mem=128G / --time=4:00:00: matches run_p53_backbone_array.sh, validated
# at this protein/dataset scale by the p53 pilot.
#
# Requires the by_seq cache warm from rebuild_warm_cache_p53_backbones.sh
# (ankh_base layers 0/12/24/36/48).
#
# Task list: 5 layers x 3 landscapes x 9 noise x 3 seeds = 405
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=5:30:00
#SBATCH --array=1-405
#SBATCH --output=jobs/logs/rb_p53_layersweep_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

BACKBONE="ankh_base"
ARCH="esm_perres_mean"
LAYERS=(0 12 24 36 48)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3)

TASKS=()
for L in "${LAYERS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do for S in "${SEEDS[@]}"; do
    TASKS+=("$L $D $N $S")
  done; done
done; done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read LAYER D N S <<< "$t"

NBYSEQ=$(find "data/embeddings/by_seq/${BACKBONE}/L${LAYER}" -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$NBYSEQ" -lt 5000 ]]; then
    echo "ABORT: by_seq cache for ${BACKBONE} L${LAYER} not warm (only $NBYSEQ files)." >&2
    exit 1
fi

CFG="configs/synthetic/p53_${D}_config.yaml"
MD="results/synthetic/meanpool_ankh_base_layersweep/L${LAYER}/p53_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] layer=$LAYER D=$D N=$N S=$S"

SECONDS=0
python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.esm_model_name="$BACKBONE" model.pooling_method="$ARCH" model.esm_layer="$LAYER" \
  integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
echo ">>> elapsed: ${SECONDS}s"
