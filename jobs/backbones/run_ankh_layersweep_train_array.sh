#!/bin/bash
# Intermediate-layer sweep for ankh_base (MeanPool/esm_perres_mean only --
# the consistent-across-metrics architecture, per the backbone comparison).
# ankh_base has 48 encoder layers (49 hidden-state points incl. the
# embedding output); this sweeps 5 evenly-spaced depths: 0 (embedding
# output), 12, 24, 36, 48 (final layer). Deliberately excludes layer 33 --
# see rebuild_warm_cache_ankh_layersweep.sh's header for why (mislabeled
# pre-existing cache).
#
# Requires src/igloo/esm_adapter.py's AnkhBackend layer support (added +
# verified via jobs/smoketests/smoketest_ankh_layers.sh) and a warm per-layer cache
# (jobs/cache/rebuild_warm_cache_ankh_layersweep.sh) before running.
#
# CRITICAL: always pass model.esm_layer explicitly (never rely on the
# config default of 33, which is an ESM2-era value meaningless for Ankh's
# 48-layer depth and would silently select the wrong layer now that
# AnkhBackend actually honors it).
#
# Task list: 5 layers x 2 proteins x 3 landscapes x 9 noise x 3 seeds = 810
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --array=1-810
#SBATCH --output=jobs/logs/rb_ankh_layersweep_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

BACKBONE="ankh_base"
ARCH="esm_perres_mean"
LAYERS=(0 12 24 36 48)
PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3)

TASKS=()
for L in "${LAYERS[@]}"; do for P in "${PROTS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do for S in "${SEEDS[@]}"; do
    TASKS+=("$L $P $D $N $S")
  done; done
done; done; done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read LAYER P D N S <<< "$t"

NBYSEQ=$(find "data/embeddings/by_seq/${BACKBONE}/L${LAYER}" -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$NBYSEQ" -lt 5000 ]]; then
    echo "ABORT: by_seq cache for ${BACKBONE} L${LAYER} not warm (only $NBYSEQ files)." >&2
    exit 1
fi

CFG="configs/synthetic/${P}_${D}_config.yaml"
MD="results/synthetic/meanpool_ankh_base_layersweep/L${LAYER}/${P}_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] layer=$LAYER backbone=$BACKBONE arch=$ARCH P=$P D=$D N=$N S=$S"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.esm_model_name="$BACKBONE" model.pooling_method="$ARCH" model.esm_layer="$LAYER" \
  integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
