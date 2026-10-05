#!/bin/bash
# p53 pLM backbone comparison, reduced scope: top 3 backbones by TOPSIS
# score from the GB1/T4Lysozyme comparison (ankh_base 0.698, ankh_large
# 0.691, esmc_600m 0.586 -- see results/synthetic/metric_suite/
# backbone_topsis_cells.csv), instead of all 7. Everything else matches the
# GB1/T4L grid: 3 archs x 3 landscapes x 9 noise x 3 seeds, single protein
# (p53) since this is a p53-only replication.
#
# IMPORTANT: explicitly overrides model.esm_layer=48 for ankh_base/
# ankh_large (the real final layer -- see rebuild_warm_cache_p53_backbones.sh
# for why the config's shared default esm_layer=33 no longer means "final"
# for Ankh now that AnkhBackend honors it for real). esmc_600m is left on
# the config default (33), unchanged, since ESMCBackend still silently
# ignores `layer` and always returns the true final layer.
#
# --mem=128G / --time=4:00:00: validated by the p53 pilot (jobs/smoketests/pilot_p53.sh
# history) at this same protein/dataset scale for MeanPool/AttnPool;
# PreMLP (esm_attention_premlp) is untested at this scale -- if any PreMLP
# tasks OOM, retry just those with a scoped memory bump (same pattern as
# the pilot's own 32G->64G->128G escalation).
#
# Requires the by_seq cache warm from rebuild_warm_cache_p53_backbones.sh.
#
# Task list: 3 backbones x 3 archs x 3 landscapes x 9 noise x 3 seeds = 729
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --array=1-729
#SBATCH --output=jobs/logs/rb_p53_backbone_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

BACKBONES=(ankh_base ankh_large esmc_600m)
ARCHS=(esm_perres_attn esm_attention_premlp esm_perres_mean)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3)

TASKS=()
for BB in "${BACKBONES[@]}"; do for A in "${ARCHS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do for S in "${SEEDS[@]}"; do
    TASKS+=("$BB $A $D $N $S")
  done; done
done; done; done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read BB ARCH D N S <<< "$t"

LAYER_OVERRIDE=""
case "$BB" in
  ankh_base|ankh_large) LAYER_OVERRIDE="model.esm_layer=48" ;;
esac

NBYSEQ=$(find "data/embeddings/by_seq/${BB}" -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$NBYSEQ" -lt 5000 ]]; then
    echo "ABORT: by_seq cache for ${BB} not warm (only $NBYSEQ files)." >&2
    exit 1
fi

CFG="configs/synthetic/p53_${D}_config.yaml"
MD="results/synthetic/${ARCH}_${BB}/p53_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] backbone=$BB arch=$ARCH D=$D N=$N S=$S layer_override=${LAYER_OVERRIDE:-none}"

SECONDS=0
python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.esm_model_name="$BB" model.pooling_method="$ARCH" $LAYER_OVERRIDE \
  integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
echo ">>> elapsed: ${SECONDS}s"
