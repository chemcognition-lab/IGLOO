#!/bin/bash
# Rebuild v3: regenerate the 4 esmc_300m-based new architectures after the
# BOS/EOS embedding fix (src/igloo/esm_adapter.py, 2026-09-15). `onehot` is
# EXCLUDED -- it never touches ESM-C and its earlier results remain valid.
#
#   channel        : item 2 - ESMAttentionChannel
#   perres_mean    : item 3a - ESMPerResidueMLP, mean pool
#   perres_attn    : item 3b - ESMPerResidueMLP, attention pool
#   crossattn      : item 3c - ESMCrossAttnPerResidue
#
# Task list: P x D x N x S x ARM = 2x3x9x3x4 = 648
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --array=1-648
#SBATCH --output=jobs/logs/rb_newarch_v3_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 30 ))

NBYSEQ=$(find data/embeddings/by_seq/esmc_300m -name '*.safetensors' 2>/dev/null | head -5000 | wc -l)
if [[ "$NBYSEQ" -lt 5000 ]]; then
    echo "ABORT: by_seq esmc_300m cache not warm (only $NBYSEQ files). Run jobs/cache/rebuild_warm_cache_v3.sh first." >&2
    exit 1
fi

PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3)
ARMS=(channel perres_mean perres_attn crossattn)

TASKS=()
for P in "${PROTS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do for S in "${SEEDS[@]}"; do for A in "${ARMS[@]}"; do
    TASKS+=("$A $P $D $N $S")
  done; done; done
done; done

echo "total tasks: ${#TASKS[@]}"
t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read ARM P D N S <<< "$t"
CFG="configs/synthetic/${P}_${D}_config.yaml"
echo ">>> [$SLURM_ARRAY_TASK_ID] ARM=$ARM P=$P D=$D N=$N S=$S"

case "$ARM" in
  channel)
    MD="results/synthetic/newarch_channel/${P}_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_attention_channel"
    ;;
  perres_mean)
    MD="results/synthetic/newarch_perres_mean/${P}_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_perres_mean"
    ;;
  perres_attn)
    MD="results/synthetic/newarch_perres_attn/${P}_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_perres_attn"
    ;;
  crossattn)
    MD="results/synthetic/newarch_crossattn/${P}_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_crossattn_perres"
    ;;
esac

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  $OVR integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
