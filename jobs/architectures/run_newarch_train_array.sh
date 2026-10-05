#!/bin/bash
# New comparison arms for the metric_suite figures (pilot scale, mirroring the
# premlp precedent: noise in {0,1} (blend semantics: 0=clean, 1=fully random)
# x 3 seeds, not the full 9-level grid — validate here before committing to a
# bigger sweep).
#
#   onehot         : item 1 - one-hot amino-acid input instead of ESM-C embedding
#                    (classic esm_attention head, esm_model_name=onehot)
#   channel        : item 2 - ESMAttentionChannel (attention pools the CHANNEL
#                    axis, not sequence length)
#   perres_mean    : item 3a - ESMPerResidueMLP, mean pool
#   perres_attn    : item 3b - ESMPerResidueMLP, attention pool
#   crossattn      : item 3c - ESMCrossAttnPerResidue (real cross-position
#                    self-attention before the per-residue MLP)
#
# Task list (index = SLURM_ARRAY_TASK_ID-1): P x D x N x S x ARM = 2x3x2x3x5 = 180
#
#SBATCH --account=aip-bensl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=jobs/logs/rb_newarch_%A_%a.out

set -u
cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 30 ))

NBYSEQ=$(find data/embeddings/by_seq -name '*.safetensors' 2>/dev/null | head -5000 | wc -l)
if [[ "$NBYSEQ" -lt 5000 ]]; then
    echo "ABORT: by_seq cache not warm (only $NBYSEQ files). Run jobs/cache/rebuild_warm_cache.sh first." >&2
    exit 1
fi

PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 1.0)
SEEDS=(1 2 3)
ARMS=(onehot channel perres_mean perres_attn crossattn)

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
  onehot)
    MD="results/synthetic/onehot/${P}_${D}/noise_${N}/seed_${S}"
    OVR="model.esm_model_name=onehot"
    ;;
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
