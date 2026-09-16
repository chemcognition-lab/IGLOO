#!/bin/bash
# Option-B rebuild v2 — TRAINING wave, BLEND label-noise parametrisation.
#
# Noise axis is now scale-free:  training.noise.label_noise_mode=blend
#   p = label_noise_level in [0,1]
#   p=0   -> exact labels
#   p=1   -> full permutation  (held-out rho ~ 0; == shuffled-label baseline)
#   corr(noisy,true) = (1-p)/sqrt((1-p)^2 + p^2)
#
# Task list (index = SLURM_ARRAY_TASK_ID-1):
#   noise sweep : P x D x N                 (54)
#   seed sweep  : P x D x N x S             (270)
#   shuffled    : P x D x S                 (30)   [kept: independent p=1 check + analysis-script compat]
#   premlp      : P x D x {0.0,1.0} x {1,2,3}(36)
# Total 390.
#
#SBATCH --account=aip-bensl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --output=jobs/logs/rb_train_v2_%A_%a.out

set -u
cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 30 ))

# fail fast if the float32 by_seq cache is not warm
NBYSEQ=$(find data/embeddings/by_seq -name '*.safetensors' 2>/dev/null | head -5000 | wc -l)
if [[ "$NBYSEQ" -lt 5000 ]]; then
    echo "ABORT: by_seq cache not warm (only $NBYSEQ files). Run jobs/rebuild_warm_cache.sh first." >&2
    exit 1
fi

PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3 4 5)

TASKS=()
for P in "${PROTS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do
    TASKS+=("noise $P $D $N 0")
    for S in "${SEEDS[@]}"; do TASKS+=("seed $P $D $N $S"); done
  done
  for S in "${SEEDS[@]}"; do TASKS+=("shuf $P $D 0 $S"); done
  for N in 0.0 1.0; do for S in 1 2 3; do TASKS+=("premlp $P $D $N $S"); done; done
done; done

echo "total tasks: ${#TASKS[@]}"
t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read KIND P D N S <<< "$t"
CFG="configs/synthetic/${P}_${D}_config.yaml"
echo ">>> [$SLURM_ARRAY_TASK_ID] $KIND P=$P D=$D N=$N S=$S"

case "$KIND" in
  noise)
    MD="results/synthetic/${P}_${D}/noise_${N}"
    python scripts/run_integrated_gradients.py "$CFG" \
      integrated_gradients.interpolation_path=linear \
      training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
      training.noise.label_noise_seed=0 training.seed=42 \
      model_dir="$MD" results_dir="${MD}/linear" force_retrain=true ;;
  seed)
    MD="results/synthetic/seedsweep/${P}_${D}/noise_${N}/seed_${S}"
    python scripts/run_integrated_gradients.py "$CFG" \
      integrated_gradients.interpolation_path=linear \
      training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
      training.noise.label_noise_seed="$S" training.seed="$S" \
      model_dir="$MD" results_dir="${MD}/linear" force_retrain=true ;;
  shuf)
    MD="results/synthetic/seedsweep/${P}_${D}/shuffled/seed_${S}"
    python scripts/run_integrated_gradients.py "$CFG" \
      integrated_gradients.interpolation_path=linear \
      training.shuffle_labels=true training.shuffle_labels_seed="$S" \
      training.seed="$S" model_dir="$MD" results_dir="${MD}/linear" force_retrain=true ;;
  premlp)
    MD="results/synthetic/premlp/${P}_${D}/noise_${N}/seed_${S}"
    python scripts/run_integrated_gradients.py "$CFG" \
      model.pooling_method=esm_attention_premlp integrated_gradients.interpolation_path=linear \
      training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
      training.noise.label_noise_seed="$S" training.seed="$S" \
      model_dir="$MD" results_dir="${MD}/linear" force_retrain=true ;;
esac
