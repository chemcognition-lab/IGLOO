#!/bin/bash
# Pilot for a 6th architecture arm: per-residue MLP with NO pooling at all --
# the per-position hidden representations are flattened and fed directly into
# a single linear head (esm_perres_flatten / ESMPerResidueFlatten). Unlike
# esm_perres_mean/esm_perres_attn, the position -> prediction weights are
# freely learned per-position coefficients, not a mean or a softmax-normalized
# convex combination. Requires fixed-length input -- true here (point
# substitutions only, no indels).
#
# Independent of the main rebuild chain (train_array_v2 / attr_array_v2 /
# newarch_train_array_v3 / run_analysis) -- runs in parallel, does not touch
# their result directories.
#
# Task list: P x D x N x S = 2 x 3 x 2 x 3 = 36
#
#SBATCH --account=aip-bensl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --output=jobs/logs/rb_perresflat_%A_%a.out

set -u
cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 30 ))

NBYSEQ=$(find data/embeddings/by_seq/esmc_300m -name '*.safetensors' 2>/dev/null | head -5000 | wc -l)
if [[ "$NBYSEQ" -lt 5000 ]]; then
    echo "ABORT: by_seq esmc_300m cache not warm (only $NBYSEQ files)." >&2
    exit 1
fi

PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 1.0)
SEEDS=(1 2 3)

TASKS=()
for P in "${PROTS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do for S in "${SEEDS[@]}"; do
    TASKS+=("$P $D $N $S")
  done; done
done; done

echo "total tasks: ${#TASKS[@]}"
t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read P D N S <<< "$t"
CFG="configs/synthetic/${P}_${D}_config.yaml"
MD="results/synthetic/newarch_perres_flatten/${P}_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] P=$P D=$D N=$N S=$S"

python scripts/run_integrated_gradients.py "$CFG" \
  model.pooling_method=esm_perres_flatten integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
