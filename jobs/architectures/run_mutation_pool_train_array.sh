#!/bin/bash
# Train the esm_mutation_pool architecture (mean-pools only over mutated
# positions relative to WT, instead of the full sequence) across the full
# noise grid, matching the other alt-architecture comparisons
# (run_newarch_train_array_v3.sh, run_fillin_premlp_full.sh).
#
# Writes into results/synthetic/mutation_pool/... which scripts/evaluation/metric_suite.py
# now recognizes (see the _newarch_dirs entry added alongside this job).
#
# Task list: 2 proteins x 3 landscapes x 9 noise x 3 seeds = 162
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --array=1-162
#SBATCH --output=jobs/logs/rb_mutation_pool_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 gcc arrow cuda/12.2
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
MD="results/synthetic/mutation_pool/${P}_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] mutation_pool P=$P D=$D N=$N S=$S"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.pooling_method=esm_mutation_pool integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
