#!/bin/bash
# Fill-in: the 7 missing intermediate noise levels for IG_premlp only, so it
# gets the same full 9-level grid as the main architecture (IG) and the
# newarch architectures (channel/perres_mean/perres_attn/crossattn, already
# covered end-to-end by jobs/architectures/run_newarch_train_array_v3.sh).
#
# Deliberately excludes:
#   - noise 0.0 and 1.0 for premlp -- already completed by the original
#     run_train_array_v2.sh submission (job 22091184).
#   - perres_mean/perres_attn/channel/crossattn -- already covered at all 9
#     noise levels by run_newarch_train_array_v3.sh (job 22128112).
#
# Writes into the SAME result directories metric_suite.py already scans
# (results/synthetic/premlp/...) -- no analysis-script changes needed.
#
# Task list: 2 proteins x 3 landscapes x 7 noise x 3 seeds = 126
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --array=1-126
#SBATCH --output=jobs/logs/rb_fillin_premlp_%A_%a.out

set -u
set -o pipefail
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
NOISE=(0.05 0.1 0.2 0.35 0.5 0.7 0.85)
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
MD="results/synthetic/premlp/${P}_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] premlp P=$P D=$D N=$N S=$S"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.pooling_method=esm_attention_premlp integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
