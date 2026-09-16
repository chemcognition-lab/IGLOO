#!/bin/bash
# Fill-in: 3 extra noise levels for IG_premlp / perres_mean / perres_attn, so
# the "focused" metric_suite comparison has more than a 2-point pilot to work
# with, without committing to the full 7-extra-level sweep (756 tasks).
#
# Noise levels chosen FROM the existing 9-level grid (0, .05, .1, .2, .35, .5,
# .7, .85, 1) so they align exactly with IG/ISM's existing noise groups:
#   0.2  (low-mid), 0.5 (mid), 0.85 (high)
# combined with the already-existing pilot points (0.0, 1.0), that's 5 of the
# 9 levels covered for these three architectures.
#
# Same 3-seed convention as the original 2-point pilot (not IG's 5 seeds) --
# keeps this a modest fill-in, not a full-scale rerun.
#
# Writes into the SAME result directories metric_suite.py already scans
# (results/synthetic/premlp/... for IG_premlp, results/synthetic/newarch_perres_mean|attn/...)
# -- no analysis-script changes needed, these noise levels just show up
# automatically next time metric_suite.py runs.
#
# Task list: 3 archs x 2 proteins x 3 landscapes x 3 noise x 3 seeds = 162
#
#SBATCH --account=aip-bensl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --output=jobs/logs/rb_fillin_%A_%a.out

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
NOISE=(0.2 0.5 0.85)
SEEDS=(1 2 3)
ARMS=(premlp perres_mean perres_attn)

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
  premlp)
    MD="results/synthetic/premlp/${P}_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_attention_premlp"
    ;;
  perres_mean)
    MD="results/synthetic/newarch_perres_mean/${P}_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_perres_mean"
    ;;
  perres_attn)
    MD="results/synthetic/newarch_perres_attn/${P}_${D}/noise_${N}/seed_${S}"
    OVR="model.pooling_method=esm_perres_attn"
    ;;
esac

python scripts/run_integrated_gradients.py "$CFG" \
  $OVR integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
