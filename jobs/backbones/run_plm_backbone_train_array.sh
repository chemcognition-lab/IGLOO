#!/bin/bash
# Full noise-grid training for one pLM backbone across all 3 alt
# architectures (perres_attn, IG_premlp/esm_attention_premlp, perres_mean),
# validated end-to-end via jobs/smoketests/pilot_plm_backbones.sh before this was written.
#
# BACKBONE must be passed via `sbatch --export=BACKBONE=<name>`. e1_* needs
# IGLOO_env_e1 (py3.12, separate venv); esmc/ankh use IGLOO_env (py3.11).
#
# Writes into results/synthetic/<arch>_<backbone>/... -- a NEW directory
# naming convention (not yet recognized by metric_suite.py; that needs its
# own follow-up once this data exists, same as the mutation_pool addition).
#
# Task list per backbone: 3 archs x 9 noise x 3 seeds x 2 proteins x 3 landscapes = 486
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --array=1-486
#SBATCH --output=jobs/logs/rb_plm_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

if [[ -z "${BACKBONE:-}" ]]; then
    echo "ABORT: BACKBONE not set. Submit with: sbatch --export=BACKBONE=<name> $0" >&2
    exit 1
fi

case "$BACKBONE" in
  e1_*)
    module load python/3.12 cuda/12.2
    source IGLOO_env_e1/bin/activate
    ;;
  *)
    module load python/3.12 gcc arrow cuda/12.2
    source IGLOO_env/bin/activate
    ;;
esac

NBYSEQ=$(find "data/embeddings/by_seq/${BACKBONE}" -name '*.safetensors' 2>/dev/null | wc -l)
if [[ "$NBYSEQ" -lt 5000 ]]; then
    echo "ABORT: by_seq cache for $BACKBONE not warm (only $NBYSEQ files)." >&2
    exit 1
fi

ARCHS=(esm_perres_attn esm_attention_premlp esm_perres_mean)
PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3)

TASKS=()
for A in "${ARCHS[@]}"; do for P in "${PROTS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do for S in "${SEEDS[@]}"; do
    TASKS+=("$A $P $D $N $S")
  done; done
done; done; done

echo "total tasks: ${#TASKS[@]}"
t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read ARCH P D N S <<< "$t"
CFG="configs/synthetic/${P}_${D}_config.yaml"
MD="results/synthetic/${ARCH}_${BACKBONE}/${P}_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] arch=$ARCH backbone=$BACKBONE P=$P D=$D N=$N S=$S"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.esm_model_name="$BACKBONE" model.pooling_method="$ARCH" \
  integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
