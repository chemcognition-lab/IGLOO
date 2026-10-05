#!/bin/bash
# ISM cross-validation for the 5 alt architectures (channel, perres_mean,
# perres_attn, crossattn, premlp), which have never had ISM run against them
# before -- only the main (esm_attention) architecture did, via
# run_attr_array_v2.sh's ism/ismS/ismX task kinds.
#
# Submit with --dependency=afterany:<training job ids> so all models exist
# first. ism_attribution.py is architecture-agnostic -- it reads
# pooling_method from each run's own saved config snapshot.
#
# Task list: 5 archs x 2 proteins x 3 landscapes x 9 noise x 3 seeds = 810
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:50:00
#SBATCH --array=1-810
#SBATCH --output=jobs/logs/rb_ism_newarch_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 30 ))

PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3)
declare -A ARCH_DIRS=(
  [channel]="newarch_channel"
  [perres_mean]="newarch_perres_mean"
  [perres_attn]="newarch_perres_attn"
  [crossattn]="newarch_crossattn"
  [premlp]="premlp"
)
ARMS=(channel perres_mean perres_attn crossattn premlp)

TASKS=()
for A in "${ARMS[@]}"; do for P in "${PROTS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do for S in "${SEEDS[@]}"; do
    TASKS+=("$A $P $D $N $S")
  done; done
done; done; done

echo "total tasks: ${#TASKS[@]}"
t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read A P D N S <<< "$t"
ARCH_DIR="${ARCH_DIRS[$A]}"
MD="results/synthetic/${ARCH_DIR}/${P}_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] ARM=$A P=$P D=$D N=$N S=$S  MD=$MD"

if [[ ! -f "${MD}/${P}_${D}_training_model.pth" ]]; then
    echo "SKIP: no trained model at $MD" >&2
    exit 0
fi

python scripts/pipeline/ism_attribution.py --run-dir "${MD}/linear" --model-dir "$MD" \
  --out results/synthetic/ism_attribution
