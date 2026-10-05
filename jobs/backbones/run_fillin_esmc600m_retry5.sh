#!/bin/bash
# Retry the 5 esmc_600m fill-in tasks that failed with Errno 122 (disk quota
# exceeded) writing thousands of per-sequence diagnostic PNGs for T4Lysozyme.
# Fixed by capping plot_integrated_gradients() to max_sequences=40
# (src/igloo/plot.py); orphaned debris from the failed attempts already
# cleaned up manually.
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:50:00
#SBATCH --array=1-5
#SBATCH --output=jobs/logs/rb_plm_fillin_esmc600m_retry5_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

BACKBONE="esmc_600m"
module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

TASKS=(
  "esm_perres_attn T4Lysozyme additive 0.05 1"
  "esm_perres_attn T4Lysozyme additive 0.05 2"
  "esm_perres_attn T4Lysozyme epistatic 0.2 1"
  "esm_perres_attn T4Lysozyme epistatic 0.05 2"
  "esm_perres_attn T4Lysozyme epistatic 0.2 2"
)

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
