#!/bin/bash
# Retry the single e1_600m task that hit the 2:50:00 wall clock (task 261:
# esm_attention_premlp/T4Lysozyme_additive/noise_0.5/seed_3 -- the longest
# protein on the largest/slowest E1 model). Already benefits from the
# plot.py per-sequence-plot cap; this was a genuine wall-clock issue, not
# the disk-quota bug, so just give it more time.
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:30:00
#SBATCH --output=jobs/logs/rb_plm_fillin_e1600m_retry1_%j.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env_e1/bin/activate

ARCH="esm_attention_premlp"
BACKBONE="e1_600m"
P="T4Lysozyme"
D="additive"
N="0.5"
S="3"
CFG="configs/synthetic/${P}_${D}_config.yaml"
MD="results/synthetic/${ARCH}_${BACKBONE}/${P}_${D}/noise_${N}/seed_${S}"
echo ">>> retry: arch=$ARCH backbone=$BACKBONE P=$P D=$D N=$N S=$S"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.esm_model_name="$BACKBONE" model.pooling_method="$ARCH" \
  integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
