#!/bin/bash
# One-hot amino-acid baseline (no pLM at all -- tests how much the protein
# language model's embedding geometry actually adds over raw residue
# identity). Earlier pilot only covered noise in {0,1} x 3 seeds under a
# different project account (aip-bensl); this is the full 9-noise-level x
# 3-seed sweep under the current account/path, matching every other method
# already in the metric_suite _band figures, so OneHot can finally appear
# alongside them.
#
# No embedding cache warm-up needed: esm_model_name=onehot bypasses the pLM
# entirely (src/igloo/embedding.py's _onehot_embeddings, a deterministic
# per-sequence encoding) -- cheapest job in this whole repo.
#
# Task list: 2 proteins x 3 landscapes x 9 noise x 3 seeds = 162
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --array=1-162
#SBATCH --output=jobs/logs/run_onehot_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

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
MD="results/synthetic/onehot/${P}_${D}/noise_${N}/seed_${S}"
echo ">>> [$SLURM_ARRAY_TASK_ID] P=$P D=$D N=$N S=$S"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.esm_model_name=onehot \
  integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
  training.noise.label_noise_seed="$S" training.seed="$S" \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
