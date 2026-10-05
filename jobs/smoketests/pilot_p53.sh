#!/bin/bash
# Small pilot: validate the new-backbone pipeline on p53 (393 residues) and
# get real per-task timing before committing to the full backbone-comparison
# + layer-sweep replication for this protein. Uses ankh_large (the
# established winner backbone), MeanPool + AttnPool (the two reliable
# architectures), noise=0.0, seed=1, across all 3 landscapes.
#
# --mem=128G (bumped from 64G, which also OOM-killed all 6 tasks -- all at
# the same ~65.5GB ceiling regardless of arch/landscape, right at the cgroup
# limit). Root cause: model training loads the full per-residue embedding
# matrix for every dataset variant (n_variants x seq_len x hidden_dim). At
# ankh_large's embedding width that's ~0.5GB for GB1, ~3.8GB for
# T4Lysozyme (both fine at 32G) but ~22.5GB alone for p53 (10,002 variants x
# 393 residues) -- with a few copies in flight during load/split/train,
# landing at ~65-70GB required lines up with the observed failures.
#
# Task list: 3 landscapes x 2 archs = 6
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --array=1-6
#SBATCH --output=jobs/logs/pilot_p53_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

BACKBONE="ankh_large"
LAND=(additive epistatic nonlinear)
ARCHS=(esm_perres_mean esm_perres_attn)

TASKS=()
for D in "${LAND[@]}"; do for A in "${ARCHS[@]}"; do
  TASKS+=("$D $A")
done; done
echo "total tasks: ${#TASKS[@]}"

t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read D ARCH <<< "$t"
CFG="configs/synthetic/p53_${D}_config.yaml"
MD="results/synthetic/pilot_p53/${ARCH}_${BACKBONE}/p53_${D}/noise_0.0/seed_1"
echo ">>> [$SLURM_ARRAY_TASK_ID] p53 landscape=$D arch=$ARCH backbone=$BACKBONE"

SECONDS=0
python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.esm_model_name="$BACKBONE" model.pooling_method="$ARCH" \
  integrated_gradients.interpolation_path=linear \
  training.noise.label_noise_mode=blend training.noise.label_noise_level=0.0 \
  training.noise.label_noise_seed=1 training.seed=1 \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
echo ">>> elapsed: ${SECONDS}s"
