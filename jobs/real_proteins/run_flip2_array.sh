#!/bin/bash
# Run the IGLOO attribution pipeline on all 4 FLIP2 alpha-amylase configs
# (mutation__by_mutation, number__one_to_many, position__close_to_far,
# position__far_to_close). Each configs/FLIP2/*.yaml is fully self-contained
# (data path, split, model, results_dir all set by the config itself), so
# each task just runs run_integrated_gradients.py against its config with no
# CLI overrides -- unlike the synthetic-data jobs elsewhere in this repo.
#
# --mem=32G: dataset is modest (3,706 variants x 425 residues on esmc_300m,
# well under the T4Lysozyme/ankh_large scale that already runs fine at 32G).
# First real-dataset run in this pipeline -- watch sacct MaxRSS on
# completion and bump if it turns out to be tight (see the p53 pilot's OOM
# history in jobs/smoketests/pilot_p53.sh for the pattern to follow if so).
#
# Task list: 4 configs
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=1-4
#SBATCH --output=jobs/logs/run_flip2_%A_%a.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

CONFIGS=(
  configs/FLIP2/alpha_amylase__mutation__by_mutation.yaml
  configs/FLIP2/alpha_amylase__number__one_to_many.yaml
  configs/FLIP2/alpha_amylase__position__close_to_far.yaml
  configs/FLIP2/alpha_amylase__position__far_to_close.yaml
)

CFG="${CONFIGS[$((SLURM_ARRAY_TASK_ID - 1))]}"
echo ">>> [$SLURM_ARRAY_TASK_ID] FLIP2 config=$CFG"

SECONDS=0
python scripts/pipeline/run_integrated_gradients.py "$CFG"
echo ">>> elapsed: ${SECONDS}s"
