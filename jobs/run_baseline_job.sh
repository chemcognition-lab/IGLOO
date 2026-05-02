#!/bin/bash
# SLURM job script for running baseline attributions (gradient_x_input,
# input_difference) for one GB1 data-model × noise combination.
#
# Required env vars (set by submit_baseline_sweep.sh via --export):
#   CONFIG_SNAPSHOT   absolute path to the config snapshot YAML
#   JOB_LABEL         human-readable label for logging (e.g. GB1_additive_noise_0.1)

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ranamoneim@gmail.com
# --job-name and --output are set by submit_baseline_sweep.sh

cd /scratch/ranaab/IGLOO || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

# Redirect HuggingFace cache to local node storage (avoids mmap errors on Lustre)
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"

echo "=========================================="
echo "BASELINE JOB: ${JOB_LABEL}"
echo "  config snapshot : ${CONFIG_SNAPSHOT}"
echo "  baselines       : gradient_x_input input_difference"
echo "  GPU             : l40s"
echo "=========================================="

python scripts/run_baseline_attributions.py "${CONFIG_SNAPSHOT}" \
    --baselines gradient_x_input input_difference
