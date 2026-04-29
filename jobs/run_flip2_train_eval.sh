#!/bin/bash
# SLURM job script for training + evaluation on a FLIP2 alpha-amylase split.
#
# Required env var (set by submit_flip2_train_eval.sh via --export):
#   CONFIG   absolute path to the YAML config

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --time=5:20:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ranamoneim@gmail.com
# --job-name and --output are set by submit_flip2_train_eval.sh

cd /scratch/ranaab/IGLOO || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${HF_HOME}"

echo "=========================================="
echo "FLIP2 TRAIN+EVAL JOB"
echo "  config : ${CONFIG}"
echo "  GPU    : l40s"
echo "=========================================="

python scripts/run_train_and_eval.py "${CONFIG}"
