#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --job-name=T4_epistatic
#SBATCH --output=jobs/logs/slurm-%j.out

# Change to project root (script lives in jobs/)
cd /project/aip-bensl/ranaab/IGLOO || exit 1

# 1. Load modules
module load python/3.12 cuda/12.2

# 2. Activate env
source IGLOO_env/bin/activate

# 3. Execute code for T4Lysozyme
python scripts/run_integrated_gradients.py configs/synthetic/T4Lysozyme_epistatic_config.yaml
