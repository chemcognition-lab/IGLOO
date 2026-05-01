#!/bin/bash
#SBATCH --job-name=fla_analysis
#SBATCH --output=results/synthetic/fla_analysis/fla_analysis_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd /scratch/ranaab/IGLOO
source IGLOO_env/bin/activate
python scripts/fla_correlation_analysis.py
