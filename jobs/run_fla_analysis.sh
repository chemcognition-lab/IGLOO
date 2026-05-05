#!/bin/bash
#SBATCH --job-name=fla_analysis
#SBATCH --output=results/synthetic/fla_analysis/fla_analysis_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00

cd /scratch/ranaab/IGLOO
source IGLOO_env/bin/activate

# Optional: pass PROTEIN env var to restrict to one protein
PROTEIN_ARG=""
if [[ -n "${PROTEIN:-}" ]]; then
    PROTEIN_ARG="--protein=${PROTEIN}"
fi

python scripts/fla_correlation_analysis.py ${PROTEIN_ARG}
