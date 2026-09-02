#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=fig2a_gb1_additive
#SBATCH --output=jobs/logs/slurm-%j.out

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"

PROJECT_ROOT=${IGLOO_ROOT}
BACKUP_ZIP=/project/aip-bensl/IGLOO/results_seed0_4.zip

cd "$PROJECT_ROOT"

# 1. Load module + set up venv (matches jobs/run_GB1_additive.sh convention)
module load python/3.12

if [ ! -x "IGLOO_env/bin/python" ]; then
    python -m venv IGLOO_env
    source IGLOO_env/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    source IGLOO_env/bin/activate
fi

# 2. Restore the GB1_additive noise-sweep, multi-seed results from the backup
#    zip (linear / nearest_neighbor / slerp interpolation methods only).
mkdir -p results/synthetic
unzip -n -q "$BACKUP_ZIP" \
    "results/synthetic/GB1_additive/*" \
    "results/synthetic/evaluation/*" \
    -d .

echo "Extracted GB1_additive result tree:"
find results/synthetic/GB1_additive -maxdepth 3 -type d | sort

# 3. Regenerate the attribution-quality figure (includes the Global-NDCG
#    panel equivalent to Figure 2A) for GB1 additive, multi-seed averaged.
python scripts/plot_ig_quality.py \
    --proteins GB1 \
    --fitness-models additive \
    --multi-seed \
    --output-dir results/synthetic/figures

echo "Done. Figures written to results/synthetic/figures/"
