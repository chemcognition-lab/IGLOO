#!/bin/bash
# Fresh regeneration of scripts/evaluation/metric_suite.py now that p53 backbone-comparison
# (696 tasks) and layer-sweep (135 tasks) are both fully complete -- p53 should
# now populate all comparison figures alongside GB1/T4Lysozyme as a genuine
# third protein. Old results/synthetic/metric_suite/ was deleted beforehand
# for a clean rebuild rather than an incremental update.
#
# CPU only, same resource profile as jobs/analysis/run_analysis.sh (which wraps this
# script among others).
#
#SBATCH --account=def-mahadeva_cpu
#SBATCH --job-name=metric_suite_fresh
#SBATCH --output=jobs/logs/metric_suite_fresh_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:15:00

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

python scripts/evaluation/metric_suite.py --root results/synthetic
