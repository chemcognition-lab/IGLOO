#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=ndcg_vs_model_perf
#SBATCH --output=jobs/logs/slurm-%j.out

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"
cd "${IGLOO_ROOT}"
module load python/3.12
source IGLOO_env/bin/activate

python scripts/plot_ndcg_vs_model_performance.py \
    --protein GB1 --fitness-model additive \
    --interp linear \
    --x-metric model_test_spearman \
    --y-metric ndcg auroc ap_graded pv_ndcg_mean sign_acc cat_sep mutation_ndcg_mean_per_seq \
    --csv-out results/synthetic/figures/GB1_additive_signal_families_vs_model_perf_raw_linear.csv

echo "Done."
