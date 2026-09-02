#!/bin/bash
# Submits the full GB1_additive noise sweep (12 noise levels x 5 seeds, linear
# interp only) plus the 5-seed shuffled-label random baseline, all for the
# REDUCED model (attention pooling + single linear layer). Mirrors the
# original full-model sweep (results/synthetic/GB1_additive) so the same 7
# metrics-vs-model-performance plots can be reproduced and compared.
#
# All 65 cells are independent and submitted fully in parallel.
#
# Usage:
#   bash jobs/submit_gb1_reduced_sweep.sh              # submit everything
#   bash jobs/submit_gb1_reduced_sweep.sh --dry-run     # print sbatch commands only

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"
cd "${IGLOO_ROOT}"
mkdir -p jobs/logs

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

NOISE_LEVELS=(0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
SEEDS=(0 1 2 3 4)

total=0
for NOISE in "${NOISE_LEVELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="GB1_reduced_linear_n${NOISE}_s${SEED}"
        CMD=(
            sbatch --parsable
            --job-name="${JOB_NAME}"
            --output="jobs/logs/${JOB_NAME}_%j.out"
            --exclude=kn010
            --export="NOISE=${NOISE},SEED=${SEED}"
            jobs/run_sweep_cell_gb1_reduced.sh
        )
        if $DRY_RUN; then
            echo "[dry-run] ${CMD[*]}"
        else
            JOBID=$("${CMD[@]}")
            echo "Submitted ${JOB_NAME} -> ${JOBID}"
        fi
        ((total++)) || true
    done
done

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="GB1_reduced_random_baseline_s${SEED}"
    CMD=(
        sbatch --parsable
        --job-name="${JOB_NAME}"
        --output="jobs/logs/${JOB_NAME}_%j.out"
        --exclude=kn010
        --export="SEED=${SEED}"
        jobs/run_random_baseline_cell_reduced.sh
    )
    if $DRY_RUN; then
        echo "[dry-run] ${CMD[*]}"
    else
        JOBID=$("${CMD[@]}")
        echo "Submitted ${JOB_NAME} -> ${JOBID}"
    fi
    ((total++)) || true
done

echo ""
echo "$($DRY_RUN && echo 'Would submit' || echo 'Submitted') ${total} jobs (60 sweep cells + 5 random-baseline cells)."
