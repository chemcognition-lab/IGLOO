#!/bin/bash
# Submits the 5 new noise levels (0.2, 0.4, 0.6, 0.7, 0.9) x 5 seeds x 3
# interpolation methods needed to fill GB1_additive out to a full 0.0-1.0
# decile noise grid (the existing 6 decile-aligned levels -- 0.0, 0.1, 0.3,
# 0.5, 0.8, 1.0 -- were already recovered from results_seed0_4.zip).
#
# For each (noise, seed) cell, the "linear" job is submitted first and
# "slerp"/"nearest_neighbor" are chained behind it with --dependency=afterok,
# since all 3 interp methods share one trained model per (noise, seed) and
# the model-loading code has no lock -- running them concurrently would
# redundantly retrain (and risks a corrupted checkpoint from concurrent
# torch.save). Cells are independent of each other and run fully in parallel.
#
# Usage:
#   bash jobs/submit_gb1_decile_fill.sh              # submit everything
#   bash jobs/submit_gb1_decile_fill.sh --dry-run     # print sbatch commands only

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"
cd "${IGLOO_ROOT}"
mkdir -p jobs/logs

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

NOISE_LEVELS=(0.2 0.4 0.6 0.7 0.9)
SEEDS=(0 1 2 3 4)

total=0
for NOISE in "${NOISE_LEVELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="GB1_additive_linear_n${NOISE}_s${SEED}"
        LINEAR_CMD=(
            sbatch --parsable
            --job-name="${JOB_NAME}"
            --output="jobs/logs/${JOB_NAME}_%j.out"
            --time="1:30:00"
            --export="NOISE=${NOISE},SEED=${SEED},INTERP=linear"
            jobs/run_sweep_cell_gb1_deciles.sh
        )
        if $DRY_RUN; then
            echo "[dry-run] ${LINEAR_CMD[*]}"
            LINEAR_JOBID="<linear_jobid>"
        else
            LINEAR_JOBID=$("${LINEAR_CMD[@]}")
            echo "Submitted ${JOB_NAME} -> ${LINEAR_JOBID}"
        fi
        ((total++)) || true

        for INTERP in slerp nearest_neighbor; do
            JOB_NAME="GB1_additive_${INTERP}_n${NOISE}_s${SEED}"
            JOB_TIME="1:30:00"
            [[ "${INTERP}" == "nearest_neighbor" ]] && JOB_TIME="2:30:00"
            DEP_CMD=(
                sbatch
                --job-name="${JOB_NAME}"
                --output="jobs/logs/${JOB_NAME}_%j.out"
                --time="${JOB_TIME}"
                --dependency="afterok:${LINEAR_JOBID}"
                --export="NOISE=${NOISE},SEED=${SEED},INTERP=${INTERP}"
                jobs/run_sweep_cell_gb1_deciles.sh
            )
            if $DRY_RUN; then
                echo "[dry-run] ${DEP_CMD[*]}"
            else
                "${DEP_CMD[@]}"
            fi
            ((total++)) || true
        done
    done
done

echo ""
echo "$($DRY_RUN && echo 'Would submit' || echo 'Submitted') ${total} jobs (25 cells x 3 interp methods)."
