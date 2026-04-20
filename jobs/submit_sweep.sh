#!/bin/bash
# submit_sweep.sh — submit the full interpolation-path × noise sweep.
#
# Proteins × data_models × interpolation_paths × noise_levels = 189 jobs
#
# Usage:
#   bash jobs/submit_sweep.sh                  # submit everything
#   bash jobs/submit_sweep.sh --dry-run        # print sbatch commands, don't run
#
# Per-protein SLURM resources:
#   GB1        → L40s GPU, 128 GB RAM, 3 h
#   T4Lysozyme → L40s GPU, 256 GB RAM, 8 h
#   p53        → H100 GPU, 512 GB RAM, 12 h

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] No jobs will actually be submitted."
fi

# ── Sweep axes ────────────────────────────────────────────────────────────────
PROTEINS=(GB1 T4Lysozyme p53)
DATA_MODELS=(additive epistatic nonlinear)
INTERP_PATHS=(linear slerp nearest_neighbor)
NOISE_LEVELS=(0.0 0.05 0.1 0.3 0.5 0.8 1.0)

# ── Per-protein resource specs ────────────────────────────────────────────────
declare -A GPU_TYPE=( [GB1]="l40s" [T4Lysozyme]="l40s" [p53]="h100" )
declare -A MEM=(      [GB1]="128G" [T4Lysozyme]="256G"  [p53]="512G" )
declare -A TIME=(     [GB1]="3:00:00" [T4Lysozyme]="8:00:00" [p53]="12:00:00" )

# ── Submit ────────────────────────────────────────────────────────────────────
total=0
submitted=0
mkdir -p jobs/logs

for PROTEIN in "${PROTEINS[@]}"; do
    for DATA_MODEL in "${DATA_MODELS[@]}"; do
        CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
        if [[ ! -f "$CONFIG" ]]; then
            echo "WARNING: config not found, skipping — ${CONFIG}"
            continue
        fi
        for INTERP in "${INTERP_PATHS[@]}"; do
            for NOISE in "${NOISE_LEVELS[@]}"; do
                ((total++)) || true
                JOB_NAME="${PROTEIN}_${DATA_MODEL}_${INTERP}_n${NOISE}"

                SBATCH_CMD=(
                    sbatch
                    --job-name="${JOB_NAME}"
                    --output="jobs/logs/${JOB_NAME}_%j.out"
                    --gres="gpu:${GPU_TYPE[$PROTEIN]}:1"
                    --mem="${MEM[$PROTEIN]}"
                    --time="${TIME[$PROTEIN]}"
                    --export="PROTEIN=${PROTEIN},DATA_MODEL=${DATA_MODEL},INTERP=${INTERP},NOISE=${NOISE}"
                    jobs/run_sweep_job.sh
                )

                if $DRY_RUN; then
                    echo "[dry-run] ${SBATCH_CMD[*]}"
                else
                    "${SBATCH_CMD[@]}"
                    ((submitted++)) || true
                fi
            done
        done
    done
done

echo ""
if $DRY_RUN; then
    echo "Would submit ${total} jobs."
else
    echo "Submitted ${submitted} / ${total} jobs."
fi
