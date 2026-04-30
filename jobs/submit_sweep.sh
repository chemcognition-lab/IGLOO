#!/bin/bash
# submit_sweep.sh — submit the full interpolation-path × noise sweep.
#
# Proteins × data_models × interpolation_paths × noise_levels × seeds
#
# Usage:
#   bash jobs/submit_sweep.sh                         # submit everything (seed=42)
#   bash jobs/submit_sweep.sh --dry-run               # print sbatch commands, don't run
#   bash jobs/submit_sweep.sh --protein=GB1           # submit only one protein
#   bash jobs/submit_sweep.sh --seed=0 --seed=1       # multiple seeds (repeat flag)
#   bash jobs/submit_sweep.sh --seed=0                # single seed override
#
# Per-protein SLURM resources:
#   GB1        → H100 GPU, 128 GB RAM, 1.5 h  (2.5 h for nearest_neighbor)
#   T4Lysozyme → H100 GPU, 256 GB RAM, 4 h    (5.5 h for nearest_neighbor)
#   p53        → H100 GPU, 512 GB RAM, 6 h    (10 h  for nearest_neighbor)

set -euo pipefail

DRY_RUN=false
EMAIL="ranamoneim@gmail.com"  # override with --email=other@address.com
PROTEIN_FILTER=""  # override with --protein=GB1
INTERP_FILTER=""   # override with --interp=slerp
SEED_FILTER=""     # override with --seed=42

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --email=*) EMAIL="${arg#--email=}" ;;
        --protein=*) PROTEIN_FILTER="${arg#--protein=}" ;;
        --interp=*) INTERP_FILTER="${arg#--interp=}" ;;
        --seed=*) SEED_FILTER="${arg#--seed=}" ;;
    esac
done

if $DRY_RUN; then
    echo "[dry-run] No jobs will actually be submitted."
fi
if [[ -n "$EMAIL" ]]; then
    echo "[notify] Job completion/failure emails will be sent to: ${EMAIL}"
fi

# ── Sweep axes ────────────────────────────────────────────────────────────────
if [[ -n "$PROTEIN_FILTER" ]]; then
    PROTEINS=("$PROTEIN_FILTER")
else
    PROTEINS=(GB1 T4Lysozyme p53)
fi
DATA_MODELS=(additive epistatic nonlinear)
if [[ -n "$INTERP_FILTER" ]]; then
    INTERP_PATHS=("$INTERP_FILTER")
else
    INTERP_PATHS=(linear slerp nearest_neighbor)
fi
if [[ -n "$SEED_FILTER" ]]; then
    SEEDS=("$SEED_FILTER")
else
    SEEDS=(42)
fi
NOISE_LEVELS=(0.0 0.05 0.1 0.3 0.5 0.8 1.0)

# ── Per-protein resource specs ────────────────────────────────────────────────
declare -A GPU_TYPE=( [GB1]="l40s" [T4Lysozyme]="h100" [p53]="h100" )
declare -A MEM=(      [GB1]="128G" [T4Lysozyme]="256G"  [p53]="512G" )
# Default time limits; nearest_neighbor jobs get longer limits (set per-interp below)
declare -A TIME=(     [GB1]="1:30:00" [T4Lysozyme]="4:00:00" [p53]="6:00:00" )
declare -A TIME_NN=(  [GB1]="2:30:00" [T4Lysozyme]="7:30:00" [p53]="23:59:00" )

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
                for SEED in "${SEEDS[@]}"; do
                ((total++)) || true
                JOB_NAME="${PROTEIN}_${DATA_MODEL}_${INTERP}_n${NOISE}_s${SEED}"

                JOB_TIME="${TIME[$PROTEIN]}"
                if [[ "$INTERP" == "nearest_neighbor" ]]; then
                    JOB_TIME="${TIME_NN[$PROTEIN]}"
                fi

                SBATCH_CMD=(
                    sbatch
                    --job-name="${JOB_NAME}"
                    --output="jobs/logs/${JOB_NAME}_%j.out"
                    --gres="gpu:${GPU_TYPE[$PROTEIN]}:1"
                    --mem="${MEM[$PROTEIN]}"
                    --time="${JOB_TIME}"
                    --export="PROTEIN=${PROTEIN},DATA_MODEL=${DATA_MODEL},INTERP=${INTERP},NOISE=${NOISE},SEED=${SEED}"
                )
                if [[ -n "$EMAIL" ]]; then
                    SBATCH_CMD+=(--mail-type=END,FAIL --mail-user="${EMAIL}")
                fi
                SBATCH_CMD+=(jobs/run_sweep_job.sh)

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
done

echo ""
if $DRY_RUN; then
    echo "Would submit ${total} jobs."
else
    echo "Submitted ${submitted} / ${total} jobs."
fi
