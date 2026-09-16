#!/bin/bash
# submit_seedsweep.sh — multi-seed noise sweep for the
# "attribution quality vs model performance, coloured by noise" figure.
#
#   proteins × data_models × noise_levels × seeds
#   default: 2 × 3 × 7 × 5 = 210 jobs   (GB1 + T4Lysozyme)
#
# Usage:
#   bash jobs/submit_seedsweep.sh --dry-run
#   bash jobs/submit_seedsweep.sh --proteins=GB1
#   bash jobs/submit_seedsweep.sh --seeds=1,2,3 --noise=0.0,0.3,1.0
#   bash jobs/submit_seedsweep.sh                # submit everything

set -euo pipefail

DRY_RUN=false
EMAIL="ranamoneim@gmail.com"
PROTEIN_FILTER=""
SEED_FILTER=""
NOISE_FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --email=*) EMAIL="${arg#--email=}" ;;
        --proteins=*) PROTEIN_FILTER="${arg#--proteins=}" ;;
        --seeds=*) SEED_FILTER="${arg#--seeds=}" ;;
        --noise=*) NOISE_FILTER="${arg#--noise=}" ;;
    esac
done

$DRY_RUN && echo "[dry-run] No jobs will actually be submitted."

ALL_PROTEINS=(GB1 T4Lysozyme)
if [[ -n "$PROTEIN_FILTER" ]]; then IFS=',' read -ra PROTEINS <<< "$PROTEIN_FILTER"
else PROTEINS=("${ALL_PROTEINS[@]}"); fi

DATA_MODELS=(additive epistatic nonlinear)

if [[ -n "$NOISE_FILTER" ]]; then IFS=',' read -ra NOISE_LEVELS <<< "$NOISE_FILTER"
else NOISE_LEVELS=(0.0 0.05 0.1 0.3 0.5 0.8 1.0); fi

if [[ -n "$SEED_FILTER" ]]; then IFS=',' read -ra SEEDS <<< "$SEED_FILTER"
else SEEDS=(1 2 3 4 5); fi

declare -A GPU_TYPE=( [GB1]="l40s" [T4Lysozyme]="l40s" )
declare -A MEM=(      [GB1]="128G" [T4Lysozyme]="256G" )
declare -A TIME=(     [GB1]="3:00:00" [T4Lysozyme]="8:00:00" )

total=0; submitted=0
mkdir -p jobs/logs

for PROTEIN in "${PROTEINS[@]}"; do
    for DATA_MODEL in "${DATA_MODELS[@]}"; do
        CONFIG="configs/synthetic/${PROTEIN}_${DATA_MODEL}_config.yaml"
        [[ -f "$CONFIG" ]] || { echo "WARNING: missing $CONFIG — skipping"; continue; }
        for NOISE in "${NOISE_LEVELS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                ((total++)) || true
                JOB_NAME="ss_${PROTEIN}_${DATA_MODEL}_n${NOISE}_s${SEED}"
                SBATCH_CMD=(
                    sbatch
                    --job-name="${JOB_NAME}"
                    --output="jobs/logs/${JOB_NAME}_%j.out"
                    --gres="gpu:${GPU_TYPE[$PROTEIN]}:1"
                    --mem="${MEM[$PROTEIN]}"
                    --time="${TIME[$PROTEIN]}"
                    --export="PROTEIN=${PROTEIN},DATA_MODEL=${DATA_MODEL},NOISE=${NOISE},SEED=${SEED}"
                )
                [[ -n "$EMAIL" ]] && SBATCH_CMD+=(--mail-type=FAIL --mail-user="${EMAIL}")
                SBATCH_CMD+=(jobs/run_seedsweep_job.sh)

                if $DRY_RUN; then echo "[dry-run] ${SBATCH_CMD[*]}"
                else "${SBATCH_CMD[@]}"; ((submitted++)) || true
                fi
            done
        done
    done
done

echo ""
$DRY_RUN && echo "Would submit ${total} jobs." || echo "Submitted ${submitted} / ${total} jobs."
