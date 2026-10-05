#!/bin/bash
# submit_flip2_train_eval.sh — submit train+eval jobs for all FLIP2 alpha-amylase splits.
#
# Usage:
#   bash jobs/real_proteins/submit_flip2_train_eval.sh              # submit all 4 splits
#   bash jobs/real_proteins/submit_flip2_train_eval.sh --dry-run    # print sbatch commands only
#   bash jobs/real_proteins/submit_flip2_train_eval.sh --split=alpha_amylase__mutation__by_mutation

set -euo pipefail

DRY_RUN=false
SPLIT_FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=true ;;
        --split=*)   SPLIT_FILTER="${arg#--split=}" ;;
    esac
done

if $DRY_RUN; then
    echo "[dry-run] No jobs will actually be submitted."
fi

SPLITS=(
    alpha_amylase__mutation__by_mutation
    alpha_amylase__number__one_to_many
    alpha_amylase__position__close_to_far
    alpha_amylase__position__far_to_close
)

mkdir -p jobs/logs

submitted=0
for SPLIT in "${SPLITS[@]}"; do
    if [[ -n "$SPLIT_FILTER" && "$SPLIT" != "$SPLIT_FILTER" ]]; then
        continue
    fi

    CONFIG="$(pwd)/configs/FLIP2/${SPLIT}.yaml"
    if [[ ! -f "$CONFIG" ]]; then
        echo "WARNING: config not found, skipping — ${CONFIG}"
        continue
    fi

    JOB_NAME="flip2_${SPLIT}"
    LOG_FILE="jobs/logs/${JOB_NAME}_%j.out"

    CMD=(
        sbatch
        --job-name="${JOB_NAME}"
        --output="${LOG_FILE}"
        --export="CONFIG=${CONFIG}"
        jobs/real_proteins/run_flip2_train_eval.sh
    )

    if $DRY_RUN; then
        echo "[dry-run] ${CMD[*]}"
    else
        "${CMD[@]}"
        echo "Submitted: ${JOB_NAME}"
        (( submitted++ )) || true
    fi
done

echo ""
if ! $DRY_RUN; then
    echo "Total jobs submitted: ${submitted}"
fi
