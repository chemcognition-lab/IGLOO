#!/bin/bash
# submit_baseline_pLM.sh — submit gradient_x_input + input_difference baseline
# jobs for the pLM comparison sweep results.
#
# Finds config snapshots from the linear results dir for each PLM × data-model
# combination and runs baselines using the already-trained model.
#
# Usage:
#   bash jobs/submit_baseline_pLM.sh                      # submit all
#   bash jobs/submit_baseline_pLM.sh --dry-run             # print only
#   bash jobs/submit_baseline_pLM.sh --plm=ankh_base       # one PLM
#   bash jobs/submit_baseline_pLM.sh --data-model=additive # one data model

set -euo pipefail

DRY_RUN=false
PLM_FILTER=""
DM_FILTER=""
PROTEIN="${PROTEIN:-GB1}"

for arg in "$@"; do
    case "$arg" in
        --dry-run)        DRY_RUN=true ;;
        --plm=*)          PLM_FILTER="${arg#--plm=}" ;;
        --data-model=*)   DM_FILTER="${arg#--data-model=}" ;;
        --protein=*)      PROTEIN="${arg#--protein=}" ;;
    esac
done

if $DRY_RUN; then
    echo "[dry-run] No jobs will actually be submitted."
fi

RESULTS_ROOT="/scratch/ranaab/IGLOO/results/synthetic"

if [[ -n "$PLM_FILTER" ]]; then
    PLMS=("$PLM_FILTER")
else
    PLMS=(ankh_base e1_150m)
fi

if [[ -n "$DM_FILTER" ]]; then
    DATA_MODELS=("$DM_FILTER")
else
    DATA_MODELS=(additive epistatic nonlinear)
fi

mkdir -p jobs/logs

total=0
submitted=0

for PLM in "${PLMS[@]}"; do
    for DM in "${DATA_MODELS[@]}"; do
        ((total++)) || true

        LINEAR_DIR="${RESULTS_ROOT}/plm_${PLM}/${PROTEIN}_${DM}/linear"

        # Pick the latest config snapshot
        SNAP=$(ls "${LINEAR_DIR}/${PROTEIN}_${DM}_config_snapshot_"*.yaml \
               2>/dev/null | sort | tail -1)

        if [[ -z "$SNAP" ]]; then
            echo "WARNING: no config snapshot in ${LINEAR_DIR} — skipping."
            continue
        fi

        # Skip if both outputs already exist
        GXI_CSV="${LINEAR_DIR}/gradient_x_input/${PROTEIN}_${DM}_integrated_gradients_results.csv"
        ID_CSV="${LINEAR_DIR}/input_difference/${PROTEIN}_${DM}_integrated_gradients_results.csv"
        if [[ -f "$GXI_CSV" && -f "$ID_CSV" ]]; then
            echo "SKIP (already done): plm_${PLM} ${PROTEIN}_${DM}"
            continue
        fi

        JOB_LABEL="${PROTEIN}_${DM}_plm_${PLM}_baselines"

        SBATCH_CMD=(
            sbatch
            --job-name="${JOB_LABEL}"
            --output="jobs/logs/${JOB_LABEL}_%j.out"
            --export="CONFIG_SNAPSHOT=${SNAP},JOB_LABEL=${JOB_LABEL}"
            jobs/run_baseline_job.sh
        )

        if $DRY_RUN; then
            echo "[dry-run] ${SBATCH_CMD[*]}"
        else
            "${SBATCH_CMD[@]}"
            echo "Submitted: ${JOB_LABEL}"
            ((submitted++)) || true
        fi
    done
done

echo ""
if $DRY_RUN; then
    echo "Would submit up to ${total} jobs."
else
    echo "Submitted ${submitted} / ${total} jobs."
fi
