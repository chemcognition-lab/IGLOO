#!/bin/bash
# submit_baseline_sweep.sh — submit gradient_x_input + input_difference baseline
# jobs for all GB1 data-model × noise combinations.
#
# Usage:
#   bash jobs/attribution/submit_baseline_sweep.sh                          # submit all GB1 jobs
#   bash jobs/attribution/submit_baseline_sweep.sh --dry-run                # print sbatch commands only
#   bash jobs/attribution/submit_baseline_sweep.sh --protein=T4Lysozyme     # different protein
#   bash jobs/attribution/submit_baseline_sweep.sh --protein=p53            # p53 protein
#   bash jobs/attribution/submit_baseline_sweep.sh --data-model=additive    # one data model only
#   bash jobs/attribution/submit_baseline_sweep.sh --noise=0.0              # one noise level only

set -euo pipefail

DRY_RUN=false
DM_FILTER=""
NOISE_FILTER=""
PROTEIN_FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)        DRY_RUN=true ;;
        --data-model=*)   DM_FILTER="${arg#--data-model=}" ;;
        --noise=*)        NOISE_FILTER="${arg#--noise=}" ;;
        --protein=*)      PROTEIN_FILTER="${arg#--protein=}" ;;
    esac
done

if $DRY_RUN; then
    echo "[dry-run] No jobs will actually be submitted."
fi

# ── Sweep axes ────────────────────────────────────────────────────────────────
if [[ -n "$DM_FILTER" ]]; then
    DATA_MODELS=("$DM_FILTER")
else
    DATA_MODELS=(additive epistatic nonlinear)
fi

if [[ -n "$NOISE_FILTER" ]]; then
    NOISE_LEVELS=("$NOISE_FILTER")
else
    NOISE_LEVELS=(0.0 0.05 0.1 0.3 0.5 0.8 1.0)
fi

PROTEIN="${PROTEIN_FILTER:-GB1}"
RESULTS_ROOT="/scratch/ranaab/IGLOO/results/synthetic"

mkdir -p jobs/logs

total=0
submitted=0

for DM in "${DATA_MODELS[@]}"; do
    for NOISE in "${NOISE_LEVELS[@]}"; do
        ((total++)) || true

        LINEAR_DIR="${RESULTS_ROOT}/${PROTEIN}_${DM}/noise_${NOISE}/linear"

        # Pick the latest config snapshot
        SNAP=$(ls "${LINEAR_DIR}/${PROTEIN}_${DM}_config_snapshot_"*.yaml \
               2>/dev/null | sort | tail -1)

        if [[ -z "$SNAP" ]]; then
            echo "WARNING: no config snapshot found in ${LINEAR_DIR} — skipping."
            continue
        fi

        # Skip if both baseline output dirs already exist and have results CSVs
        GXI_CSV="${RESULTS_ROOT}/${PROTEIN}_${DM}/noise_${NOISE}/gradient_x_input/${PROTEIN}_${DM}_integrated_gradients_results.csv"
        ID_CSV="${RESULTS_ROOT}/${PROTEIN}_${DM}/noise_${NOISE}/input_difference/${PROTEIN}_${DM}_integrated_gradients_results.csv"
        if [[ -f "$GXI_CSV" && -f "$ID_CSV" ]]; then
            echo "SKIP (already done): ${PROTEIN}_${DM} noise_${NOISE}"
            continue
        fi

        JOB_LABEL="${PROTEIN}_${DM}_noise_${NOISE}_baselines"

        SBATCH_CMD=(
            sbatch
            --job-name="${JOB_LABEL}"
            --output="jobs/logs/${JOB_LABEL}_%j.out"
            --export="CONFIG_SNAPSHOT=${SNAP},JOB_LABEL=${JOB_LABEL}"
            jobs/attribution/run_baseline_job.sh
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
    echo "Would submit up to ${total} jobs (already-done ones would be skipped at runtime)."
else
    echo "Submitted ${submitted} / ${total} jobs."
fi
