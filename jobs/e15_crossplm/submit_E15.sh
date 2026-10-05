#!/bin/bash
# submit_E15.sh — submit the full E15 cross-pLM sweep.
#
# E15 PHASE 1 (T4 Lysozyme): 1 protein × 3 landscapes × 1 pLM × 5 seeds = 15 jobs
# E15 PHASE 2 (p53):         1 protein × 3 landscapes × 1 pLM × 5 seeds = 15 jobs (after E12)
#
# Usage:
#   bash jobs/e15_crossplm/submit_E15.sh                    # submit PHASE 1 only
#   bash jobs/e15_crossplm/submit_E15.sh --phase=2          # submit PHASE 2 only (requires E12)
#   bash jobs/e15_crossplm/submit_E15.sh --phase=both       # submit both phases
#   bash jobs/e15_crossplm/submit_E15.sh --dry-run          # print sbatch commands, don't run
#   bash jobs/e15_crossplm/submit_E15.sh --protein=p53      # submit only p53 jobs
#
# Per-protein SLURM resources:
#   T4 Lysozyme → L40s GPU, 64 GB RAM, 6 h
#   p53         → L40s GPU, 128 GB RAM, 10 h

set -euo pipefail

DRY_RUN=false
EMAIL="ranamoneim@gmail.com"  # override with --email=other@address.com
PHASE="1"  # phase 1 or 2 or both
PROTEIN_FILTER=""  # empty = all proteins for that phase

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --email=*) EMAIL="${arg#--email=}" ;;
        --phase=*) PHASE="${arg#--phase=}" ;;
        --protein=*) PROTEIN_FILTER="${arg#--protein=}" ;;
    esac
done

if $DRY_RUN; then
    echo "[dry-run] No jobs will actually be submitted."
fi
if [[ -n "$EMAIL" ]]; then
    echo "[notify] Job completion/failure emails will be sent to: ${EMAIL}"
fi

# ── Sweep axes ────────────────────────────────────────────────────────────────
# PHASE 1: T4 Lysozyme only
# PHASE 2: p53 only (requires E12 to be complete)

if [ "$PHASE" = "1" ]; then
    PROTEINS_TO_SUBMIT=(T4_lysozyme)
    PHASE_NAME="PHASE 1"
elif [ "$PHASE" = "2" ]; then
    PROTEINS_TO_SUBMIT=(p53)
    PHASE_NAME="PHASE 2 (requires E12)"
    # Check E12 completion
    if [ ! -f "results/revision/E12_p53/diagnostic.md" ]; then
        echo "ERROR: E12 (p53 diagnostic) not found at results/revision/E12_p53/diagnostic.md"
        echo "PHASE 2 requires E12 to be complete before queuing."
        echo "Skipping PHASE 2 submission."
        exit 1
    fi
elif [ "$PHASE" = "both" ]; then
    PROTEINS_TO_SUBMIT=(T4_lysozyme p53)
    PHASE_NAME="PHASE 1 + PHASE 2"
else
    echo "ERROR: Invalid --phase=$PHASE (use 1, 2, or both)"
    exit 1
fi

LANDSCAPES=(additive epistatic nonlinear)
PLMS=(esmc_300m)
SEEDS=(1 2 3 4 5)

# ── Per-protein resource specs ────────────────────────────────────────────────
declare -A GPU_TYPE=(
    [T4_lysozyme]="l40s"
    [p53]="l40s"
)
declare -A MEM=(
    [T4_lysozyme]="64G"
    [p53]="128G"
)
declare -A TIME=(
    [T4_lysozyme]="06:00:00"
    [p53]="10:00:00"
)

# ── Submit ────────────────────────────────────────────────────────────────────
total=0
submitted=0
mkdir -p jobs/logs

for PROTEIN in "${PROTEINS_TO_SUBMIT[@]}"; do
    # Determine which phase(s) to run for this protein
    if [ "$PHASE" = "1" ] || [ "$PHASE" = "both" ]; then
        PHASES_TO_RUN=(phase1)
    else
        PHASES_TO_RUN=()
    fi

    if [ "$PHASE" = "2" ] || ([ "$PHASE" = "both" ] && [ "$PROTEIN" = "p53" ]); then
        PHASES_TO_RUN+=(phase2)
    fi

    for PHASE_LABEL in "${PHASES_TO_RUN[@]}"; do
        for LANDSCAPE in "${LANDSCAPES[@]}"; do
            for PLM in "${PLMS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    ((total++)) || true
                    JOB_NAME="E15_${PHASE_LABEL}_${PROTEIN}_${LANDSCAPE}_${PLM}_s${SEED}"

                    SBATCH_CMD=(
                        sbatch
                        --job-name="${JOB_NAME}"
                        --output="jobs/logs/${JOB_NAME}_%j.out"
                        --gres="gpu:${GPU_TYPE[$PROTEIN]}:1"
                        --mem="${MEM[$PROTEIN]}"
                        --time="${TIME[$PROTEIN]}"
                        --export="PROTEIN=${PROTEIN},LANDSCAPE=${LANDSCAPE},PLM=${PLM},SEED=${SEED},PHASE=${PHASE_LABEL}"
                    )
                    if [[ -n "$EMAIL" ]]; then
                        SBATCH_CMD+=(--mail-type=END,FAIL --mail-user="${EMAIL}")
                    fi
                    SBATCH_CMD+=(jobs/e15_crossplm/run_E15_job.sh)

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
    echo "Would submit ${total} jobs for E15 ${PHASE_NAME}."
else
    echo "Submitted ${submitted} / ${total} jobs for E15 ${PHASE_NAME}."
fi
echo ""
echo "Monitor jobs with: squeue -u $USER | grep E15"
echo "Check logs with:   ls -la jobs/logs/ | grep E15"
