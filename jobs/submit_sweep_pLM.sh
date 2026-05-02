#!/bin/bash
# submit_sweep_pLM.sh — submit the pLM comparison sweep.
#
# Proteins × data_models × pLMs × interpolation_paths × seeds
# Noise is fixed at 0.0 (clean data) — use submit_sweep.sh for noise sweeps.
#
# Results land in:
#   results/synthetic/plm_{PLM}/{PROTEIN}_{DATA_MODEL}/{INTERP}/
#
# Usage:
#   bash jobs/submit_sweep_pLM.sh                         # submit everything (seed=42)
#   bash jobs/submit_sweep_pLM.sh --dry-run               # print commands, don't run
#   bash jobs/submit_sweep_pLM.sh --protein=GB1           # one protein only
#   bash jobs/submit_sweep_pLM.sh --plm=ankh_base         # one pLM only
#   bash jobs/submit_sweep_pLM.sh --data-model=nonlinear  # one data model only
#   bash jobs/submit_sweep_pLM.sh --interp=linear         # one path only
#   bash jobs/submit_sweep_pLM.sh --seed=0 --seed=1       # multiple seeds
#
# Per-protein SLURM resources (same tiers as submit_sweep.sh):
#   GB1        → H100 GPU, 128 GB RAM, 2 h   (3 h for nearest_neighbor)
#   T4Lysozyme → H100 GPU, 256 GB RAM, 4 h   (7.5 h for nearest_neighbor)
#   p53        → H100 GPU, 512 GB RAM, 8 h   (23:59 for nearest_neighbor)
# Large pLMs (ankh_large, e1_600m) get +50 % wall time automatically.

set -euo pipefail

DRY_RUN=false
EMAIL="ranamoneim@gmail.com"
PROTEIN_FILTER=""
DM_FILTER=""
PLM_FILTER=""
INTERP_FILTER=""
SEED_FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --email=*)      EMAIL="${arg#--email=}" ;;
        --protein=*)    PROTEIN_FILTER="${arg#--protein=}" ;;
        --data-model=*) DM_FILTER="${arg#--data-model=}" ;;
        --plm=*)        PLM_FILTER="${arg#--plm=}" ;;
        --interp=*)     INTERP_FILTER="${arg#--interp=}" ;;
        --seed=*)       SEED_FILTER="${arg#--seed=}" ;;
    esac
done

if $DRY_RUN; then
    echo "[dry-run] No jobs will actually be submitted."
fi
if [[ -n "$EMAIL" ]]; then
    echo "[notify] Completion/failure emails → ${EMAIL}"
fi

# ── Sweep axes ────────────────────────────────────────────────────────────────
if [[ -n "$PROTEIN_FILTER" ]]; then
    PROTEINS=("$PROTEIN_FILTER")
else
    PROTEINS=(GB1 T4Lysozyme p53)
fi

if [[ -n "$DM_FILTER" ]]; then
    DATA_MODELS=("$DM_FILTER")
else
    DATA_MODELS=(additive epistatic nonlinear)
fi

if [[ -n "$PLM_FILTER" ]]; then
    PLMS=("$PLM_FILTER")
else
    PLMS=(esmc_300m ankh_base ankh_large e1_150m e1_300m e1_600m)
fi

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

# ── Per-protein resource specs ────────────────────────────────────────────────
declare -A GPU_TYPE=( [GB1]="l40s"  [T4Lysozyme]="h100"  [p53]="h100" )
declare -A MEM=(      [GB1]="192G"  [T4Lysozyme]="256G"  [p53]="512G" )
declare -A TIME=(     [GB1]="2:00:00" [T4Lysozyme]="4:00:00" [p53]="8:00:00" )
declare -A TIME_NN=(  [GB1]="3:00:00" [T4Lysozyme]="7:30:00" [p53]="23:59:00" )

# Large pLMs (need longer embedding / forward pass time): multiply limit by 1.5
LARGE_PLMS=(ankh_large e1_600m)

_is_large_plm() {
    local plm="$1"
    for lp in "${LARGE_PLMS[@]}"; do
        [[ "$plm" == "$lp" ]] && return 0
    done
    return 1
}

# Add N minutes to an H:MM:SS or HH:MM:SS string. Returns new time string.
_add_half_time() {
    local t="$1"
    IFS=: read -r h m s <<< "$t"
    local total=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    local extra=$(( total / 2 ))       # +50 %
    local new=$(( total + extra ))
    if (( new >= 86400 )); then new=86399; fi  # cap at 23:59:59
    printf "%02d:%02d:%02d" $(( new/3600 )) $(( (new%3600)/60 )) $(( new%60 ))
}

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
        for PLM in "${PLMS[@]}"; do
            for INTERP in "${INTERP_PATHS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                ((total++)) || true
                JOB_NAME="${PROTEIN}_${DATA_MODEL}_${PLM}_${INTERP}_s${SEED}"

                JOB_TIME="${TIME[$PROTEIN]}"
                if [[ "$INTERP" == "nearest_neighbor" ]]; then
                    JOB_TIME="${TIME_NN[$PROTEIN]}"
                fi
                if _is_large_plm "$PLM"; then
                    JOB_TIME="$(_add_half_time "$JOB_TIME")"
                fi

                SBATCH_CMD=(
                    sbatch
                    --job-name="${JOB_NAME}"
                    --output="jobs/logs/${JOB_NAME}_%j.out"
                    --gres="gpu:${GPU_TYPE[$PROTEIN]}:1"
                    --mem="${MEM[$PROTEIN]}"
                    --time="${JOB_TIME}"
                    --export="PROTEIN=${PROTEIN},DATA_MODEL=${DATA_MODEL},PLM=${PLM},INTERP=${INTERP},SEED=${SEED}"
                )
                if [[ -n "$EMAIL" ]]; then
                    SBATCH_CMD+=(--mail-type=END,FAIL --mail-user="${EMAIL}")
                fi
                SBATCH_CMD+=(jobs/run_sweep_pLM_job.sh)

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
