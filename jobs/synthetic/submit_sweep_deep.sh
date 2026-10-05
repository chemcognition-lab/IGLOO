#!/bin/bash
# submit_sweep_deep.sh — noise sweep for the GB1 "deep" mutation landscape.
#
# Sweeps: additive_deep × epistatic_deep × nonlinear_deep
#       × interpolation_paths × noise_levels × seeds
#
# Reuses run_sweep_job.sh by setting DATA_MODEL=additive_deep etc., which
# resolves to the correct config and results paths automatically:
#   config  : configs/synthetic/GB1_{DATA_MODEL}_config.yaml
#   results : results/synthetic/GB1_{DATA_MODEL}/noise_{NOISE}/{INTERP}/
#
# Prerequisite: generate the deep datasets first:
#   python scripts/data/generate_synthetic_datasets.py \
#       --proteins GB1 --models additive epistatic nonlinear \
#       --mutation-profile deep --output-dir data/synthetic/deep
#
# Usage:
#   bash jobs/synthetic/submit_sweep_deep.sh                   # submit everything (seed=42)
#   bash jobs/synthetic/submit_sweep_deep.sh --dry-run         # print commands, don't run
#   bash jobs/synthetic/submit_sweep_deep.sh --interp=linear   # one interp path only
#   bash jobs/synthetic/submit_sweep_deep.sh --seed=0          # single seed override

set -euo pipefail

DRY_RUN=false
EMAIL="ranamoneim@gmail.com"
INTERP_FILTER=""
SEED_FILTER=""
MODEL_FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=true ;;
        --email=*)     EMAIL="${arg#--email=}" ;;
        --interp=*)    INTERP_FILTER="${arg#--interp=}" ;;
        --seed=*)      SEED_FILTER="${arg#--seed=}" ;;
        --model=*)     MODEL_FILTER="${arg#--model=}" ;;
    esac
done

if $DRY_RUN; then
    echo "[dry-run] No jobs will actually be submitted."
fi
if [[ -n "$EMAIL" ]]; then
    echo "[notify] Completion/failure emails → ${EMAIL}"
fi

# ── Sweep axes ────────────────────────────────────────────────────────────────
PROTEINS=(GB1)   # deep datasets only exist for GB1 right now

if [[ -n "$MODEL_FILTER" ]]; then
    DATA_MODELS=("${MODEL_FILTER}_deep")
else
    DATA_MODELS=(additive_deep epistatic_deep nonlinear_deep)
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

NOISE_LEVELS=(0.0 0.05 0.1 0.3 0.5 0.8 1.0)

# ── Per-protein resource specs (GB1 deep → l40s; small protein) ───────────────
GPU_TYPE="l40s"
MEM="192G"
TIME_DEFAULT="1:30:00"
TIME_NN="2:30:00"

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

                JOB_TIME="${TIME_DEFAULT}"
                if [[ "$INTERP" == "nearest_neighbor" ]]; then
                    JOB_TIME="${TIME_NN}"
                fi

                SBATCH_CMD=(
                    sbatch
                    --job-name="${JOB_NAME}"
                    --output="jobs/logs/${JOB_NAME}_%j.out"
                    --gres="gpu:${GPU_TYPE}:1"
                    --mem="${MEM}"
                    --time="${JOB_TIME}"
                    --export="PROTEIN=${PROTEIN},DATA_MODEL=${DATA_MODEL},INTERP=${INTERP},NOISE=${NOISE},SEED=${SEED}"
                )
                if [[ -n "$EMAIL" ]]; then
                    SBATCH_CMD+=(--mail-type=END,FAIL --mail-user="${EMAIL}")
                fi
                SBATCH_CMD+=(jobs/synthetic/run_sweep_job.sh)

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
