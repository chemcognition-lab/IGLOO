#!/bin/bash
# Generic SLURM job for one cell of the E15 cross-pLM sweep.
#
# Required env vars (set by submit_E15.sh via --export):
#   PROTEIN      e.g. T4_lysozyme
#   LANDSCAPE    e.g. additive
#   PLM          e.g. esmc_300m
#   SEED         e.g. 1
#   PHASE        e.g. phase1 or phase2
#
# Resource directives are intentionally absent here — submit_E15.sh passes
# them via `sbatch --mem=... --time=... --gres=...` on the command line.

#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=8
# --output is set by submit_E15.sh on the sbatch command line

cd /project/6111759/ranaab/IGLOO || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

# Set Python path for imports
export PYTHONPATH="/project/6111759/ranaab/IGLOO:${PYTHONPATH}"

# Derived paths
RESULTS_DIR="results/revision/E15_crosspLM/${PROTEIN}/${PLM}"

echo "=========================================="
echo "E15 JOB: ${PROTEIN} / ${LANDSCAPE} / ${PLM} / seed=${SEED} (${PHASE})"
echo "  results_dir : ${RESULTS_DIR}"
echo "=========================================="

mkdir -p "${RESULTS_DIR}"

# Run E15 experiment
if [ "${PHASE}" = "phase1" ]; then
    python scripts/e15_crossplm/run_E15_phase1.py \
        --plms "${PLM}" \
        --landscapes "${LANDSCAPE}" \
        --seeds "${SEED}" \
        --output-dir results/revision/E15_crosspLM
elif [ "${PHASE}" = "phase2" ]; then
    python scripts/e15_crossplm/run_E15_phase2.py \
        --plms "${PLM}" \
        --landscapes "${LANDSCAPE}" \
        --seeds "${SEED}" \
        --output-dir results/revision/E15_crosspLM
else
    echo "ERROR: Invalid PHASE=${PHASE}"
    exit 1
fi

echo "=========================================="
echo "Job complete"
echo "=========================================="
