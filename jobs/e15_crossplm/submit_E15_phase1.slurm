#!/bin/bash
#SBATCH --job-name=E15_phase1_T4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=logs/E15_phase1_%j.log
#SBATCH --error=logs/E15_phase1_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ranaab@example.com

# E15 PHASE 1 — T4 Lysozyme Cross-pLM Comparison
# Long-pole experiment: run in parallel with E1, E9, E12

set -e  # Exit on error

echo "=========================================="
echo "E15 PHASE 1 — T4 Lysozyme Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Setup environment
cd /project/6111759/ranaab/IGLOO

# Activate conda environment if needed
# source ~/.bashrc
# conda activate IGLOO_env

# Create logs directory
mkdir -p logs

# Set reproducibility
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Run E15 PHASE 1
echo "Running E15 PHASE 1..."
python run_E15_phase1.py \
  --output-dir results/revision/E15_crosspLM \
  2>&1 | tee -a logs/E15_phase1_${SLURM_JOB_ID}.log

PHASE1_EXIT=$?

if [ $PHASE1_EXIT -eq 0 ]; then
  echo ""
  echo "✓ E15 PHASE 1 COMPLETED SUCCESSFULLY"
  echo "Results saved to: results/revision/E15_crosspLM/"
  echo "Check: results/revision/E15_crosspLM/metrics_long.csv"
else
  echo ""
  echo "✗ E15 PHASE 1 FAILED (exit code: $PHASE1_EXIT)"
  exit $PHASE1_EXIT
fi

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
