#!/bin/bash
#SBATCH --job-name=E15_phase2_p53
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=logs/E15_phase2_%j.log
#SBATCH --error=logs/E15_phase2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ranaab@example.com
#SBATCH --dependency=singleton  # Wait for other E15 jobs to complete

# E15 PHASE 2 — p53 Cross-pLM Comparison
# IMPORTANT: Run this AFTER E12 (p53 diagnostic) completes
# This job automatically waits for E12 results before starting

set -e  # Exit on error

echo "=========================================="
echo "E15 PHASE 2 — p53 Job"
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

# Check E12 status
echo "Checking E12 (p53 diagnostic) status..."
if [ -f "results/revision/E12_p53/diagnostic.md" ]; then
  echo "✓ E12 diagnostic found"
  echo "E12 path: results/revision/E12_p53/diagnostic.md"
else
  echo "⚠ E12 diagnostic not found yet"
  echo "Waiting for E12 to complete..."
  sleep 300  # Wait 5 minutes and retry
  if [ ! -f "results/revision/E12_p53/diagnostic.md" ]; then
    echo "✗ E12 still not complete. Aborting PHASE 2."
    exit 1
  fi
fi
echo ""

# Run E15 PHASE 2
echo "Running E15 PHASE 2 (p53)..."
python run_E15_phase2.py \
  --output-dir results/revision/E15_crosspLM \
  2>&1 | tee -a logs/E15_phase2_${SLURM_JOB_ID}.log

PHASE2_EXIT=$?

if [ $PHASE2_EXIT -eq 0 ]; then
  echo ""
  echo "✓ E15 PHASE 2 COMPLETED SUCCESSFULLY"
  echo "Results saved to: results/revision/E15_crosspLM/"
  echo "Check: results/revision/E15_crosspLM/metrics_long.csv"
  echo ""
  echo "Next steps:"
  echo "  1. Extend E1/E3 results to PHASE 2 runs"
  echo "  2. Merge PHASE 1 + PHASE 2 results"
  echo "  3. Regenerate Table 3 (n=9 proteins)"
  echo "  4. Run Friedman/Kendall's W tests"
else
  echo ""
  echo "✗ E15 PHASE 2 FAILED (exit code: $PHASE2_EXIT)"
  exit $PHASE2_EXIT
fi

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
