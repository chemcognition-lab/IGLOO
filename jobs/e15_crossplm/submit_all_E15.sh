#!/bin/bash
# E15 Master Job Submission Script
# Submits all E15 jobs to SLURM cluster

set -e

PROJECT_ROOT="/project/6111759/ranaab/IGLOO"
JOBS_DIR="$PROJECT_ROOT/jobs"

echo "=========================================="
echo "E15 Job Submission Manager"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -d "$PROJECT_ROOT/results" ]; then
  echo "✗ Error: Not in IGLOO project directory"
  echo "  Expected: $PROJECT_ROOT"
  exit 1
fi

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Parse arguments
ACTION="${1:-both}"
PHASE1_ONLY=false
PHASE2_ONLY=false
DRY_RUN=false

case "$ACTION" in
  phase1)
    PHASE1_ONLY=true
    echo "Mode: PHASE 1 only (T4 lysozyme)"
    ;;
  phase2)
    PHASE2_ONLY=true
    echo "Mode: PHASE 2 only (p53, after E12)"
    ;;
  both)
    echo "Mode: Both phases (sequential)"
    ;;
  dry-run)
    DRY_RUN=true
    echo "Mode: Dry run (print commands, don't submit)"
    ;;
  *)
    echo "Usage: $0 [phase1|phase2|both|dry-run]"
    echo ""
    echo "Options:"
    echo "  phase1   - Submit only PHASE 1 (T4 lysozyme)"
    echo "  phase2   - Submit only PHASE 2 (p53, requires E12)"
    echo "  both     - Submit both phases sequentially (default)"
    echo "  dry-run  - Print commands without submitting"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo ""

# Submit PHASE 1
if [ "$PHASE1_ONLY" = true ] || [ "$ACTION" = "both" ]; then
  echo "Submitting E15 PHASE 1 (T4 lysozyme)..."

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit: sbatch $JOBS_DIR/submit_E15_phase1.slurm"
  else
    PHASE1_JOB=$(sbatch "$JOBS_DIR/submit_E15_phase1.slurm" | awk '{print $NF}')
    echo "✓ PHASE 1 submitted with Job ID: $PHASE1_JOB"
    echo "  Command: sbatch $JOBS_DIR/submit_E15_phase1.slurm"
    echo "  Monitor with: squeue -j $PHASE1_JOB"
    echo "  Logs: tail -f $PROJECT_ROOT/logs/E15_phase1_*.log"
  fi
  echo ""
fi

# Submit PHASE 2 (with dependency on E12)
if [ "$PHASE2_ONLY" = true ] || [ "$ACTION" = "both" ]; then
  echo "Submitting E15 PHASE 2 (p53)..."
  echo "⚠ PHASE 2 requires E12 (p53 diagnostic) to be complete"
  echo "  PHASE 2 will wait for E12 results before starting"
  echo ""

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit: sbatch $JOBS_DIR/submit_E15_phase2.slurm"
  else
    PHASE2_JOB=$(sbatch "$JOBS_DIR/submit_E15_phase2.slurm" | awk '{print $NF}')
    echo "✓ PHASE 2 submitted with Job ID: $PHASE2_JOB"
    echo "  Command: sbatch $JOBS_DIR/submit_E15_phase2.slurm"
    echo "  Monitor with: squeue -j $PHASE2_JOB"
    echo "  Logs: tail -f $PROJECT_ROOT/logs/E15_phase2_*.log"
  fi
  echo ""
fi

echo "=========================================="
echo "Submission complete!"
echo ""
echo "Next steps:"
echo "  1. Monitor job status: squeue -u $USER | grep E15"
echo "  2. Check logs: ls -la $PROJECT_ROOT/logs/"
echo "  3. View results: $PROJECT_ROOT/results/revision/E15_crosspLM/"
echo ""
echo "See E15_EXECUTION_PLAN.md for detailed instructions"
echo "=========================================="
