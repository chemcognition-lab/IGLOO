#!/bin/bash
# Zip results/synthetic/factor_ablation (27GB, 3402 files) and, only if the
# archive verifies cleanly, delete the original directory to free the inode
# count (a single .zip only costs 1 inode vs. 3402 for the unpacked files).
#
#SBATCH --account=def-mahadeva_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/zip_factor_ablation_%j.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

SRC="results/synthetic/factor_ablation"
ZIP="results/synthetic/factor_ablation.zip"

echo "=== zipping $SRC -> $ZIP ==="
zip -rq "$ZIP" "$SRC"
echo "zip exit code: $?"

echo "=== verifying archive integrity ==="
if unzip -tq "$ZIP" > /tmp/zip_test_$SLURM_JOB_ID.log 2>&1; then
    echo "OK: archive verified clean"
    echo "=== deleting original directory ==="
    rm -rf "$SRC"
    echo "done. remaining: $(ls -d $SRC 2>&1)"
else
    echo "FAILED verification -- NOT deleting original. See test log:"
    cat /tmp/zip_test_$SLURM_JOB_ID.log
    exit 1
fi

echo "=== final sizes ==="
du -sh "$ZIP"
