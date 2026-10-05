#!/bin/bash
# One-time maintenance: copy IGLOO_env and IGLOO_env_e1 from /project (inode
# quota exhausted, 500K/500K) to /home (77K/500K, plenty of headroom).
# Does NOT delete the /project originals -- that's a manual follow-up step
# after verifying the copies are complete and functional.
#SBATCH --account=def-mahadeva_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/move_venvs_%j.out

set -u
set -o pipefail
cd /project/6007171/ranaab/IGLOO || exit 1

SRC_A="IGLOO_env"
SRC_B="IGLOO_env_e1"
DST="/home/ranaab"

for V in "$SRC_A" "$SRC_B"; do
  echo "=== rsync $V -> $DST/$V ==="
  rsync -a --checksum "$V"/ "$DST/$V"/
  echo "=== verifying file counts for $V ==="
  n_src=$(find "$V" -type f | wc -l)
  n_dst=$(find "$DST/$V" -type f | wc -l)
  echo "src=$n_src dst=$n_dst"
  if [[ "$n_src" -ne "$n_dst" ]]; then
    echo "MISMATCH for $V -- not touching original, needs investigation" >&2
  else
    echo "OK: $V file counts match"
  fi
done

echo "=== done, /project originals left untouched pending manual review ==="
