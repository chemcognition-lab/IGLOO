#!/bin/bash
# Regenerate the ISM outputs that were corrupted by the noise-tagging bug in
# scripts/pipeline/ism_attribution.py (it read training.noise.label_noise_std, which is
# always 0 under the v2 blend-noise mode -- every noise level silently
# overwrote the same output file). The script is now fixed (reads
# label_noise_level when label_noise_mode=blend). This re-runs ONLY the two
# affected task kinds from run_attr_array_v2.sh:
#
#   ism   : P x D x N            (54)  -- noise-sweep, single seed
#   ismS  : P x D x {0,.35,1} x S (90)  -- multi-seed at 3 noise levels
#
# NOT re-run (unaffected by the bug):
#   ismX (shuffled-label ISM) -- different tag branch, never touched noise_v
#   fa / mech / casc / pdiag  -- don't read label_noise_std at all
#
# Models already exist (force_retrain not needed) -- this is attribution-only,
# should run much faster than the original training+attribution array.
#
#SBATCH --account=aip-bensl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=jobs/logs/rb_ismfix_%A_%a.out

set -u
cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 30 ))

PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3 4 5)

is_in() { local x=$1; shift; for v in "$@"; do [[ "$x" == "$v" ]] && return 0; done; return 1; }

TASKS=()
for P in "${PROTS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do
    TASKS+=("ism $P $D $N -")
    is_in "$N" 0.0 0.35 1.0 && for S in "${SEEDS[@]}"; do TASKS+=("ismS $P $D $N $S"); done
  done
done; done

echo "total tasks: ${#TASKS[@]}"
t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read KIND P D N X <<< "$t"
NMD="results/synthetic/${P}_${D}/noise_${N}"
echo ">>> [$SLURM_ARRAY_TASK_ID] $KIND P=$P D=$D N=$N X=$X"

case "$KIND" in
  ism)
    python scripts/pipeline/ism_attribution.py --run-dir "${NMD}/linear" --model-dir "$NMD" \
      --out results/synthetic/ism_attribution ;;
  ismS)
    SD="results/synthetic/seedsweep/${P}_${D}/noise_${N}/seed_${X}"
    python scripts/pipeline/ism_attribution.py --run-dir "${SD}/linear" --model-dir "$SD" \
      --out results/synthetic/ism_attribution ;;
esac
