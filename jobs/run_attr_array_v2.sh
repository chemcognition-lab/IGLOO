#!/bin/bash
# Option-B rebuild v2 — ATTRIBUTION / DIAGNOSTICS wave, BLEND noise grid.
# Submit with --dependency=afterany:<train_array_v2_jobid> so models exist first.
#
# Noise axis: blend p in {0, .05, .1, .2, .35, .5, .7, .85, 1}
#
# Tasks (index = SLURM_ARRAY_TASK_ID-1):
#   factor ablation : P x D x N x {signed,gradient,displacement,grad_input}  (216)
#   ISM noise       : P x D x N                                              (54)
#   ISM multi-seed  : P x D x {0.0,0.35,1.0} x S                             (90)
#   ISM shuffled    : P x D x S                                             (30)
#   mechanism       : P x D x {0.0,0.5,1.0}                                  (18)
#   cascading       : P x D x {0.0,0.35,0.7,1.0}                             (24)
#   premlp diag     : P x D x {0.0,1.0} x {mech,casc}                        (24)
# Total 456.
#
#SBATCH --account=aip-bensl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:50:00
#SBATCH --output=jobs/logs/rb_attr_v2_%A_%a.out

set -u
cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
sleep $(( RANDOM % 30 ))

PROTS=(GB1 T4Lysozyme)
LAND=(additive epistatic nonlinear)
NOISE=(0.0 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)
SEEDS=(1 2 3 4 5)
FA=(signed gradient displacement grad_input)

is_in() { local x=$1; shift; for v in "$@"; do [[ "$x" == "$v" ]] && return 0; done; return 1; }

TASKS=()
for P in "${PROTS[@]}"; do for D in "${LAND[@]}"; do
  for N in "${NOISE[@]}"; do
    for M in "${FA[@]}"; do TASKS+=("fa $P $D $N $M"); done
    TASKS+=("ism $P $D $N -")
    is_in "$N" 0.0 0.35 1.0      && for S in "${SEEDS[@]}"; do TASKS+=("ismS $P $D $N $S"); done
    is_in "$N" 0.0 0.5 1.0       && TASKS+=("mech $P $D $N -")
    is_in "$N" 0.0 0.35 0.7 1.0  && TASKS+=("casc $P $D $N -")
  done
  for S in "${SEEDS[@]}"; do TASKS+=("ismX $P $D 0 $S"); done
  for N in 0.0 1.0; do TASKS+=("pdiagM $P $D $N -"); TASKS+=("pdiagC $P $D $N -"); done
done; done

echo "total tasks: ${#TASKS[@]}"
t="${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
read KIND P D N X <<< "$t"
CFG="configs/synthetic/${P}_${D}_config.yaml"
NMD="results/synthetic/${P}_${D}/noise_${N}"
echo ">>> [$SLURM_ARRAY_TASK_ID] $KIND P=$P D=$D N=$N X=$X"

case "$KIND" in
  fa)
    RD="results/synthetic/factor_ablation/${P}_${D}/noise_${N}/${X}"
    python scripts/run_integrated_gradients.py "$CFG" \
      integrated_gradients.interpolation_path=linear integrated_gradients.attribution_method="$X" \
      integrated_gradients.completeness_validation.enabled=false \
      training.noise.label_noise_mode=blend training.noise.label_noise_level="$N" \
      model_dir="$NMD" results_dir="$RD" force_retrain=false ;;
  ism)
    python scripts/ism_attribution.py --run-dir "${NMD}/linear" --model-dir "$NMD" \
      --out results/synthetic/ism_attribution ;;
  ismS)
    SD="results/synthetic/seedsweep/${P}_${D}/noise_${N}/seed_${X}"
    python scripts/ism_attribution.py --run-dir "${SD}/linear" --model-dir "$SD" \
      --out results/synthetic/ism_attribution ;;
  ismX)
    SD="results/synthetic/seedsweep/${P}_${D}/shuffled/seed_${X}"
    python scripts/ism_attribution.py --run-dir "${SD}/linear" --model-dir "$SD" \
      --out results/synthetic/ism_attribution ;;
  mech)
    python scripts/attribution_mechanism.py --run-dir "${NMD}/linear" --model-dir "$NMD" \
      --n-variants 300 --steps 64 --out results/synthetic/attribution_mechanism ;;
  casc)
    python scripts/cascading_randomization.py --run-dir "${NMD}/linear" --model-dir "$NMD" \
      --n-variants 50 --steps 100 --out results/synthetic/cascading_randomization ;;
  pdiagM|pdiagC)
    PD="results/synthetic/premlp/${P}_${D}/noise_${N}/seed_1"
    if [[ "$KIND" == pdiagM ]]; then
      python scripts/attribution_mechanism.py --run-dir "${PD}/linear" --model-dir "$PD" \
        --n-variants 300 --steps 64 --out results/synthetic/attribution_mechanism_premlp
    else
      python scripts/cascading_randomization.py --run-dir "${PD}/linear" --model-dir "$PD" \
        --n-variants 50 --steps 100 --out results/synthetic/cascading_randomization_premlp
    fi ;;
esac
