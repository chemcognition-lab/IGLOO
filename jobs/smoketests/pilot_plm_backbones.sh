#!/bin/bash
# Pilot: one real training+IG run (production settings: batch_size from
# config, steps=500) per (architecture x backbone x protein) combo, before
# committing to the full 2916-task grid (3 archs x 6 backbones x full noise
# grid x 3 seeds x 2 proteins x 3 landscapes).
#
# Tests both GB1 (56 aa) and T4Lysozyme (164 aa) since the longer sequence +
# larger embedding dim combo is exactly what caused the e1_600m OOM during
# cache warming -- want to catch that here, not mid-way through the full grid.
#
# e1_* backbones need IGLOO_env_e1 (py3.12); ankh/esmc use IGLOO_env (py3.11).
#
# Task list: 3 archs x 6 backbones x 2 proteins = 36
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-35
#SBATCH --output=jobs/logs/pilot_plm_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

ARCHS=(esm_perres_attn esm_attention_premlp esm_perres_mean)
BACKBONES=(esmc_600m ankh_base ankh_large e1_150m e1_300m e1_600m)
PROTS=(GB1_additive T4Lysozyme_additive)

TASKS=()
for A in "${ARCHS[@]}"; do for BB in "${BACKBONES[@]}"; do for P in "${PROTS[@]}"; do
  TASKS+=("$A $BB $P")
done; done; done

t="${TASKS[$SLURM_ARRAY_TASK_ID]}"
read ARCH BB PROT <<< "$t"
echo ">>> [$SLURM_ARRAY_TASK_ID] arch=$ARCH backbone=$BB protein=$PROT"

case "$BB" in
  e1_*)
    module load python/3.12 cuda/12.2
    source IGLOO_env_e1/bin/activate
    ;;
  *)
    module load python/3.12 gcc arrow cuda/12.2
    source IGLOO_env/bin/activate
    ;;
esac

CFG="configs/synthetic/${PROT}_config.yaml"
MD="results/synthetic/_pilot_plm/${ARCH}_${BB}/${PROT}"

python scripts/pipeline/run_integrated_gradients.py "$CFG" \
  model.esm_model_name="$BB" model.pooling_method="$ARCH" \
  integrated_gradients.interpolation_path=linear \
  training.seed=1 \
  model_dir="$MD" results_dir="${MD}/linear" force_retrain=true
