#!/bin/bash
#SBATCH --account=rrg-mahadeva
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --job-name=test_nan_fix
#SBATCH --output=jobs/logs/test_nan_fix_%j.out

cd /home/ranaab/projects/rrg-mahadeva/ranaab/IGLOO || exit 1

module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

echo "Testing bfloat16 NaN fix on H100..."
python scripts/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml \
    ig.interpolation_path=linear \
    training.noise.label_noise_std=0.0 \
    model_dir=results/synthetic/GB1_additive/noise_0.0 \
    results_dir=jobs/logs/test_nan_fix_results \
    force_retrain=false
