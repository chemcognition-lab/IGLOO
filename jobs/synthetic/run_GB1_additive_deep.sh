#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --job-name=GB1_additive_deep
#SBATCH --output=jobs/logs/slurm-%j.out

# Change to project root (script lives in jobs/)
cd /project/aip-bensl/ranaab/IGLOO || exit 1

# 1. Load modules
module load python/3.12 cuda/12.2

# 2. Activate env
source IGLOO_env/bin/activate

# 3. Generate the deep-mutation dataset (sporadic 1–10 mutations, peaks at 3–5)
python scripts/data/generate_synthetic_datasets.py \
    --proteins GB1 \
    --models additive \
    --mutation-profile deep \
    --output-dir data/synthetic/deep \
    --seed 42

# 4. Run IGLOO analysis on the deep dataset
python scripts/pipeline/run_integrated_gradients.py configs/synthetic/GB1_additive_deep_config.yaml
