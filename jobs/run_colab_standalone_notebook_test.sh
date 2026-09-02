#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=colab_standalone_nb_test
#SBATCH --output=jobs/logs/slurm-%j.out

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"
cd "${IGLOO_ROOT}"
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

cd colab_standalone
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=igloo_env \
    --ExecutePreprocessor.timeout=600 \
    --output=IGLOO_playground_TESTRUN.ipynb \
    IGLOO_playground.ipynb

echo "Test execution succeeded."
