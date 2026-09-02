#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=colab_standalone_isotest
#SBATCH --output=jobs/logs/slurm-%j.out

set -euo pipefail

# Override for other machines/users:  IGLOO_ROOT=/path/to/IGLOO sbatch ...
IGLOO_ROOT="${IGLOO_ROOT:-/home/uoftshen/scratch/IGLOO}"
module load python/3.12 cuda/12.2

# Deliberately use an absolute path to the venv while running FROM the isolated
# copy -- the venv is just a Python interpreter + installed packages, not part
# of what we're testing. What we ARE testing is that nothing in the isolated
# folder's own code/imports/caching reaches back into ${IGLOO_ROOT}.
VENV_PY="${IGLOO_ROOT}"/IGLOO_env/bin

cd /home/uoftshen/scratch/colab_isolation_test
"$VENV_PY/jupyter" nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=igloo_env \
    --ExecutePreprocessor.timeout=600 \
    --output=IGLOO_playground_ISOTEST.ipynb \
    IGLOO_playground.ipynb

echo "Isolated standalone test execution succeeded."
