#!/bin/bash

# 1. Load modules
module load python/3.12 cuda/12.2

# 2. Activate env
source .venv/bin/activate

# 3. Execute code for p53
python scripts/run_integrated_gradients.py configs/synthetic/p53_nonlinear_config.yaml
