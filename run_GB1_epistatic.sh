#!/bin/bash

# 1. Load modules
module load python/3.12 cuda/12.2

# 2. Activate env
source IGLOO_env/bin/activate

# 3. Execute code for GB1
python scripts/run_integrated_gradients.py configs/synthetic/GB1_epistatic_config.yaml
