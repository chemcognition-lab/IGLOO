#!/bin/bash
# Warm the by_seq embedding cache for the FLIP2 alpha-amylase datasets
# (esmc_300m, layer 33 -- matches every configs/FLIP2/*.yaml). All 4 configs
# (mutation__by_mutation, number__one_to_many, position__close_to_far,
# position__far_to_close) share the exact same 3,706 unique sequences (same
# underlying data, just different train/test split labelings), so this
# warms the cache once from a single combined CSV rather than 4 times.
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/warm_cache_flip2_%j.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

python - <<'PY'
import pandas as pd
from src.igloo.embedding import get_embeddings

df = pd.read_csv("data/FLIP2_processed/alpha_amylase__mutation__by_mutation_combined.csv")
seqs = sorted(df["SEQ"].unique())
print(f"FLIP2 alpha_amylase / esmc_300m: warming {len(seqs)} unique sequences ...", flush=True)
get_embeddings(seqs, layer=33, batch_size=32, esm_model_name="esmc_300m")
print("FLIP2 alpha_amylase / esmc_300m: done", flush=True)
PY

echo "=== by_seq store for esmc_300m L33 ==="
find "data/embeddings/by_seq/esmc_300m/L33" -name '*.safetensors' 2>/dev/null | wc -l
