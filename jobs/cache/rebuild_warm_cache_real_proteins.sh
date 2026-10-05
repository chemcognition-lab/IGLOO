#!/bin/bash
# Warm the ankh_base (true final layer, L48) embedding cache for the two
# real-protein families being switched onto our best-validated setup
# (ankh_base + MeanPool/AttnPool, per the GB1/T4Lysozyme/p53 comparison):
#   - alpha-amylase (FLIP2): 4 configs share the same 3,706 sequences
#   - beta-lactamase (ProteinGym BLAT_ECOLX): 4 DMS studies share the same
#     ~4,997 sequences (single-mutant library of one wildtype)
# Both warmed once each rather than per-config, same as the FLIP2/p53
# cache-warm scripts.
#
# layer=48 explicitly (not the stale configs' old esm_layer=33 default) --
# see rebuild_warm_cache_p53_backbones.sh for why 48 is ankh_base's true
# final layer now that AnkhBackend honors `layer` for real.
#
# Task list: 2 (alpha-amylase, beta-lactamase)
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/logs/warm_cache_real_proteins_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
import pandas as pd
from src.igloo.embedding import get_embeddings

task_id = int(sys.argv[1])

if task_id == 0:
    df = pd.read_csv("data/FLIP2_processed/alpha_amylase__mutation__by_mutation_combined.csv")
    seqs = sorted(df["SEQ"].unique())
    label = "alpha_amylase"
else:
    seqs = set()
    for name in ("Deng_2012", "Firnberg_2014", "Jacquier_2013", "Stiffler_2015"):
        for split in ("train", "test"):
            f = f"data/ProteinGym_DMS_Assays_Sub_processed/BLAT_ECOLX_{name}_{split}.csv"
            seqs |= set(pd.read_csv(f)["SEQ"])
    seqs = sorted(seqs)
    label = "beta_lactamase (BLAT_ECOLX)"

print(f"{label} / ankh_base L48: warming {len(seqs)} unique sequences ...", flush=True)
get_embeddings(seqs, layer=48, batch_size=32, esm_model_name="ankh_base")
print(f"{label} / ankh_base L48: done", flush=True)
PY

echo "=== by_seq store for ankh_base/L48 ==="
find "data/embeddings/by_seq/ankh_base/L48" -name '*.safetensors' 2>/dev/null | wc -l
