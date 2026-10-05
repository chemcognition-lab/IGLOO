#!/bin/bash
# Smoke-test: load each new pLM backbone and embed one short test sequence,
# reporting peak GPU memory. Used to size the MIG tier for cache-warming and
# training before committing to the full grid.
#
# ankh/esmc use IGLOO_env (py3.11); e1_* need IGLOO_env_e1 (py3.12, separate
# venv because E1's pyproject.toml requires python>=3.12).
#
# Task list: 6 backbones
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --array=0-5
#SBATCH --output=jobs/logs/smoketest_backbone_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

BACKBONES=(esmc_600m ankh_base ankh_large e1_150m e1_300m e1_600m)
BB="${BACKBONES[$SLURM_ARRAY_TASK_ID]}"
echo ">>> smoke-testing backbone: $BB"

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

python - "$BB" <<'PY'
import sys, time
import torch
sys.path.insert(0, ".")
from src.igloo.esm_adapter import load_esm_backend

model_name = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

t0 = time.time()
backend = load_esm_backend(model_name, device)
t1 = time.time()
print(f"loaded {model_name} in {t1-t0:.1f}s, family={backend.family}")

# GB1 WT, 56 residues -- same test sequence used throughout this project
wt = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
seqs = [wt, wt[:30]]  # two short sequences of different length, forces padding path

t2 = time.time()
emb = backend.get_sequence_embeddings(seqs, layer=33, batch_size=2)
t3 = time.time()
print(f"embedded {len(seqs)} seqs in {t3-t2:.1f}s -> shape {emb.shape}")

if torch.cuda.is_available():
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak GPU memory: {peak_gb:.2f} GB")

print(f"RESULT OK {model_name}")
PY
