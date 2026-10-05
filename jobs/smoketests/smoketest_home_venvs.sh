#!/bin/bash
# One-off: confirm real GPU jobs work when sourcing the venvs from their NEW
# /home location, BEFORE the /project copies are deleted+symlinked. Mirrors
# smoketest_new_backbones.sh but points explicitly at /home/ranaab/IGLOO_env*
# (absolute paths) rather than the relative "IGLOO_env/bin/activate" that
# job scripts normally use, so this doesn't accidentally fall back to the
# still-present /project originals.
#
# Task 0: esmc_600m via /home/ranaab/IGLOO_env      (py3.11 main env)
# Task 1: e1_150m    via /home/ranaab/IGLOO_env_e1   (py3.12 E1 env)
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --array=0-1
#SBATCH --output=jobs/logs/smoketest_home_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  BB="esmc_600m"
  module load python/3.12 gcc arrow cuda/12.2
  source /home/ranaab/IGLOO_env/bin/activate
else
  BB="e1_150m"
  module load python/3.12 cuda/12.2
  source /home/ranaab/IGLOO_env_e1/bin/activate
fi
echo ">>> smoke-testing backbone: $BB via $(python -c 'import sys; print(sys.prefix)')"

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

wt = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
seqs = [wt, wt[:30]]

t2 = time.time()
emb = backend.get_sequence_embeddings(seqs, layer=33, batch_size=2)
t3 = time.time()
print(f"embedded {len(seqs)} seqs in {t3-t2:.1f}s -> shape {emb.shape}")

if torch.cuda.is_available():
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak GPU memory: {peak_gb:.2f} GB")

print(f"RESULT OK {model_name}")
PY
