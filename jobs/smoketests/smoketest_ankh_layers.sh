#!/bin/bash
# Functional validation of the new Ankh intermediate-layer support
# (src/igloo/esm_adapter.py: AnkhBackend.get_sequence_embeddings). Checks:
#   1. num_layers() reports a sane value
#   2. layer=None and layer=num_layers (explicit final index) match closely
#   3. layer=0 (embedding output) and a middle layer differ substantially
#      from the final layer -- confirms layer selection actually does
#      something, not a silent no-op
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=jobs/logs/smoketest_ankh_layers_%j.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

python - <<'PY'
import sys, torch
import numpy as np
sys.path.insert(0, ".")
from src.igloo.esm_adapter import load_esm_backend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

backend = load_esm_backend("ankh_base", device)
n_layers = backend.num_layers()
print(f"ankh_base num_layers={n_layers} (hidden_states will have {n_layers+1} entries)")

wt = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
seqs = [wt, wt[:30]]

emb_none  = backend.get_sequence_embeddings(seqs, layer=None, batch_size=2)
emb_final = backend.get_sequence_embeddings(seqs, layer=n_layers, batch_size=2)
emb_mid   = backend.get_sequence_embeddings(seqs, layer=n_layers // 2, batch_size=2)
emb_embed = backend.get_sequence_embeddings(seqs, layer=0, batch_size=2)

print(f"shapes: none={emb_none.shape} final={emb_final.shape} mid={emb_mid.shape} embed={emb_embed.shape}")
assert emb_none.shape == emb_final.shape == emb_mid.shape == emb_embed.shape, "shape mismatch across layers"

d_none_final = float(np.abs(emb_none - emb_final).mean())
d_final_mid  = float(np.abs(emb_final - emb_mid).mean())
d_final_embed = float(np.abs(emb_final - emb_embed).mean())
print(f"mean|none - final|  = {d_none_final:.6f}  (expect ~0, same layer two ways)")
print(f"mean|final - mid|   = {d_final_mid:.6f}   (expect >0, different layers)")
print(f"mean|final - embed| = {d_final_embed:.6f} (expect >0 and larger than mid, embedding vs final)")

assert d_none_final < 1e-4, "layer=None should match layer=num_layers (final) almost exactly"
assert d_final_mid > 1e-3, "middle layer should differ meaningfully from final layer"
assert d_final_embed > 1e-3, "embedding-layer output should differ meaningfully from final layer"

# out-of-range check
try:
    backend.get_sequence_embeddings(seqs, layer=n_layers + 5, batch_size=2)
    print("ERROR: expected ValueError for out-of-range layer, none raised")
    sys.exit(1)
except ValueError as e:
    print(f"out-of-range layer correctly rejected: {e}")

print("RESULT OK: Ankh layer selection verified")
PY
