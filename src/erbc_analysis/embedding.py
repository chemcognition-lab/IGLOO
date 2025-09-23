import os
from pathlib import Path

import esm
import torch
from safetensors.torch import load_file, save_file

# Determine script directory and set paths relative to it

def get_embeddings(sequences, layer=33, batch_size=16, dataset_name='PafA'):
    # Always compute project_root as the top-level project directory (containing 'data')
    import inspect
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    # Find the project root by searching upwards for a directory containing 'data'
    script_dir = os.path.dirname(os.path.abspath(caller_file))
    project_root = script_dir
    while not os.path.isdir(os.path.join(project_root, 'data')):
        parent = os.path.dirname(project_root)
        if parent == project_root:
            raise RuntimeError("Could not find project root containing 'data' directory.")
        project_root = parent

    embeddings = []
    emb_dir = Path(project_root) / 'data' / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    n = len(sequences)
    batch_files = []
    for i in range(0, n, batch_size):
        start = i
        end = min(i + batch_size, n)
        batch_file = emb_dir / f"{dataset_name}_embeddings_batch_{start}_{end-1}.safetensors"
        batch_files.append(batch_file)

    # If all batch files exist, load and return them (skip ESM model)
    if all(f.exists() for f in batch_files):
        for f in batch_files:
            emb = load_file(str(f))["embedding"]
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0).numpy()

    # Only load ESM model if needed
    print("Loading ESM model...")
    esm_model_name = 'esm2_t33_650M_UR50D'
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    for i in range(0, n, batch_size):
        start = i
        end = min(i + batch_size, n)
        batch_file = emb_dir / f"{dataset_name}_embeddings_batch_{start}_{end-1}.safetensors"
        if batch_file.exists():
            emb = load_file(str(batch_file))["embedding"]
            embeddings.append(emb)
            continue
        batch = [(str(j), sequences[j]) for j in range(start, end)]
        _, _, tokens = batch_converter(batch)
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[layer], return_contacts=False)
        reps = out['representations'][layer].cpu()  # (batch, L+2, D)
        Lp2 = reps.shape[1]
        emb = reps[:, 1:Lp2-1, :].contiguous()  # (batch, L, D)
        save_file({"embedding": emb}, str(batch_file))
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0).numpy()
