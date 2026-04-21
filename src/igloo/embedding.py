import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_DEFAULT_ESM_MODEL = 'esmc_300m'


def get_embeddings(sequences, layer=33, batch_size=16, dataset_name='PafA', subdir=None,
                   esm_backend=None, esm_model_name=_DEFAULT_ESM_MODEL):
    """
    Return ESM embeddings for *sequences*, using a disk cache under data/embeddings/.

    Parameters
    ----------
    esm_backend : ESM2Backend | ESMCBackend, optional
        A pre-loaded backend from :mod:`src.igloo.esm_adapter`.  When provided and
        a cache miss occurs, this backend is used directly — avoiding loading a
        second ESM model.  If *None*, a temporary backend is constructed from
        *esm_model_name* on the first cache miss.
    esm_model_name : str
        Model identifier used **only** when *esm_backend* is not supplied and at
        least one batch file is missing from disk.  Defaults to
        ``'esm2_t33_650M_UR50D'``.
    """
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
    if subdir:
        emb_dir = emb_dir / subdir
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
            if torch.isnan(emb).any():
                print(f"WARNING: NaN detected in cached embeddings {f.name} — deleting corrupt file.")
                f.unlink()
                break  # fall through to the recompute path below
            embeddings.append(emb)
        else:
            # All files loaded without NaN — fast return
            if embeddings:
                max_len = max(emb.shape[1] for emb in embeddings)
                padded_embeddings = []
                for emb in embeddings:
                    if emb.shape[1] < max_len:
                        padding = torch.zeros(emb.shape[0], max_len - emb.shape[1], emb.shape[2])
                        emb = torch.cat([emb, padding], dim=1)
                    padded_embeddings.append(emb)
                return torch.cat(padded_embeddings, dim=0).numpy()
            else:
                return torch.empty(0).numpy()
        # Reset for recompute path
        embeddings = []

    # Only initialise a backend if needed (cache miss present and none supplied)
    _backend = None

    for i in range(0, n, batch_size):
        start = i
        end = min(i + batch_size, n)
        batch_file = emb_dir / f"{dataset_name}_embeddings_batch_{start}_{end-1}.safetensors"
        if batch_file.exists():
            emb = load_file(str(batch_file))["embedding"]
            embeddings.append(emb)
            continue

        # --- Cache miss: compute this batch ---
        if _backend is None:
            if esm_backend is not None:
                _backend = esm_backend
            else:
                from .esm_adapter import load_esm_backend  # lazy import
                _backend = load_esm_backend(esm_model_name, torch.device('cpu'))

        batch_seqs = sequences[start:end]
        emb_np = _backend.get_sequence_embeddings(batch_seqs, layer=layer,
                                                   batch_size=len(batch_seqs))
        emb = torch.from_numpy(emb_np)  # (batch, L, D)
        if torch.isnan(emb).any():
            nan_count = torch.isnan(emb).sum().item()
            raise RuntimeError(
                f"ESM produced NaN embeddings for batch {start}-{end-1} "
                f"({nan_count} NaN values). This typically indicates a bfloat16 "
                f"overflow for long sequences. Ensure the ESM adapter returns "
                f"float32 tensors (check esm_adapter.py)."
            )
        # Atomic write: write to a temp file then rename so concurrent jobs
        # don't corrupt a partially-written cache file (POSIX rename is atomic).
        tmp_fd, tmp_path = tempfile.mkstemp(dir=emb_dir, suffix=".tmp")
        try:
            os.close(tmp_fd)
            save_file({"embedding": emb}, tmp_path)
            os.replace(tmp_path, str(batch_file))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        embeddings.append(emb)

    # Handle variable sequence lengths by padding to max length
    if embeddings:
        max_len = max(emb.shape[1] for emb in embeddings)
        padded_embeddings = []
        for emb in embeddings:
            if emb.shape[1] < max_len:
                # Pad with zeros to match max length
                padding = torch.zeros(emb.shape[0], max_len - emb.shape[1], emb.shape[2])
                emb = torch.cat([emb, padding], dim=1)
            padded_embeddings.append(emb)
        return torch.cat(padded_embeddings, dim=0).numpy()
    else:
        return torch.empty(0).numpy()
