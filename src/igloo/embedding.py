import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

_DEFAULT_ESM_MODEL = 'esmc_300m'

# One-hot baseline "embedding": no pLM at all, just amino-acid identity.
# Used whenever esm_model_name == 'onehot' (see get_embeddings below). 21 dims
# = the 20 standard amino acids (same order as src/igloo/synthetic_data.py's
# AMINO_ACIDS) + 1 catch-all channel for anything else (X, U, gaps, ...).
_ONEHOT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_ONEHOT_DIM = len(_ONEHOT_ALPHABET) + 1


def _onehot_embeddings(sequences) -> np.ndarray:
    """Return a ``(N, Lmax, 21)`` float32 one-hot array — no ESM call, no
    cache. Free to (re)compute, so unlike the pLM path nothing is persisted
    to disk."""
    n = len(sequences)
    if n == 0:
        return np.empty((0, 0, _ONEHOT_DIM), dtype=np.float32)
    max_len = max(len(s) for s in sequences)
    out = np.zeros((n, max_len, _ONEHOT_DIM), dtype=np.float32)
    for i, s in enumerate(sequences):
        for j, aa in enumerate(s):
            k = _ONEHOT_ALPHABET.find(aa.upper())
            out[i, j, k if k >= 0 else _ONEHOT_DIM - 1] = 1.0
    return out


# ---------------------------------------------------------------------------
# Content-addressed per-sequence embedding cache
# ---------------------------------------------------------------------------
# Every unique (sequence, layer, esm_model) is stored EXACTLY ONCE at
#   data/embeddings/by_seq/<esm_model>/L<layer>/<hh>/<sha1(seq)>.safetensors
# (sharded by the first 2 hex chars of the hash to keep directories small).
#
# Legacy `<dataset_name>_embeddings_batch_<a>_<b>.safetensors` files are still
# READ (positionally, read-only — never migrated into the per-seq store) so
# datasets that have not been regenerated keep working unchanged.


def _project_root(caller_file: str) -> Path:
    d = os.path.dirname(os.path.abspath(caller_file))
    while not os.path.isdir(os.path.join(d, 'data')):
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("Could not find project root containing 'data' directory.")
        d = parent
    return Path(d)


def _seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode('utf-8')).hexdigest()


def _seq_path(base: Path, esm_model_name: str, layer: int, seq: str) -> Path:
    h = _seq_hash(seq)
    return base / 'by_seq' / esm_model_name / f'L{layer}' / h[:2] / f'{h}.safetensors'


def _atomic_save(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        os.close(fd)
        save_file({"embedding": tensor.contiguous()}, tmp)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_seq(path: Path):
    if not path.exists():
        return None
    try:
        emb = load_file(str(path))["embedding"].clone()  # .clone() breaks mmap backing (avoids Lustre SIGBUS)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    if emb.ndim == 2:
        emb = emb.unsqueeze(0)
    if torch.isnan(emb).any():
        print(f"WARNING: NaN in cached embedding {path.name} — deleting.")
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return emb


def _trim_padding(emb_row: torch.Tensor) -> torch.Tensor:
    """emb_row: (L, D). Drop trailing all-zero rows (batch padding)."""
    nz = (emb_row.abs().sum(dim=-1) != 0)
    if not nz.any():
        return emb_row
    last = int(torch.nonzero(nz, as_tuple=False).max().item())
    return emb_row[: last + 1]


def _read_legacy_batches(emb_dir: Path, dataset_name: str, sequences, batch_size: int,
                         out: list) -> None:
    """Fill still-missing rows in *out* from legacy `<dataset_name>_batch_<a>_<b>`
    files, positionally.  READ-ONLY — nothing is written to the per-seq store, so
    a caller passing a non-canonical list can at worst get a wrong result for that
    one call (same behaviour as the pre-content-addressed code), never a
    persistent poisoned cache entry."""
    n = len(sequences)
    for i in range(0, n, batch_size):
        start, end = i, min(i + batch_size, n)
        if all(out[j] is not None for j in range(start, end)):
            continue
        f = emb_dir / f"{dataset_name}_embeddings_batch_{start}_{end - 1}.safetensors"
        if not f.exists():
            continue
        try:
            batch = load_file(str(f))["embedding"].clone()  # .clone() breaks mmap backing (avoids Lustre SIGBUS)
        except Exception:
            continue
        for j in range(end - start):
            gi = start + j
            if out[gi] is not None or j >= batch.shape[0]:
                continue
            row = _trim_padding(batch[j])
            if torch.isnan(row).any() or not (0 <= row.shape[0] - len(sequences[gi]) <= 4):
                continue
            out[gi] = row.unsqueeze(0)


def get_embeddings(sequences, layer=33, batch_size=16, dataset_name='PafA', subdir=None,
                   esm_backend=None, esm_model_name=_DEFAULT_ESM_MODEL, device=None):
    """
    Return ESM embeddings for *sequences* as a ``(N, Lmax, D)`` float32 numpy
    array (zero-padded to the longest sequence).

    Newly computed embeddings are stored content-addressed under
    ``data/embeddings/by_seq/`` — one file per unique
    ``(sequence, layer, esm_model_name)``, shared across all call sites.
    ``dataset_name`` / ``subdir`` are only used to locate legacy batched cache
    files for backward-compatible reads; they do not affect where new
    embeddings are written.

    ``esm_model_name == 'onehot'`` bypasses the pLM entirely and returns a
    one-hot amino-acid encoding instead (see ``_onehot_embeddings``) — no
    caching, no backend, ``layer``/``esm_backend`` are ignored.
    """
    if esm_model_name == 'onehot':
        return _onehot_embeddings(sequences)

    import inspect
    root = _project_root(inspect.stack()[1].filename)
    base = root / 'data' / 'embeddings'
    legacy_dir = base / subdir if subdir else base
    base.mkdir(parents=True, exist_ok=True)

    n = len(sequences)
    if n == 0:
        return torch.empty(0).numpy()

    out = [None] * n

    # 1. content-addressed per-sequence cache
    for i, s in enumerate(sequences):
        out[i] = _load_seq(_seq_path(base, esm_model_name, layer, s))

    # 2. legacy batched files (read-only backward compatibility)
    if any(o is None for o in out):
        _read_legacy_batches(legacy_dir, dataset_name, sequences, batch_size, out)

    # 3. compute remaining misses, write each to the per-seq store
    missing = [i for i in range(n) if out[i] is None]
    if missing:
        _backend = esm_backend
        for k in range(0, len(missing), batch_size):
            idx = missing[k:k + batch_size]
            batch_seqs = [sequences[i] for i in idx]
            if _backend is None:
                from .esm_adapter import load_esm_backend
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                _backend = load_esm_backend(esm_model_name, device)
            emb_np = _backend.get_sequence_embeddings(batch_seqs, layer=layer,
                                                     batch_size=len(batch_seqs))
            emb = torch.from_numpy(np.asarray(emb_np))
            if torch.isnan(emb).any():
                raise RuntimeError(
                    f"ESM produced NaN embeddings ({int(torch.isnan(emb).sum())} values). "
                    "Ensure the ESM adapter returns float32 (see esm_adapter.py)."
                )
            for j, gi in enumerate(idx):
                row = _trim_padding(emb[j]).unsqueeze(0)
                out[gi] = row
                _atomic_save(row, _seq_path(base, esm_model_name, layer, sequences[gi]))

    # 4. assemble -> (N, Lmax, D) padded
    max_len = max(o.shape[1] for o in out)
    padded = []
    for o in out:
        if o.shape[1] < max_len:
            pad = torch.zeros(o.shape[0], max_len - o.shape[1], o.shape[2], dtype=o.dtype)
            o = torch.cat([o, pad], dim=1)
        padded.append(o)
    return torch.cat(padded, dim=0).to(torch.float32).numpy()
