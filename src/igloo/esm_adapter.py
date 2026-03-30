"""
ESM model adapter layer.

Provides a unified interface for ESM-2 (classic fair-esm API) and ESM-C
(EvolutionaryScale new API) so the rest of the codebase is model-family-agnostic.

Usage
-----
    from src.igloo.esm_adapter import load_esm_backend

    backend = load_esm_backend("esm2_t33_650M_UR50D", device)  # ESM-2
    backend = load_esm_backend("esmc_300m", device)              # ESM-C

    embeddings = backend.get_sequence_embeddings(sequences, layer=33)
    # → numpy array of shape (N, L, D)

Notes
-----
* Token-level IG requires direct access to ESM-2 model internals (embed_tokens,
  transformer layers). This is not supported for ESM-C.  The pipeline will raise
  a clear ValueError if token-level IG is attempted with an ESM-C model.
* ESM-C ignores the `layer` argument (it always returns final-layer embeddings).
  Pass any value; it will be silently ignored.
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Model-name helpers
# ---------------------------------------------------------------------------

ESMC_PREFIXES = ("esmc",)

_DEFAULT_ESM_MODEL = 'esmc_300m'


def infer_model_family(model_name: str) -> str:
    """Return ``'esm2'`` or ``'esmc'`` based on the model name string."""
    lower = model_name.lower()
    if any(lower.startswith(p) for p in ESMC_PREFIXES):
        return "esmc"
    return "esm2"


# ---------------------------------------------------------------------------
# ESM-2 backend
# ---------------------------------------------------------------------------


class ESM2Backend:
    """Wraps the classic ``esm.pretrained`` API."""

    family: str = "esm2"

    def __init__(self, model: torch.nn.Module, alphabet) -> None:
        self.model = model
        self.alphabet = alphabet

    @classmethod
    def from_pretrained(cls, model_name: str, device: torch.device) -> ESM2Backend:
        import esm as esm_lib  # type: ignore[import]

        print(f"Loading ESM-2 model {model_name}...")
        model, alphabet = esm_lib.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device).eval()
        return cls(model, alphabet)

    def get_sequence_embeddings(
        self,
        sequences: list[str],
        layer: int = 33,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return ``(N, L, D)`` float32 numpy array with BOS/EOS stripped."""
        batch_converter = self.alphabet.get_batch_converter()
        device = next(self.model.parameters()).device
        all_embs: list[torch.Tensor] = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch = [(str(j), s) for j, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            with torch.no_grad():
                out = self.model(tokens, repr_layers=[layer], return_contacts=False)
            reps = out["representations"][layer].cpu()
            emb = reps[:, 1:-1, :].contiguous()  # strip BOS/EOS → (batch, L, D)
            all_embs.append(emb)

        return _concat_with_padding(all_embs)


# ---------------------------------------------------------------------------
# ESM-C backend
# ---------------------------------------------------------------------------


class ESMCBackend:
    """Wraps the new EvolutionaryScale ``ESMC`` API."""

    family: str = "esmc"

    def __init__(self, client: torch.nn.Module, model_name: str) -> None:
        self.model = client     # ESMC is also a nn.Module
        self.alphabet = None    # no alphabet concept in ESM-C
        self._model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str, device: torch.device) -> ESMCBackend:
        from esm.models.esmc import ESMC  # type: ignore[import]

        # Normalise versioned names:  esmc-300m-2024-12  →  esmc_300m
        key = model_name.lower().replace("-", "_")
        parts = []
        for p in key.split("_"):
            # Stop at year-like token (e.g. "2024", "2024")
            if p.isdigit() and int(p) > 2000:
                break
            parts.append(p)
        key = "_".join(parts)  # e.g. esmc_300m

        print(f"Loading ESM-C model {key}...")
        client = ESMC.from_pretrained(key).to(device)
        client.eval()
        return cls(client, model_name)

    def get_sequence_embeddings(
        self,
        sequences: list[str],
        layer: int | None = None,  # ignored; ESM-C always returns final-layer
        batch_size: int = 16,      # kept for interface parity; ESM-C encodes one at a time
    ) -> np.ndarray:
        """Return ``(N, L, D)`` float32 numpy array with BOS/EOS stripped."""
        from esm.sdk.api import ESMProtein, LogitsConfig  # type: ignore[import]

        all_embs: list[torch.Tensor] = []
        for seq in sequences:
            protein = ESMProtein(sequence=seq)
            protein_tensor = self.model.encode(protein)
            with torch.no_grad():
                logits_output = self.model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
            # ESM-C embeddings are already (1, L, D) — no BOS/EOS tokens in output
            emb = logits_output.embeddings.cpu()  # (1, L, D)
            all_embs.append(emb)

        return _concat_with_padding(all_embs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_esm_backend(model_name: str, device: torch.device) -> ESM2Backend | ESMCBackend:
    """Return the correct backend for *model_name*.

    Args:
        model_name: e.g. ``'esm2_t33_650M_UR50D'`` or ``'esmc_300m'``.
        device: PyTorch device to load the model onto.

    Returns:
        :class:`ESM2Backend` for ESM-2 models, :class:`ESMCBackend` for ESM-C models.
    """
    family = infer_model_family(model_name)
    if family == "esmc":
        return ESMCBackend.from_pretrained(model_name, device)
    return ESM2Backend.from_pretrained(model_name, device)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _concat_with_padding(embs: list[torch.Tensor]) -> np.ndarray:
    """Concatenate ``(batch_i, L_i, D)`` tensors, zero-padding to ``max_len``."""
    if not embs:
        return np.empty(0, dtype=np.float32)
    max_len = max(e.shape[1] for e in embs)
    padded = []
    for e in embs:
        if e.shape[1] < max_len:
            pad = torch.zeros(e.shape[0], max_len - e.shape[1], e.shape[2], dtype=e.dtype, device=e.device)
            e = torch.cat([e, pad], dim=1)
        padded.append(e)
    return torch.cat(padded, dim=0).to(torch.float32).cpu().numpy()
