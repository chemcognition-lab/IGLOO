"""
ESM model adapter layer.

Provides a unified interface for ESM-2 (classic fair-esm API), ESM-C
(EvolutionaryScale new API), Ankh (T5-encoder protein LM), and Profluent E1
(masked encoder family) so the rest of the codebase is model-family-agnostic.

Usage
-----
    from src.igloo.esm_adapter import load_esm_backend

    backend = load_esm_backend("esm2_t33_650M_UR50D", device)  # ESM-2
    backend = load_esm_backend("esmc_300m", device)              # ESM-C
    backend = load_esm_backend("ankh_base", device)              # Ankh base
    backend = load_esm_backend("ankh_large", device)             # Ankh large
    backend = load_esm_backend("ankh3_large", device)            # Ankh3 large
    backend = load_esm_backend("ankh3_xl", device)               # Ankh3 XL
    backend = load_esm_backend("e1_150m", device)                # Profluent E1-150M
    backend = load_esm_backend("e1_300m", device)                # Profluent E1-300M
    backend = load_esm_backend("e1_600m", device)                # Profluent E1-600M

    embeddings = backend.get_sequence_embeddings(sequences, layer=33)
    # → numpy array of shape (N, L, D)

Notes
-----
* Token-level IG requires direct access to ESM-2 model internals (embed_tokens,
  transformer layers). This is not supported for ESM-C, Ankh, or E1.  The
  pipeline will raise a clear ValueError if token-level IG is attempted with
  those models.
* ESM-C, Ankh, and E1 ignore the `layer` argument (they always return
  final-layer embeddings).  Pass any value; it will be silently ignored.
* Ankh requires ``pip install ankh``.
* E1 requires cloning https://github.com/Profluent-AI/E1 and
  ``pip install -e .`` (optionally ``pip install flash-attn``).
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Model-name helpers
# ---------------------------------------------------------------------------

ESMC_PREFIXES = ("esmc",)
ANKH_PREFIXES = ("ankh",)
E1_PREFIXES   = ("e1",)

_DEFAULT_ESM_MODEL = 'esmc_300m'

# Ankh model key → loader function name
# (ankh v1.x uses load_base_model / load_large_model;
#  ankh v2+ uses load_ankh_base / load_ankh_large;
#  ankh3 models were added in later releases)
_ANKH_LOADERS: dict[str, str] = {
    "ankh_base":   "load_base_model",
    "ankh_large":  "load_large_model",
    "ankh3_large": "load_ankh3_large",
    "ankh3_xl":    "load_ankh3_xl",
}

# E1 model key → HuggingFace repo ID
_E1_REPOS: dict[str, str] = {
    "e1_150m": "Profluent-Bio/E1-150m",
    "e1_300m": "Profluent-Bio/E1-300m",
    "e1_600m": "Profluent-Bio/E1-600m",
}


def infer_model_family(model_name: str) -> str:
    """Return ``'esm2'``, ``'esmc'``, ``'ankh'``, or ``'e1'`` based on the model name."""
    lower = model_name.lower()
    if any(lower.startswith(p) for p in ESMC_PREFIXES):
        return "esmc"
    if any(lower.startswith(p) for p in ANKH_PREFIXES):
        return "ankh"
    if any(lower.startswith(p) for p in E1_PREFIXES):
        return "e1"
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

    #: Alias so ``backend.get_embeddings(seqs)`` works alongside the canonical name.
    get_embeddings = get_sequence_embeddings


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
        client = ESMC.from_pretrained(key)
        # Materialize all mmap-backed tensors into real CPU RAM before moving
        # to device.  Without this, safetensors leaves tensors backed by a memory-
        # mapped file handle; on network filesystems (Lustre, NFS) the first
        # dereference of that mapping raises SIGBUS ("Bus error").
        # Must cover both parameters() AND buffers() — the latter includes embedding
        # tables and positional encodings that are registered via register_buffer().
        for p in client.parameters():
            p.data = p.data.clone()
        for module in client.modules():
            for buf_name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[buf_name] = buf.clone()
        client = client.to(device).float()  # force float32 to avoid bfloat16 NaN overflow
        client.eval()
        return cls(client, model_name)

    def get_sequence_embeddings(
        self,
        sequences: list[str],
        layer: int | None = None,  # ignored; ESM-C always returns final-layer
        batch_size: int = 16,      # kept for interface parity; ESM-C encodes one at a time
    ) -> np.ndarray:
        """Return ``(N, L, D)`` float32 numpy array with BOS/EOS stripped.

        CORRECTNESS NOTE (found 2026-09-15): ``client.logits(...)`` does NOT
        strip BOS/EOS — verified empirically: encoding a 56-residue sequence
        returns an (1, 58, 960) embedding (58 = 56 + BOS + EOS, tokenizer ids
        0 and 2). The previous version of this method returned that
        (1, L+2, D) tensor uncorrected (the docstring's claim that ESM-C
        "already" excludes special tokens was wrong), so every downstream
        consumer — training, evaluation, and embedding-level IG — silently
        operated on an off-by-one-shifted sequence: reported "position 1" was
        actually the BOS token, positions 2..L were actually residues 1..L-1,
        and the true last residue was dropped entirely by the `[:len(seq)]`
        truncation in sequence_ig.py. Every ESM-C-trained model and every
        ESM-C-based attribution result computed before this fix needs to be
        regenerated from a purged embedding cache to be trusted.
        """
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
            # ESM-C logits output includes BOS (index 0) and EOS (index -1) boundary
            # tokens — strip them so the returned tensor is (1, L, D) aligned to the
            # actual amino-acid sequence (same convention as ESM-2/Ankh/E1).
            # Cast to float32 before moving to CPU: bfloat16 on H100 can produce
            # NaN for long sequences due to softmax overflow.
            emb = logits_output.embeddings.float().cpu()[:, 1:-1, :]  # (1, L, D)
            all_embs.append(emb)

        return _concat_with_padding(all_embs)

    #: Alias so ``backend.get_embeddings(seqs)`` works alongside the canonical name.
    get_embeddings = get_sequence_embeddings


# ---------------------------------------------------------------------------
# Profluent E1 backend
# ---------------------------------------------------------------------------


class E1Backend:
    """Wraps the Profluent E1 masked encoder family (E1-150m/300m/600m)."""

    family: str = "e1"

    def __init__(self, model: torch.nn.Module, batch_preparer, model_name: str) -> None:
        self.model = model
        self._batch_preparer = batch_preparer
        self.alphabet = None   # no alphabet concept — kept for interface parity
        self._model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str, device: torch.device) -> "E1Backend":
        from E1.batch_preparer import E1BatchPreparer    # type: ignore[import]
        from E1.modeling import E1ForMaskedLM            # type: ignore[import]

        key = model_name.lower().replace("-", "_")
        repo_id = _E1_REPOS.get(key)
        if repo_id is None:
            raise ValueError(
                f"Unknown E1 model '{model_name}'. "
                f"Valid keys: {list(_E1_REPOS)}"
            )
        print(f"Loading Profluent E1 model {key} from {repo_id}...")
        model = E1ForMaskedLM.from_pretrained(repo_id)
        model = model.to(device).eval()

        # E1 uses flex_attention which requires Triton for compilation.
        # On systems without Triton (e.g. this HPC env), disable dynamo so
        # PyTorch falls back to the eager SDPA implementation.
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.disable = True
        except Exception:
            pass

        batch_preparer = E1BatchPreparer()
        return cls(model, batch_preparer, model_name)

    def get_sequence_embeddings(
        self,
        sequences: list[str],
        layer: int | None = None,   # ignored; E1 always returns final-layer
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return ``(N, L, D)`` float32 numpy array (boundary tokens stripped)."""
        device = next(self.model.parameters()).device
        all_embs: list[torch.Tensor] = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch = self._batch_preparer.get_batch_kwargs(batch_seqs, device=str(device))

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    within_seq_position_ids=batch["within_seq_position_ids"],
                    global_position_ids=batch["global_position_ids"],
                    sequence_ids=batch["sequence_ids"],
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            # outputs.embeddings: (B, L_padded, D)
            # For single-sequence inputs, last_sequence == only sequence.
            embeddings = outputs.embeddings.float().cpu()   # (B, L_padded, D)
            seq_ids    = batch["sequence_ids"].cpu()        # (B, L_padded)

            # residue_selector: True for amino-acid tokens (not BOS/EOS/PAD)
            residue_mask = ~(
                self._batch_preparer
                .get_boundary_token_mask(batch["input_ids"])
                .cpu()
            )  # (B, L_padded)

            for j in range(embeddings.shape[0]):
                # For single-sequence input the last (only) sequence id is 0
                last_seq_id = seq_ids[j].max().item()
                last_seq_mask = seq_ids[j] == last_seq_id   # (L_padded,)
                sel = last_seq_mask & residue_mask[j]        # (L_padded,)
                emb = embeddings[j][sel].unsqueeze(0)        # (1, seq_len, D)
                all_embs.append(emb)

        return _concat_with_padding(all_embs)

    #: Alias so ``backend.get_embeddings(seqs)`` works alongside the canonical name.
    get_embeddings = get_sequence_embeddings


# ---------------------------------------------------------------------------
# Ankh backend
# ---------------------------------------------------------------------------


class AnkhBackend:
    """Wraps the ``ankh`` T5-encoder protein language model API."""

    family: str = "ankh"

    def __init__(self, model: torch.nn.Module, tokenizer, model_name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.alphabet = None   # no alphabet concept — kept for interface parity
        self._model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str, device: torch.device) -> "AnkhBackend":
        import ankh as ankh_lib  # type: ignore[import]

        key = model_name.lower().replace("-", "_")
        loader_name = _ANKH_LOADERS.get(key)
        if loader_name is None:
            raise ValueError(
                f"Unknown Ankh model '{model_name}'. "
                f"Valid keys: {list(_ANKH_LOADERS)}"
            )
        if not hasattr(ankh_lib, loader_name):
            available = [k for k, fn in _ANKH_LOADERS.items() if hasattr(ankh_lib, fn)]
            raise ValueError(
                f"Ankh model '{key}' requires loader '{loader_name}' which is not "
                f"present in the installed ankh version ({ankh_lib.__version__!r}). "
                f"Available models: {available}. Try upgrading ankh."
            )
        loader = getattr(ankh_lib, loader_name)
        print(f"Loading Ankh model {key}...")
        model, tokenizer = loader()
        model = model.to(device).float().eval()
        return cls(model, tokenizer, model_name)

    def num_layers(self) -> int:
        """Number of encoder layers (hidden_states has this + 1 entries:
        index 0 is the embedding output, index num_layers is the final
        layer -- same as last_hidden_state)."""
        return int(self.model.config.num_layers)

    def get_sequence_embeddings(
        self,
        sequences: list[str],
        layer: int | None = None,   # None = final layer (last_hidden_state);
                                     # otherwise index into hidden_states,
                                     # 0=embedding output .. num_layers=final
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return ``(N, L, D)`` float32 numpy array (EOS/padding stripped)."""
        device = next(self.model.parameters()).device
        all_embs: list[torch.Tensor] = []
        want_hidden_states = layer is not None

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            # Ankh tokenizer expects a list-of-lists (char-level words)
            split_seqs = [list(seq) for seq in batch_seqs]
            inputs = self.tokenizer(
                split_seqs,
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            input_ids      = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=want_hidden_states,
                )
            if want_hidden_states:
                n_hidden = len(out.hidden_states)
                if not (0 <= layer < n_hidden):
                    raise ValueError(
                        f"layer={layer} out of range for {self._model_name} "
                        f"(hidden_states has {n_hidden} entries: 0=embedding "
                        f"output .. {n_hidden - 1}=final layer)"
                    )
                # T5EncoderModel hidden_states[i]: (batch, L+1, D), +1 = trailing EOS
                hidden = out.hidden_states[layer].float().cpu()
            else:
                # last_hidden_state == hidden_states[-1]; kept as a separate
                # path so the unlayered call (the common case) doesn't pay
                # for output_hidden_states=True's extra memory.
                hidden = out.last_hidden_state.float().cpu()  # (batch, L+1, D)

            # Strip EOS (last real token per sequence) and collect individually
            # so _concat_with_padding can pad to a uniform length.
            for j in range(hidden.shape[0]):
                seq_len = int(attention_mask[j].sum().item()) - 1  # exclude EOS
                emb = hidden[j, :seq_len, :].unsqueeze(0)          # (1, L, D)
                all_embs.append(emb)

        return _concat_with_padding(all_embs)

    #: Alias so ``backend.get_embeddings(seqs)`` works alongside the canonical name.
    get_embeddings = get_sequence_embeddings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_esm_backend(
    model_name: str,
    device: torch.device | None = None,
) -> "ESM2Backend | ESMCBackend | AnkhBackend | E1Backend":
    """Return the correct backend for *model_name*.

    Args:
        model_name: e.g. ``'esm2_t33_650M_UR50D'``, ``'esmc_300m'``,
            ``'ankh_base'``, ``'ankh_large'``, ``'e1_300m'``, etc.
        device: PyTorch device to load the model onto. Defaults to
            ``cuda`` if available, otherwise ``cpu``.

    Returns:
        :class:`ESM2Backend`, :class:`ESMCBackend`, :class:`AnkhBackend`, or
        :class:`E1Backend` depending on the model name prefix.
    """
    family = infer_model_family(model_name)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if family == "esmc":
        return ESMCBackend.from_pretrained(model_name, device)
    if family == "ankh":
        return AnkhBackend.from_pretrained(model_name, device)
    if family == "e1":
        return E1Backend.from_pretrained(model_name, device)
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
