#!/usr/bin/env python3
"""
Token-level integrated gradients through the WHOLE ESM-C model (pilot).

Standard "embedding-level" IG (sequence_ig.py, attribution_mechanism.py's
signed_ig) treats ESM-C as a frozen, precomputed feature extractor: baseline
and target are the FINAL per-residue embeddings (after all transformer
layers), and only the small predictor MLP is differentiated. Any redistribution
of a mutation's effect that ESM-C's own self-attention performs is baked into
the (fixed) input Δx and is invisible to that gradient.

This variant interpolates at the TOKEN-embedding layer (client.embed's output,
BEFORE any transformer layer) and runs the interpolated embedding through the
*entire* ESM-C transformer stack at every integration step, so ∂F/∂x reflects
the pLM's own contextualization, not just the predictor head.

Verified against the real (fixed, 2026-09-15) ESM-C source:
  client.embed        -- plain nn.Embedding(64, d_model); baseline/target are
                          its output rows at the mutated token id(s)
  client.transformer   -- TransformerStack.forward(x, sequence_id) takes the
                          continuous (B,L,D) embedding directly: the hook point
  client.forward()     -- no torch.no_grad() (only the high-level .logits()
                          convenience wrapper has one) -> backprop-able

Cost: a full ~300M-parameter forward+backward per integration step, vs. just
the small MLP for embedding-level IG (roughly 300-1000x per step). This script
is a PILOT: few variants, few steps, meant to validate correctness
(completeness axiom) and see whether token-level attributions actually differ
from embedding-level ones -- not to run at sweep scale.

Usage:
    python scripts/token_level_ig_esmc.py \
        --run-dir   .../GB1_additive/noise_0.0/linear \
        --model-dir .../GB1_additive/noise_0.0 \
        --n-variants 5 --steps 20
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.igloo.utils import create_model
from src.igloo.data_processing import set_global_normalization_params
from src.igloo import data_processing as _dp
from scripts.attribution_mechanism import sample_variants, emb_of, signed_ig

warnings.filterwarnings("ignore", category=RuntimeWarning)
_GT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def load_model(model_dir: Path, config: dict, device):
    name = config["data"]["name"]
    ckpt = torch.load(model_dir / f"{name}_training_model.pth",
                      map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    model = create_model({**mc, **config["model"], "input_dim": mc["input_dim"],
                          "L": mc["L"]}, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    npar = ckpt["normalization_params"]
    set_global_normalization_params(npar["mean"], npar["scale"])
    return model


def _normalize(emb: torch.Tensor) -> torch.Tensor:
    """Same global (mean, scale) normalization the predictor was trained under."""
    b, l, d = emb.shape
    flat = emb.reshape(b, -1)
    flat = (flat - _dp._global_scaler_mean) / _dp._global_scaler_scale
    return flat.reshape(b, l, d)


def load_esmc(device):
    from esm.models.esmc import ESMC
    client = ESMC.from_pretrained("esmc_300m").to(device).float()
    client.eval()
    for p in client.parameters():
        p.requires_grad_(False)
    return client


def token_embeddings(client, seq: str, device):
    """Tokenize + look up token embeddings (WITH BOS/EOS, ids 0 and 2).
    Returns (tokens (1, L+2) long, embedding (1, L+2, D) float)."""
    tokens = client._tokenize([seq]).to(device)
    with torch.no_grad():
        emb = client.embed(tokens)
    return tokens, emb


def predict_from_token_embedding(client, predictor_model, x, tokens):
    """token embedding (1,L+2,D) -> full transformer -> strip BOS/EOS -> global
    normalize -> predictor MLP. Mirrors the fixed ESMCBackend convention
    (embedding.py / esm_adapter.py) exactly, so results are comparable to the
    embedding-level pipeline."""
    sequence_id = tokens != client.tokenizer.pad_token_id
    post, _, _ = client.transformer(x, sequence_id=sequence_id)   # (1, L+2, D)
    stripped = post[:, 1:-1, :]                                    # (1, L, D) -- drop BOS/EOS
    xn = _normalize(stripped)
    mask = torch.ones(1, xn.shape[1], device=x.device)
    return predictor_model(xn, mask)


def token_level_signed_ig(client, predictor_model, wt_seq: str, var_seq: str, steps: int, device):
    """Full trapezoidal signed IG, interpolated at the TOKEN-embedding layer,
    with a full ESM-C transformer forward+backward at every step. Returns
    (signed_ig (L,) BOS/EOS-stripped, f0, f1, completeness_pred, completeness_true)."""
    tok0, emb0 = token_embeddings(client, wt_seq, device)
    tok1, emb1 = token_embeddings(client, var_seq, device)
    assert tok0.shape == tok1.shape, (
        "token-level IG pilot assumes point substitutions only (same token count); "
        f"got {tok0.shape} vs {tok1.shape}"
    )

    diff = (emb1 - emb0).squeeze(0)                    # (L+2, D) -- ~0 away from mutated position(s)
    Lp2, D = diff.shape
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    tw = torch.ones(steps + 1, device=device)
    tw[0] = tw[-1] = 0.5
    acc = torch.zeros(Lp2, D, device=device)

    with torch.no_grad():
        f0 = predict_from_token_embedding(client, predictor_model, emb0, tok0).item()
        f1 = predict_from_token_embedding(client, predictor_model, emb1, tok1).item()

    for a, w in zip(alphas, tw):
        x = (emb0 + a * (emb1 - emb0)).detach().requires_grad_(True)
        out = predict_from_token_embedding(client, predictor_model, x, tok0)
        g = torch.autograd.grad(out.sum(), x, retain_graph=False)[0]
        acc += w * g.squeeze(0).detach()
    acc /= steps

    signed_full = (acc * diff).sum(dim=-1)             # (L+2,) incl. BOS/EOS
    signed = signed_full[1:-1].cpu().numpy()             # (L,)
    completeness_pred = float(signed_full.sum().item())
    completeness_true = f1 - f0
    return signed, f0, f1, completeness_pred, completeness_true


def main():
    project_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--n-variants", type=int, default=5, help="pilot scale -- keep small")
    ap.add_argument("--steps", type=int, default=20, help="pilot scale -- keep small")
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "token_level_ig")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(sorted(args.run_dir.glob("*_config_snapshot_*.yaml"))[-1].read_text())
    protein = cfg["data"]["name"].split("_")[0]
    w = np.asarray(json.loads(
        (_GT_DIR / f"{protein}_ground_truth_weights.json").read_text())["importance_vector"],
        dtype=float)
    wt = cfg["data"]["wildtype"]["sequence"]
    layer = cfg["model"]["esm_layer"]
    subdir = cfg.get("integrated_gradients", {}).get("embeddings_subdir") or cfg.get("embeddings_subdir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = len(w)
    tag = cfg["data"]["name"] + "__" + args.model_dir.name
    print(f"{tag}  device={device}  L={L}  steps={args.steps}  n_variants={args.n_variants}")

    print("loading predictor model + ESM-C client ...")
    model = load_model(args.model_dir, cfg, device)
    client = load_esmc(device)

    x_wt_final = torch.from_numpy(emb_of(wt, layer, subdir)).float().to(device)

    seqs = sample_variants(args.run_dir, wt, args.n_variants)
    print(f"{len(seqs)} variants sampled")

    rows = []
    for k, s in enumerate(seqs):
        t0 = time.perf_counter()

        # --- embedding-level IG (existing method, precomputed final embeddings) ---
        x1_final = torch.from_numpy(emb_of(s, layer, subdir)).float().to(device)
        signed_emb, gmag_emb, dxmag_emb, _ = signed_ig(model, x_wt_final, x1_final, args.steps)
        with torch.no_grad():
            mask = torch.ones(1, L, device=device)
            f0_emb = model(_normalize(x_wt_final), mask).item()
            f1_emb = model(_normalize(x1_final), mask).item()
        comp_emb_pred = float(signed_emb.sum())
        comp_emb_true = f1_emb - f0_emb

        # --- token-level IG (new: full pLM backprop) ---
        try:
            signed_tok, f0_tok, f1_tok, comp_tok_pred, comp_tok_true = token_level_signed_ig(
                client, model, wt, s, args.steps, device)
        except AssertionError as e:
            print(f"  [{k+1}/{len(seqs)}] skip ({e})")
            continue

        elapsed = time.perf_counter() - t0
        r_emb_tok = pearsonr(signed_emb[:L], signed_tok[:L])[0]
        rho_emb_tok = spearmanr(signed_emb[:L], signed_tok[:L])[0]
        rho_emb_w = spearmanr(np.abs(signed_emb[:L]), w)[0]
        rho_tok_w = spearmanr(np.abs(signed_tok[:L]), w)[0]

        print(f"  [{k+1}/{len(seqs)}] {elapsed:.1f}s  "
              f"completeness_err(emb)={abs(comp_emb_pred-comp_emb_true):.4f}  "
              f"completeness_err(tok)={abs(comp_tok_pred-comp_tok_true):.4f}  "
              f"r(emb,tok)={r_emb_tok:+.3f}  rho(emb,tok)={rho_emb_tok:+.3f}  "
              f"rho(emb,w)={rho_emb_w:+.3f}  rho(tok,w)={rho_tok_w:+.3f}")

        rows.append({
            "variant": s, "elapsed_s": elapsed,
            "f0_emb": f0_emb, "f1_emb": f1_emb,
            "completeness_pred_emb": comp_emb_pred, "completeness_true_emb": comp_emb_true,
            "completeness_err_emb": abs(comp_emb_pred - comp_emb_true),
            "f0_tok": f0_tok, "f1_tok": f1_tok,
            "completeness_pred_tok": comp_tok_pred, "completeness_true_tok": comp_tok_true,
            "completeness_err_tok": abs(comp_tok_pred - comp_tok_true),
            "pearson_emb_vs_tok": r_emb_tok, "spearman_emb_vs_tok": rho_emb_tok,
            "spearman_emb_vs_w": rho_emb_w, "spearman_tok_vs_w": rho_tok_w,
        })

    if not rows:
        print("no variants succeeded"); return

    df = pd.DataFrame(rows)
    out_csv = args.out / f"token_level_ig_{tag}.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== SUMMARY (mean across variants) ===")
    print(f"  completeness error   embedding-level: {df.completeness_err_emb.mean():.4f}   "
          f"token-level: {df.completeness_err_tok.mean():.4f}")
    print(f"  agreement (embedding-level vs token-level attribution):  "
          f"pearson={df.pearson_emb_vs_tok.mean():+.3f}  spearman={df.spearman_emb_vs_tok.mean():+.3f}")
    print(f"  vs ground truth w:   embedding-level rho={df.spearman_emb_vs_w.mean():+.3f}   "
          f"token-level rho={df.spearman_tok_vs_w.mean():+.3f}")
    print(f"  mean time per variant (token-level, {args.steps} steps): {df.elapsed_s.mean():.1f}s")
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
