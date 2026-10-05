#!/usr/bin/env python3
"""
Verify the mechanism: on ESMAttention1d, everything after the attention pool
operates on a single 960-vector, so

    dF/dx_i  ≈  attn_i · (dF/dpooled)
               └──┬──┘   └────┬─────┘
        attention weight   ONE vector, identical for every position (the MLP)

Consequences we check, over a sample of test variants (per run):
  1. corr( attn_i , mean_v ||IG_i(v)|| )                 -> should be ~1
  2. corr( attn_i * mean_v ||dx_i(v)|| , mean_v signed_IG_i(v) )  -> should be ~1
  3. corr( mean signed_IG_i , mean ||IG_i|| )            -> gradient magnitude ~ ranking
  4. corr( attn_i(original) , attn_i(MLP-randomized) )   -> should be exactly ~1
     (attention weights don't change when you randomize fc1..final)
  5. spearman of each of {attn, ||dx||, attn*||dx||, signed_IG} vs ground-truth w

Usage:
    python scripts/sanity_checks/attribution_mechanism.py \
        --run-dir   .../GB1_additive/noise_0.0/linear \
        --model-dir .../GB1_additive/noise_0.0 \
        [--n-variants 40] [--steps 64] [--out results/synthetic/attribution_mechanism]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.igloo.utils import create_model
from src.igloo.data_processing import set_global_normalization_params
from src.igloo import data_processing as _dp
from src.igloo.embedding import get_embeddings
from scripts.sanity_checks.cascading_randomization import randomize_module, CASCADE

warnings.filterwarnings("ignore", category=RuntimeWarning)
_GT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
# layers to randomize for the "does the MLP affect attention?" check.
# `pre` (per-residue MLP before the pool) is deliberately EXCLUDED here — it
# sits in front of the pool, so it *should* change attention/attribution on the
# premlp architecture; excluding it isolates the post-pool MLP.
_MLP_LAYERS = ["final", "fc4", "res_proj", "fc3", "fc2", "fc1", "layernorm"]


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
    b, l, d = emb.shape
    flat = emb.reshape(b, -1)
    flat = (flat - _dp._global_scaler_mean) / _dp._global_scaler_scale
    return flat.reshape(b, l, d)


def attn_weights(model, x: torch.Tensor) -> np.ndarray:
    """softmax attention weight per position for input x (1,L,D), normalised."""
    with torch.no_grad():
        xn = _normalize(x)
        if hasattr(model, "pre"):                        # premlp architecture
            xn = xn + model.pre(xn)
        logit = model.attention1d.layer(xn).view(-1)     # (L,)
        return torch.softmax(logit, dim=-1).cpu().numpy()


def signed_ig(model, x0: torch.Tensor, x1: torch.Tensor, steps: int):
    """Return (signed_ig_i (L,), grad_mag_i (L,), g = dF/dpooled (D,))."""
    device = x0.device
    L, D = x0.shape[1], x0.shape[2]
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    tw = torch.ones(steps + 1, device=device)
    tw[0] = tw[-1] = 0.5
    diff = (x1 - x0).squeeze(0)                           # (L,D)
    acc = torch.zeros(L, D, device=device)

    for a, w in zip(alphas, tw):
        x = (x0 + a * (x1 - x0)).detach().requires_grad_(True)
        xn = _normalize(x)
        mask = torch.ones(1, L, device=device)
        out = model(xn, mask)
        g_in = torch.autograd.grad(out.sum(), x, retain_graph=False)[0]
        acc += w * g_in.squeeze(0).detach()
    acc /= steps

    # g = dF/dpooled  at the actual sequence: pool manually, differentiate the MLP
    xn = _normalize(x1.detach())
    with torch.no_grad():
        xp = xn + model.pre(xn) if hasattr(model, "pre") else xn
        p = (torch.softmax(model.attention1d.layer(xp).view(1, -1), -1).unsqueeze(-1)
             * xp).sum(dim=1)                              # (1,D) = pooled
    p = p.detach().requires_grad_(True)
    y = model.final(_mlp_from_pooled(model, p))
    g = torch.autograd.grad(y.sum(), p)[0].squeeze(0).detach().cpu().numpy()

    signed = (acc * diff).sum(dim=-1).cpu().numpy()
    gmag = acc.norm(dim=-1).cpu().numpy()
    dxmag = diff.norm(dim=-1).cpu().numpy()
    return signed, gmag, dxmag, g


def _mlp_from_pooled(model, p):
    x = model.layernorm(p)
    o1 = model.drop1(model.gelu1(model.fc1(x)))
    o2 = model.drop2(model.gelu2(model.fc2(o1)))
    o3 = model.drop3(model.gelu3(model.fc3(o2)))
    o4 = model.drop4(model.gelu4(model.fc4(o3)))
    o4 = o4 + model.res_proj(o1)
    return o4


def sample_variants(run_dir: Path, wt: str, n: int):
    ig = list(run_dir.glob("*_integrated_gradients_results.csv"))[0]
    names = pd.read_csv(ig, usecols=["sequence_name"])["sequence_name"].unique()
    real = [x for x in names if "WT" not in x and "Poly" not in x and "Activity" not in x
            and "[test]" in x]
    rng = np.random.default_rng(0)
    order = real if n <= 0 else rng.permutation(real)
    out = []
    for nm in order:
        s = list(wt)
        ok = True
        for tok in nm.split("[")[0].split(";"):
            tok = tok.strip()
            try:
                w_, pos, m_ = tok[0], int(tok[1:-1]), tok[-1]
                if s[pos - 1] != w_:
                    ok = False; break
                s[pos - 1] = m_
            except (ValueError, IndexError):
                ok = False; break
        if ok:
            out.append("".join(s))
        if n > 0 and len(out) >= n:
            break
    return out


def emb_of(seq: str, layer: int, subdir):
    import hashlib
    h = hashlib.md5(seq.encode()).hexdigest()[:8]
    return get_embeddings([seq], layer=layer, batch_size=1,
                          dataset_name=f"sequence_ig_seq_{h}", subdir=subdir)


def main():
    project_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--n-variants", type=int, default=0,
                    help="0 = every analysed test variant (default); >0 = random subsample")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "attribution_mechanism")
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
    print(f"{tag}  device={device}  L={L}")

    model = load_model(args.model_dir, cfg, device)
    x_wt = torch.from_numpy(emb_of(wt, layer, subdir)).float().to(device)

    attn = attn_weights(model, x_wt)[:L]

    seqs = sample_variants(args.run_dir, wt, args.n_variants)
    print(f"{len(seqs)} variants")
    S, AbsS, G, DX, GG = [], [], [], [], []
    for k, s in enumerate(seqs):
        x1 = torch.from_numpy(emb_of(s, layer, subdir)).float().to(device)
        try:
            si, gm, dxm, g = signed_ig(model, x_wt, x1, args.steps)
        except Exception as e:
            print(f"  skip ({e})"); continue
        S.append(si[:L]); AbsS.append(np.abs(si[:L]))
        G.append(gm[:L]); DX.append(dxm[:L]); GG.append(g)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(seqs)}")
    n_used = len(S)
    S = np.mean(S, 0); AbsS = np.mean(AbsS, 0)
    G = np.mean(G, 0); DX = np.mean(DX, 0)
    GG = np.stack(GG)  # (n, D)

    # MLP-randomized attention (should be identical)
    model_rand = load_model(args.model_dir, cfg, device)
    for ln in _MLP_LAYERS:
        if hasattr(model_rand, ln):
            randomize_module(getattr(model_rand, ln))
    attn_rand = attn_weights(model_rand, x_wt)[:L]

    def _c(a, b):
        return pearsonr(a, b)[0], spearmanr(a, b)[0]

    rows = {
        "tag": tag, "n_variants": n_used,
        "pearson(attn, mean|IG|)": _c(attn, G)[0],
        "spearman(attn, mean|IG|)": _c(attn, G)[1],
        "pearson(attn*meandx, mean|signedIG|)": _c(attn * DX, AbsS)[0],
        "spearman(attn*meandx, mean|signedIG|)": _c(attn * DX, AbsS)[1],
        "spearman(mean|signedIG|, mean|IG|)": _c(AbsS, G)[1],
        "spearman(attn_orig, attn_MLPrandom)": _c(attn, attn_rand)[1],
        "pearson(attn_orig, attn_MLPrandom)": _c(attn, attn_rand)[0],
        "g_direction_cos_dispersion": float(np.mean([
            1 - abs(np.dot(GG[i] / (np.linalg.norm(GG[i]) + 1e-9),
                           GG[j] / (np.linalg.norm(GG[j]) + 1e-9)))
            for i in range(len(GG)) for j in range(i + 1, len(GG))])) if len(GG) > 1 else np.nan,
        "spearman(attn, w)": _c(attn, w)[1],
        "spearman(|dx|, w)": _c(DX, w)[1],
        "spearman(attn*|dx|, w)": _c(attn * DX, w)[1],
        "spearman(signedIG, w)": _c(S, w)[1],
    }
    df = pd.DataFrame([rows])
    df.to_csv(args.out / f"mechanism_{tag}.csv", index=False)
    print(df.T.to_string(header=False))

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for ax, (a, b, xl, yl) in zip(axes, [
        (attn, G, "attention weight attn_i", "mean |IG_i|"),
        (attn * DX, AbsS, "attn_i · mean||Δx_i||", "mean |signed IG_i|"),
        (attn, attn_rand, "attn_i (trained MLP)", "attn_i (MLP randomized)"),
    ]):
        ax.scatter(a, b, s=22, alpha=0.7)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"Pearson r = {pearsonr(a, b)[0]:+.3f}", fontsize=10)
    fig.suptitle(f"Attribution mechanism — {tag}\n"
                 "per-position attribution ≈ attention weight × embedding displacement; "
                 "MLP has no per-position effect", fontsize=12, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(args.out / f"mechanism_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"saved {args.out.name}/mechanism_{tag}.png")


if __name__ == "__main__":
    main()
