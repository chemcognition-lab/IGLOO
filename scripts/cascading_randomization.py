#!/usr/bin/env python3
"""
Path-2: cascading model-parameter randomization (Adebayo et al. sanity check).

Progressively randomize the predictor's layers from the OUTPUT end toward the
INPUT end, recomputing integrated-gradients attributions at each step.  If the
attribution barely changes as the model is destroyed, it is not driven by the
learned predictor (→ it is pLM-embedding geometry).

Cascade order (ESMAttention1d): final -> fc4(+res_proj) -> fc3 -> fc2 -> fc1
-> layernorm -> attention1d.

For each cascade level we report, over a sample of test variants:
  spearman_vs_original : Spearman(agg |attr|_level, agg |attr|_level0)
  spearman_vs_gt       : Spearman(agg |attr|_level, ground-truth importance w)

Usage:
    python scripts/cascading_randomization.py \
        --run-dir /project/aip-bensl/ranaab/IGLOO/results/synthetic/GB1_additive/noise_0.0/linear \
        --model-dir /project/aip-bensl/ranaab/IGLOO/results/synthetic/GB1_additive/noise_0.0 \
        [--n-variants 60] [--steps 100] [--out results/synthetic/cascading_randomization]
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.igloo.utils import create_model
from src.igloo.data_processing import set_global_normalization_params
from src.igloo.sequence_ig import analyze_sequence_with_integrated_gradients

warnings.filterwarnings("ignore", category=RuntimeWarning)

# output-end -> input-end.  `pre` (per-residue MLP before the pool) only exists
# on ESMAttention1dPreMLP; randomize_module silently skips missing attrs.
CASCADE = [
    ("final", ["final"]),
    ("+fc4", ["fc4", "res_proj"]),
    ("+fc3", ["fc3"]),
    ("+fc2", ["fc2"]),
    ("+fc1", ["fc1"]),
    ("+layernorm", ["layernorm"]),
    ("+attention1d", ["attention1d"]),
    ("+pre", ["pre"]),
]
_GT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


class _Shim:
    """Minimal wrapper so analyze_sequence_with_integrated_gradients finds
    .predictor_model / .pooling_method / .esm_layer."""
    def __init__(self, predictor, pooling_method, esm_layer):
        self.predictor_model = predictor
        self.pooling_method = pooling_method
        self.esm_layer = esm_layer


def randomize_module(m: torch.nn.Module) -> None:
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)):
            torch.nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                torch.nn.init.zeros_(mod.bias)
        elif isinstance(mod, torch.nn.LayerNorm):
            torch.nn.init.ones_(mod.weight)
            torch.nn.init.zeros_(mod.bias)


def load_predictor(model_dir: Path, config: dict, device) -> tuple:
    name = config["data"]["name"]
    ckpt = torch.load(model_dir / f"{name}_training_model.pth",
                      map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    model = create_model({**mc, **config["model"], "input_dim": mc["input_dim"],
                          "L": mc["L"]}, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    np_ = ckpt["normalization_params"]
    set_global_normalization_params(np_["mean"], np_["scale"])
    return model, ckpt.get("pooling_method", config["model"]["pooling_method"]), \
        ckpt.get("esm_layer", config["model"]["esm_layer"])


def sample_variants(run_dir: Path, wt_seq: str, n: int, seed: int = 0) -> list[str]:
    ig = list(run_dir.glob("*_integrated_gradients_results.csv"))[0]
    names = pd.read_csv(ig, usecols=["sequence_name"])["sequence_name"].unique()
    real = [n_ for n_ in names
            if "WT" not in n_ and "Poly" not in n_ and "Activity" not in n_
            and "[test]" in n_]
    if len(real) < n:
        real = [n_ for n_ in names if "WT" not in n_ and "Poly" not in n_
                and "Activity" not in n_]
    rng = np.random.default_rng(seed)
    picks = rng.choice(real, size=min(n, len(real)), replace=False)
    # variant name -> sequence by applying the mutations to WT
    seqs = []
    for nm in picks:
        s = list(wt_seq)
        ok = True
        for tok in nm.split("[")[0].split(";"):
            tok = tok.strip()
            try:
                wt_aa, pos, mut_aa = tok[0], int(tok[1:-1]), tok[-1]
                if s[pos - 1] != wt_aa:
                    ok = False
                    break
                s[pos - 1] = mut_aa
            except (ValueError, IndexError):
                ok = False
                break
        if ok:
            seqs.append("".join(s))
    return seqs


def agg_attr(model_shim, seqs: list[str], wt_seq: str, device, steps: int,
             embeddings_subdir: str | None) -> np.ndarray:
    """Mean over variants of |position_gradients|."""
    acc = None
    used = 0
    for s in seqs:
        try:
            res = analyze_sequence_with_integrated_gradients(
                sequence=s, baseline_sequence=wt_seq, model=model_shim,
                alphabet=None, device=device, steps=steps, batch_size=10,
                method="embedding", attribution_method="signed",
                sequence_name="v", baseline_name="WT",
                interpolation_path="linear", embeddings_subdir=embeddings_subdir,
            )
        except Exception as e:
            print(f"    skip variant ({e})")
            continue
        pg = np.abs(np.asarray(res["position_gradients"], dtype=float))
        acc = pg if acc is None else acc + pg
        used += 1
    if acc is None:
        raise RuntimeError("no variants succeeded")
    print(f"    aggregated over {used}/{len(seqs)} variants")
    return acc / used


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="the .../linear/ dir with the IG results + config snapshot")
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="dir with {name}_training_model.pth (usually run-dir's parent)")
    ap.add_argument("--n-variants", type=int, default=60)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "cascading_randomization")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    snaps = sorted(args.run_dir.glob("*_config_snapshot_*.yaml"))
    config = yaml.safe_load(snaps[-1].read_text())
    protein_key = config["data"]["name"].split("_")[0]
    protein = {"GB1": "GB1", "T4Lysozyme": "T4Lysozyme",
               "p53": "p53"}.get(protein_key, protein_key)
    w = np.asarray(json.loads(
        (_GT_DIR / f"{protein}_ground_truth_weights.json").read_text())["importance_vector"],
        dtype=float)
    wt_seq = config["data"]["wildtype"]["sequence"]
    emb_subdir = config.get("integrated_gradients", {}).get("embeddings_subdir") \
        or config.get("embeddings_subdir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  protein={protein}  L={len(w)}")

    seqs = sample_variants(args.run_dir, wt_seq, args.n_variants)
    print(f"sampled {len(seqs)} test variants")

    tag = config["data"]["name"] + "__" + args.run_dir.parent.name

    records = []
    ref = None
    for level_idx, (label, _) in enumerate([("original", [])] + CASCADE):
        model, pooling, esm_layer = load_predictor(args.model_dir, config, device)
        randomized = []
        for lbl, mods in CASCADE[:level_idx]:
            for mn in mods:
                if hasattr(model, mn):
                    randomize_module(getattr(model, mn))
                    randomized.append(mn)
        shim = _Shim(model, pooling, esm_layer)
        print(f"\n[{level_idx}] {label}  (randomized: {randomized or 'none'})")
        a = agg_attr(shim, seqs, wt_seq, device, args.steps, emb_subdir)
        a = a[: len(w)]
        if ref is None:
            ref = a
        sp_ref, _ = spearmanr(a, ref)
        sp_gt, _ = spearmanr(a, w)
        records.append({"tag": tag, "level": level_idx, "label": label,
                        "randomized": ";".join(randomized),
                        "spearman_vs_original": float(sp_ref),
                        "spearman_vs_gt": float(sp_gt)})
        print(f"    spearman vs original = {sp_ref:+.3f}   vs ground truth = {sp_gt:+.3f}")

    df = pd.DataFrame(records)
    csv_path = args.out / f"cascading_{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot(df["level"], df["spearman_vs_original"], "o-", label="Spearman vs original attribution")
    ax.plot(df["level"], df["spearman_vs_gt"], "s-", label="Spearman vs ground-truth importance")
    ax.set_xticks(df["level"])
    ax.set_xticklabels(df["label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(-0.3, 1.05)
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.set_title(f"Cascading randomization — {tag}\n"
                 "flat curve ⇒ attribution is not driven by the learned model",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    for ext in ("png", "svg"):
        fig.savefig(args.out / f"cascading_{tag}.{ext}", dpi=150)
    plt.close(fig)
    print(f"saved {args.out.name}/cascading_{tag}.png / .svg")


if __name__ == "__main__":
    main()
