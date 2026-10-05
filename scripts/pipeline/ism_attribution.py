#!/usr/bin/env python3
"""
In-silico mutagenesis (ISM) as a model-DEPENDENT attribution baseline.

For every position i, substitute all 19 other amino acids, predict with the
trained model, and take ISM_i = mean_a | pred(WT: i->a) - pred(WT) |.  This
bypasses the gradient pathology (see [[path2-plm-attribution-investigation]]):
it directly measures how much the model's *prediction* depends on position i.

Per run: writes an ISM position vector + its Spearman / NDCG / AUROC vs the
ground-truth importance vector, and the model's own test Spearman (from the run
log).  Aggregate across runs with `ism_attribution_analysis.py`.

Usage:
    python scripts/pipeline/ism_attribution.py \
        --run-dir   .../GB1_additive/noise_0.0/linear \
        --model-dir .../GB1_additive/noise_0.0 \
        [--out results/synthetic/ism_attribution]
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.igloo.utils import create_model
from src.igloo.mutagenesis import compute_mutation_effects

warnings.filterwarnings("ignore", category=RuntimeWarning)
_GT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
_SP_RE = re.compile(r"Spearman\s*[ρr]\s*=\s*(-?\d+\.\d+)")


def load_model(model_dir: Path, config: dict, device):
    name = config["data"]["name"]
    ckpt = torch.load(model_dir / f"{name}_training_model.pth",
                      map_location=device, weights_only=False)
    mc = ckpt["model_config"]
    model = create_model({**mc, **config["model"], "input_dim": mc["input_dim"],
                          "L": mc["L"]}, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def model_test_spearman(run_dir: Path) -> float:
    logs = sorted(run_dir.glob("*_run_log_*.txt"))
    if not logs:
        return np.nan
    m = _SP_RE.findall(logs[-1].read_text(errors="ignore"))
    return float(m[-1]) if m else np.nan


def main():
    project_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "ism_attribution")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(sorted(args.run_dir.glob("*_config_snapshot_*.yaml"))[-1].read_text())
    protein = cfg["data"]["name"].split("_")[0]
    w = np.asarray(json.loads(
        (_GT_DIR / f"{protein}_ground_truth_weights.json").read_text())["importance_vector"],
        dtype=float)
    wt = cfg["data"]["wildtype"]["sequence"]
    layer = cfg["model"]["esm_layer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _noise_cfg = cfg.get("training", {}).get("noise", {})
    if str(_noise_cfg.get("label_noise_mode", "gaussian")).lower() == "blend":
        noise_v = float(_noise_cfg.get("label_noise_level", 0.0))
    else:
        noise_v = float(_noise_cfg.get("label_noise_std", 0.0))
    if "shuffled" in str(args.model_dir):
        tag = f"{cfg['data']['name']}__shuffled__{args.model_dir.name}"
    else:
        tag = f"{cfg['data']['name']}__noise_{noise_v:g}"
        if args.model_dir.name.startswith("seed_"):
            tag += f"__{args.model_dir.name}"          # seed-sweep run
    print(f"{tag}  device={device}  L={len(w)}  ({len(wt)*19} mutants)")

    model, _ = load_model(args.model_dir, cfg, device)
    mdf = compute_mutation_effects(
        model, wt, device, esm_layer=layer,
        pooling_method=cfg["model"]["pooling_method"],
        batch_size=64, progress=False,
        dataset_name=f"ism_{cfg['data']['name']}",
        embeddings_subdir=cfg.get("integrated_gradients", {}).get("embeddings_subdir")
        or cfg.get("embeddings_subdir"),
        esm_model_name=cfg["model"].get("esm_model_name", "esmc_300m"),
    )
    mdf.to_csv(args.out / f"ism_mutations_{tag}.csv", index=False)

    g = mdf.groupby("position")["effect_vs_wt"]
    pos = g.apply(lambda s: s.abs().mean()).sort_index()          # ISM_i (mean |Δ|)
    pos_max = g.apply(lambda s: s.abs().max()).sort_index()       # ISM_i (max |Δ|)
    pos_signed = g.mean().sort_index()                            # signed mean Δ
    L = min(len(pos), len(w))
    a, amax, w_ = pos.to_numpy()[:L], pos_max.to_numpy()[:L], w[:L]
    ann = w_ > (w_.min() + 1e-9)

    def _q(v):
        sp = spearmanr(v, w_)[0]
        nd = ndcg_score(w_[None, :], v[None, :])
        au = roc_auc_score(ann, v) if 0 < ann.sum() < L else np.nan
        return sp, nd, au

    sp_mean, nd_mean, au_mean = _q(a)
    sp_max, nd_max, au_max = _q(amax)

    row = {
        "tag": tag, "protein": protein,
        "landscape": cfg["data"]["name"].split("_", 1)[1],
        "noise": noise_v,
        "model_test_spearman": model_test_spearman(args.run_dir),
        "ism_spearman_vs_w": sp_mean, "ism_ndcg": nd_mean, "ism_auroc": au_mean,
        "ism_spearman_vs_w_max": sp_max, "ism_ndcg_max": nd_max,
        "n_positions": L,
    }
    pd.DataFrame([row]).to_csv(args.out / f"ism_summary_{tag}.csv", index=False)
    pd.DataFrame({"position": np.arange(1, L + 1), "ism_mean_abs": a,
                  "ism_max_abs": amax, "ism_signed_mean": pos_signed.to_numpy()[:L],
                  "gt_w": w_}).to_csv(args.out / f"ism_position_{tag}.csv", index=False)

    print(json.dumps(row, indent=2, default=float))


if __name__ == "__main__":
    main()
