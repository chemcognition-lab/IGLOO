#!/usr/bin/env python3
"""
Path-2: ablate the attribution 'factors' and see which one carries the signal.

  signed        = <grad, displacement>   (product; completeness-exact baseline)
  gradient      = || mean_k grad_k ||     (model gradient only)
  displacement  = || x1 - x0 ||           (pLM embedding displacement only; NO model)

Reads:
    <root>/factor_ablation/{Protein}_{landscape}/noise_{N}/{method}/
        *_position_importance_summary.csv   -> per-position attribution
        *_run_log_*.txt                     -> model test Spearman (signed/gradient)
    data/synthetic/{Protein}_ground_truth_weights.json -> importance_vector

Produces, per (protein, landscape):
  - attribution quality (Spearman |a| vs w, NDCG, AUROC) for each method vs noise
  - the key comparison: does `displacement` alone match `signed`?
  - correlation of each method's aggregate vector WITH the `signed` vector
    (how much of the product is explained by displacement alone)

Usage:
    python scripts/factor_ablation_analysis.py \
        [--root /project/aip-bensl/ranaab/IGLOO/results/synthetic] \
        [--out  results/synthetic/factor_ablation_analysis]
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ndcg_score, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except ImportError:
    pass

PROTEINS = ["GB1", "T4Lysozyme"]
LANDSCAPES = ["additive", "epistatic", "nonlinear"]
METHODS = ["signed", "gradient", "displacement"]
METHOD_COLOR = {"signed": "#1f77b4", "gradient": "#d62728", "displacement": "#2ca02c"}
_SP_RE = re.compile(r"Spearman\s*[ρr]\s*=\s*(-?\d+\.\d+)")
_GT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def load_gt(protein: str) -> np.ndarray:
    j = json.loads((_GT_DIR / f"{protein}_ground_truth_weights.json").read_text())
    return np.asarray(j["importance_vector"], dtype=float)


def _model_perf(leaf: Path) -> float:
    logs = sorted(leaf.glob("*_run_log_*.txt"))
    if not logs:
        return np.nan
    m = _SP_RE.findall(logs[-1].read_text(errors="ignore"))
    return float(m[-1]) if m else np.nan


def _pos_vector(leaf: Path, L: int) -> np.ndarray | None:
    f = list(leaf.glob("*_position_importance_summary.csv"))
    if not f:
        return None
    v = pd.read_csv(f[0]).sort_values("position")["importance_score"].to_numpy()
    return v if len(v) == L else None


def quality(a: np.ndarray, w: np.ndarray) -> dict:
    aa = np.abs(a)
    ann = w > (w.min() + 1e-9)
    sp, _ = spearmanr(aa, w)
    nd = ndcg_score(w[None, :], aa[None, :])
    au = roc_auc_score(ann, aa) if 0 < ann.sum() < len(w) else np.nan
    return {"spearman": sp, "ndcg": nd, "auroc": au}


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for protein in PROTEINS:
        w = load_gt(protein)
        for landscape in LANDSCAPES:
            base = root / "factor_ablation" / f"{protein}_{landscape}"
            if not base.is_dir():
                continue
            for noise_dir in sorted(base.glob("noise_*")):
                noise = float(noise_dir.name.split("_", 1)[1])
                vecs = {}
                for method in METHODS:
                    v = _pos_vector(noise_dir / method, len(w))
                    if v is not None:
                        vecs[method] = v
                if not vecs:
                    continue
                signed_v = vecs.get("signed")
                for method, v in vecs.items():
                    row = {"protein": protein, "landscape": landscape,
                           "noise": noise, "method": method,
                           "model_spearman": _model_perf(noise_dir / method),
                           **quality(v, w)}
                    if signed_v is not None and method != "signed":
                        r, _ = spearmanr(np.abs(v), np.abs(signed_v))
                        row["corr_to_signed"] = r
                    rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_dir: Path) -> None:
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    if not combos:
        print("  nothing to plot yet")
        return
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))

    for metric, mlabel in [("ndcg", "NDCG"), ("spearman", "Spearman(|a|, w)"),
                           ("auroc", "AUROC")]:
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow),
                                 squeeze=False, layout="constrained")
        for ax, (p, l) in zip(axes.flat, combos):
            g = df[(df.protein == p) & (df.landscape == l)]
            for method in METHODS:
                gm = g[g.method == method].sort_values("noise")
                if gm.empty:
                    continue
                ax.plot(gm["noise"], gm[metric], "o-", color=METHOD_COLOR[method],
                        label=method)
            ax.set_title(f"{p} — {l}", fontweight="bold", fontsize=10)
            ax.set_xlabel("Label noise (std)")
            ax.set_ylabel(f"attribution {mlabel}  (vs ground truth)")
            ax.legend(fontsize=8)
        for ax in axes.flat[len(combos):]:
            ax.set_visible(False)
        fig.suptitle(f"Factor ablation — attribution {mlabel} by isolated factor",
                     fontsize=13, fontweight="bold")
        for ext in ("png", "svg"):
            fig.savefig(out_dir / f"factor_ablation_{metric}.{ext}", dpi=150)
        plt.close(fig)
        print(f"  saved {out_dir.name}/factor_ablation_{metric}.png / .svg")

    # how much of the `signed` product does `displacement` alone explain?
    sub = df[df.method.isin(["gradient", "displacement"]) & df["corr_to_signed"].notna()]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for method in ["displacement", "gradient"]:
            gm = sub[sub.method == method]
            ax.scatter(gm["noise"] + (0.01 if method == "gradient" else -0.01),
                       gm["corr_to_signed"], color=METHOD_COLOR[method], label=method, s=45)
        ax.set_xlabel("Label noise (std)")
        ax.set_ylabel("Spearman(|factor|, |signed product|)  per position")
        ax.set_title("How much of the signed product is explained by each factor alone?",
                     fontsize=11, fontweight="bold")
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.legend()
        for ext in ("png", "svg"):
            fig.savefig(out_dir / f"factor_corr_to_signed.{ext}", dpi=150)
        plt.close(fig)
        print(f"  saved {out_dir.name}/factor_corr_to_signed.png / .svg")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("/project/aip-bensl/ranaab/IGLOO/results/synthetic"))
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "factor_ablation_analysis")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.root}/factor_ablation ...")
    df = collect(args.root)
    if df.empty:
        print("No factor-ablation runs finished yet. Re-run when the fa_* jobs land.")
        return
    csv_path = args.out / "factor_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} rows)")

    piv = df.pivot_table(index=["protein", "landscape", "noise"],
                         columns="method", values="ndcg")
    print("\nAttribution NDCG by isolated factor:")
    print(piv.to_string())

    plot(df, args.out)

    # headline: mean |NDCG(displacement) - NDCG(signed)|
    d = df.pivot_table(index=["protein", "landscape", "noise"], columns="method",
                       values="ndcg")
    if {"signed", "displacement"}.issubset(d.columns):
        gap = (d["displacement"] - d["signed"]).abs().mean()
        print(f"\nMean |NDCG(displacement) - NDCG(signed)| = {gap:.3f}  "
              f"(small => the product adds little over pLM displacement alone)")


if __name__ == "__main__":
    main()
