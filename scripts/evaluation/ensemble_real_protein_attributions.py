#!/usr/bin/env python3
"""
Combine the 5-seed real-protein attribution ensemble (seed=42 + seed_{1,2,3,4})
into a per-position consensus attribution with a cross-seed reliability ratio.

For each (config, arch) combo, loads the 5 seeds' raw per-variant
*_integrated_gradients_results.csv (full precision -- the position-level summary
csv rounds to 4 decimals, which causes spurious seed agreement for small effects),
aligns on position, and computes:
  - ensemble_mean_gradient   : mean over seeds of mean_gradient (signed IG, linear avg)
  - ensemble_mean_abs_gradient: mean over seeds of mean_abs_gradient (magnitude, linear avg)
  - seed_std                 : std over seeds of mean_gradient (cross-seed disagreement,
                                distinct from each seed's own std_gradient which is
                                within-seed, across-variant)
  - reliability_ratio        : |ensemble_mean_gradient| / (seed_std + eps)
                                high = seeds agree (robust signal), low = seed-sensitive

Usage:
    python scripts/evaluation/ensemble_real_protein_attributions.py
    python scripts/evaluation/ensemble_real_protein_attributions.py --root results/FLIP2 results/ProteinGym
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

SEEDS = ["42", "1", "2", "3", "4"]  # "42" = base dir (no seed_* subfolder)
EPS = 1e-8


def _find_base_and_seed_dirs(combo_dir: str) -> dict[str, str]:
    """Map seed label -> directory containing that seed's summary csv."""
    dirs = {"42": combo_dir}
    for s in ("1", "2", "3", "4"):
        d = os.path.join(combo_dir, f"seed_{s}")
        if os.path.isdir(d):
            dirs[s] = d
    return dirs


def _load_seed_summary(combo_dir: str, seed_dir: str) -> pd.DataFrame | None:
    # The *_position_importance_summary.csv rounds mean_gradient to 4 decimals
    # (src/igloo/utils.py), which causes spurious seed_std=0 collisions for
    # small-effect positions and blows up reliability_ratio. Recompute
    # mean_gradient/mean_abs_gradient at full precision from the raw
    # per-variant *_integrated_gradients_results.csv instead.
    candidates = glob.glob(os.path.join(seed_dir, "*_integrated_gradients_results.csv"))
    if not candidates:
        return None
    path = candidates[0]
    raw = pd.read_csv(path, usecols=["position", "integrated_gradient"])
    grouped = raw.groupby("position")["integrated_gradient"].agg(
        mean_gradient="mean", mean_abs_gradient=lambda s: s.abs().mean()
    )
    return grouped.reset_index()


def combine_combo(combo_dir: str) -> pd.DataFrame | None:
    seed_dirs = _find_base_and_seed_dirs(combo_dir)
    if len(seed_dirs) < 2:
        return None

    per_seed = {}
    for seed, d in seed_dirs.items():
        df = _load_seed_summary(combo_dir, d)
        if df is None:
            print(f"  [warn] no summary csv for seed={seed} in {d}, skipping seed")
            continue
        per_seed[seed] = df.set_index("position")

    if len(per_seed) < 2:
        print(f"  [warn] fewer than 2 seeds with data in {combo_dir}, skipping combo")
        return None

    positions = None
    for seed, df in per_seed.items():
        positions = df.index if positions is None else positions.intersection(df.index)
    positions = positions.sort_values()

    mean_gradient_mat = np.stack(
        [per_seed[s].loc[positions, "mean_gradient"].to_numpy() for s in per_seed]
    )  # (n_seeds, n_positions)
    mean_abs_gradient_mat = np.stack(
        [per_seed[s].loc[positions, "mean_abs_gradient"].to_numpy() for s in per_seed]
    )

    ensemble_mean_gradient = mean_gradient_mat.mean(axis=0)
    ensemble_mean_abs_gradient = mean_abs_gradient_mat.mean(axis=0)
    seed_std = mean_gradient_mat.std(axis=0, ddof=1)
    reliability_ratio = np.abs(ensemble_mean_gradient) / (seed_std + EPS)

    out = pd.DataFrame({
        "position": positions,
        "ensemble_mean_gradient": ensemble_mean_gradient,
        "ensemble_mean_abs_gradient": ensemble_mean_abs_gradient,
        "seed_std": seed_std,
        "reliability_ratio": reliability_ratio,
        "n_seeds": len(per_seed),
    })
    return out.sort_values("reliability_ratio", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["results/FLIP2", "results/ProteinGym"])
    ap.add_argument("--out-suffix", default="_ensemble_attribution.csv")
    args = ap.parse_args()

    combo_dirs = []
    for root in args.roots:
        for d in sorted(glob.glob(os.path.join(root, "*_esm_perres_*"))):
            if os.path.isdir(d) and "seed_" not in os.path.basename(d):
                combo_dirs.append(d)

    print(f"found {len(combo_dirs)} config/arch combos")
    for combo_dir in combo_dirs:
        print(f"combining {combo_dir}")
        result = combine_combo(combo_dir)
        if result is None:
            continue
        out_path = os.path.join(combo_dir, os.path.basename(combo_dir.rstrip("/")) + args.out_suffix)
        result.to_csv(out_path, index=False)
        print(f"  -> {out_path} ({len(result)} positions, {result['n_seeds'].iloc[0]} seeds)")


if __name__ == "__main__":
    main()
